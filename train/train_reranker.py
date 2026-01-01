# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers>=4.0.0",
#     "datasets",
#     "torch",
#     "accelerate",
#     "huggingface_hub",
#     "einops",
# ]
# ///
"""
Fine-tune jina-reranker-v2-base-multilingual for affiliation matching.

Single-phase mixed training:
- Original curated dataset (dominant)
- Small subset of hardest AffilGood cases (supplement)

Usage:
    uv run train_reranker.py --org cometadata
"""

import argparse
import logging
import random

import torch
from datasets import Dataset, load_dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import (
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_original_dataset(dataset_name: str, val_split: float = 0.05):
    """
    Load the original curated triplet dataset.

    Uses last val_split% for validation (hardest examples, preserves curriculum).

    Returns:
        train_triplets: List of triplet dicts
        val_triplets: List of triplet dicts
    """
    logger.info(f"Loading original dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    logger.info(f"Loaded {len(dataset)} triplets")

    # Split: last N% for val (hardest examples)
    split_idx = int(len(dataset) * (1 - val_split))
    train_triplets = [
        {"anchor": row["anchor"], "positive": row["positive"], "negative": row["negative"]}
        for row in dataset.select(range(split_idx))
    ]
    val_triplets = [
        {"anchor": row["anchor"], "positive": row["positive"], "negative": row["negative"]}
        for row in dataset.select(range(split_idx, len(dataset)))
    ]

    logger.info(f"Original: {len(train_triplets)} train, {len(val_triplets)} val (hardest)")
    return train_triplets, val_triplets


def load_filtered_affilgood(scored_dataset: str, margin_threshold: float, max_samples: int, seed: int = 42):
    """
    Load pre-scored AffilGood dataset filtered by margin threshold.

    Keeps only triplets where the fine-tuned model struggles (low margin).
    """
    logger.info(f"Loading scored AffilGood: {scored_dataset}")
    dataset = load_dataset(scored_dataset, split="train")
    logger.info(f"Loaded {len(dataset)} scored triplets")

    # Filter by margin threshold
    filtered = dataset.filter(lambda x: x["margin"] < margin_threshold)
    logger.info(f"Filtered to {len(filtered)} triplets with margin < {margin_threshold}")

    triplets = [
        {"anchor": row["anchor"], "positive": row["positive"], "negative": row["negative"]}
        for row in filtered
    ]

    # Downsample if requested
    if max_samples and len(triplets) > max_samples:
        random.seed(seed)
        triplets = random.sample(triplets, max_samples)
        logger.info(f"Downsampled to {len(triplets)} triplets")

    return triplets


def triplets_to_pairs(triplets: list[dict]) -> list[dict]:
    """Convert triplets to labeled pairs for training."""
    pairs = []
    for triplet in triplets:
        pairs.append({
            "query": triplet["anchor"],
            "document": triplet["positive"],
            "label": 1,
        })
        pairs.append({
            "query": triplet["anchor"],
            "document": triplet["negative"],
            "label": 0,
        })
    return pairs


def load_model(model_name: str, output_model_name: str):
    """Load the Jina reranker model for fine-tuning."""
    logger.info(f"Loading model: {model_name}")

    model = CrossEncoder(
        model_name,
        trust_remote_code=True,
        automodel_args={"torch_dtype": "auto"},
        model_card_data=CrossEncoderModelCardData(
            language="multilingual",
            license="cc-by-nc-4.0",
            model_name=output_model_name,
            model_id=output_model_name,
        ),
    )

    logger.info(f"Model max length: {model.max_length}")
    logger.info(f"Model num labels: {model.num_labels}")

    return model


def create_evaluator(val_triplets: list[dict], batch_size: int = 16):
    """Create a reranking evaluator from validation triplets."""
    # Filter out samples where positive == negative
    reranking_samples = []
    skipped = 0
    for row in val_triplets:
        if row["positive"] != row["negative"]:
            reranking_samples.append({
                "query": row["anchor"],
                "positive": [row["positive"]],
                "documents": [row["positive"], row["negative"]],
            })
        else:
            skipped += 1

    logger.info(f"Evaluator: {len(reranking_samples)} samples (skipped {skipped} identical)")

    return CrossEncoderRerankingEvaluator(
        samples=reranking_samples,
        batch_size=batch_size,
        name="affiliation-val",
    )


def create_training_args(
    output_dir: str,
    hub_model_id: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    eval_steps: int,
    push_to_hub: bool = False,
):
    """Create training arguments."""
    return CrossEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        fp16=False,
        bf16=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_affiliation-val_ndcg@10",
        logging_steps=100,
        logging_first_step=True,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        seed=42,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune reranker with mixed dataset training"
    )
    parser.add_argument(
        "--org",
        type=str,
        required=True,
        help="HuggingFace organization to push the model to",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="jina-reranker-v2-multilingual-affiliations-v4",
        help="Name for the fine-tuned model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cometadata/triplet-loss-for-embedding-affiliations-sample-1",
        help="Original curated triplet dataset",
    )
    parser.add_argument(
        "--affilgood-scored",
        type=str,
        default="cometadata/affilgood-scored",
        help="Pre-scored AffilGood dataset",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.05,
        help="Keep AffilGood triplets with margin below this (default: 0.05 = hardest cases)",
    )
    parser.add_argument(
        "--affilgood-ratio",
        type=float,
        default=0.1,
        help="AffilGood samples as fraction of original dataset (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="jinaai/jina-reranker-v2-base-multilingual",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Local directory for checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Training epochs (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Steps between evaluations (default: 500)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Fraction of original data for validation (default: 0.05)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    hub_model_id = f"{args.org}/{args.model_name}"

    logger.info("=" * 60)
    logger.info("Single-Phase Mixed Training")
    logger.info("=" * 60)
    logger.info(f"Original dataset: {args.dataset}")
    logger.info(f"AffilGood supplement: {args.affilgood_scored} (margin < {args.margin_threshold})")
    logger.info(f"AffilGood ratio: {args.affilgood_ratio:.0%} of original")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Output: {hub_model_id}")
    logger.info("=" * 60)

    # Load original dataset
    original_train, original_val = load_original_dataset(args.dataset, args.val_split)

    # Calculate AffilGood sample count (fraction of original)
    affilgood_count = int(len(original_train) * args.affilgood_ratio)
    logger.info(f"Target AffilGood samples: {affilgood_count} ({args.affilgood_ratio:.0%} of {len(original_train)})")

    # Load AffilGood subset
    affilgood_train = load_filtered_affilgood(
        args.affilgood_scored,
        args.margin_threshold,
        max_samples=affilgood_count,
    )

    # Mix datasets
    mixed_triplets = original_train + affilgood_train
    logger.info(f"Mixed dataset: {len(original_train)} original + {len(affilgood_train)} AffilGood = {len(mixed_triplets)} total")

    # Shuffle the mixed dataset
    random.seed(42)
    random.shuffle(mixed_triplets)
    logger.info("Shuffled mixed dataset")

    # Convert to pairs
    train_pairs = triplets_to_pairs(mixed_triplets)
    val_pairs = triplets_to_pairs(original_val)

    logger.info(f"Training pairs: {len(train_pairs)}")
    logger.info(f"Validation pairs: {len(val_pairs)}")

    # Create datasets
    train_dataset = Dataset.from_list(train_pairs)
    val_dataset = Dataset.from_list(val_pairs)

    # Load model
    model = load_model(args.base_model, hub_model_id)
    loss = BinaryCrossEntropyLoss(model=model)

    # Create evaluator
    evaluator = create_evaluator(original_val, batch_size=args.batch_size)

    # Evaluate base model
    logger.info("Evaluating base model...")
    base_results = evaluator(model)
    logger.info(f"Base model: {base_results}")

    # Training
    logger.info("=" * 60)
    logger.info(f"TRAINING: {args.epochs} epochs on mixed dataset")
    logger.info("=" * 60)

    training_args = create_training_args(
        output_dir=f"{args.output_dir}/mixed",
        hub_model_id=hub_model_id,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,
        push_to_hub=False,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()

    # Final evaluation
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    final_results = evaluator(model)
    logger.info(f"Final model: {final_results}")

    logger.info(f"Saving model to {args.output_dir}/final")
    model.save_pretrained(f"{args.output_dir}/final")

    logger.info(f"Pushing to Hub: {hub_model_id}")
    model.push_to_hub(hub_model_id, exist_ok=True)

    # Summary
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset mix: {len(original_train)} original + {len(affilgood_train)} AffilGood")
    logger.info(f"AffilGood filter: margin < {args.margin_threshold}")
    logger.info(f"Base model:  {base_results}")
    logger.info(f"Final model: {final_results}")
    logger.info(f"Model: https://huggingface.co/{hub_model_id}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
