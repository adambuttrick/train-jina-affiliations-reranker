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

Usage:
    uv run train_reranker.py --org cometadata --dataset cometadata/triplet-loss-for-embedding-affiliations-sample-1
"""

import argparse
import logging

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
from sentence_transformers.sampler import DefaultBatchSampler
from torch.utils.data.sampler import SequentialSampler


class SequentialBatchSampler(DefaultBatchSampler):
    """Batch sampler that preserves dataset order."""
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> None:
        sampler = SequentialSampler(dataset)
        torch.utils.data.sampler.BatchSampler.__init__(
            self, sampler, batch_size=batch_size, drop_last=drop_last
        )
        self.valid_label_columns = valid_label_columns
        self.generator = generator
        self.seed = seed

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_and_convert_dataset(dataset_name: str, val_split: float = 0.15):
    """
    Load triplet dataset and convert to labeled pairs.

    Last val_split% becomes validation set (hardest examples).

    Args:
        dataset_name: HuggingFace dataset identifier
        val_split: Fraction of data to use for validation (default 0.15)

    Returns:
        train_dataset: Dataset with query, document, label columns
        val_dataset: Dataset with query, document, label columns
        val_triplets: Original triplets for reranking evaluator
    """
    logger.info(f"Loading dataset: {dataset_name}")
    triplets = load_dataset(dataset_name, split="train")
    logger.info(f"Loaded {len(triplets)} triplets")

    # Convert triplets to labeled pairs
    # Each triplet → (anchor, positive, 1) then (anchor, negative, 0)
    pairs = []
    for row in triplets:
        pairs.append({
            "query": row["anchor"],
            "document": row["positive"],
            "label": 1,
        })
        pairs.append({
            "query": row["anchor"],
            "document": row["negative"],
            "label": 0,
        })

    logger.info(f"Converted to {len(pairs)} labeled pairs")

    # Split: first 85% for training (easy→moderate), last 15% for validation (hardest)
    split_idx = int(len(pairs) * (1 - val_split))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # Also get validation triplets for reranking evaluator
    triplet_split_idx = int(len(triplets) * (1 - val_split))
    val_triplets = triplets.select(range(triplet_split_idx, len(triplets)))

    logger.info(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    logger.info(f"Val triplets for evaluator: {len(val_triplets)}")

    train_dataset = Dataset.from_list(train_pairs)
    val_dataset = Dataset.from_list(val_pairs)

    return train_dataset, val_dataset, val_triplets


def load_model(model_name: str, output_model_name: str):
    """
    Load the Jina reranker model for fine-tuning.

    Args:
        model_name: HuggingFace model identifier
        output_model_name: Name for the fine-tuned model (for model card)

    Returns:
        CrossEncoder model ready for training
    """
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


def create_evaluator(val_triplets, batch_size: int = 16):
    """
    Create a reranking evaluator from validation triplets.

    Evaluates model's ability to rank the positive affiliation
    above the negative one.

    Args:
        val_triplets: Dataset with anchor, positive, negative columns
        batch_size: Batch size for evaluation

    Returns:
        CrossEncoderRerankingEvaluator instance
    """
    logger.info(f"Creating reranking evaluator with {len(val_triplets)} samples")
    reranking_samples = [
        {
            "query": row["anchor"],
            "positive": [row["positive"]],
            "documents": [row["positive"], row["negative"]],
        }
        for row in val_triplets
    ]

    evaluator = CrossEncoderRerankingEvaluator(
        samples=reranking_samples,
        batch_size=batch_size,
        name="affiliation-val",
    )

    return evaluator


def create_training_args(
    output_dir: str,
    hub_model_id: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    eval_steps: int = 500,
):
    """
    Create training arguments.

    Args:
        output_dir: Local directory for checkpoints
        hub_model_id: HuggingFace Hub model ID for pushing
        num_epochs: Number of training epochs
        batch_size: Training and eval batch size
        learning_rate: Learning rate
        eval_steps: Steps between evaluations

    Returns:
        CrossEncoderTrainingArguments instance
    """
    args = CrossEncoderTrainingArguments(
        output_dir=output_dir,

        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,

        fp16=False,
        bf16=True,

        batch_sampler=SequentialBatchSampler,

        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_affiliation-val_ndcg@10",

        logging_steps=100,
        logging_first_step=True,

        push_to_hub=True,
        hub_model_id=hub_model_id,

        seed=42,
    )

    logger.info(f"Training args created: {num_epochs} epochs, batch {batch_size}, lr {learning_rate}")
    logger.info("Curriculum learning enabled: SequentialBatchSampler (no shuffle)")

    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune jina-reranker-v2-base-multilingual for affiliation matching"
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
        default="jina-reranker-v2-multilingual-affiliations",
        help="Name for the fine-tuned model (default: jina-reranker-v2-multilingual-affiliations)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cometadata/triplet-loss-for-embedding-affiliations-sample-1",
        help="HuggingFace dataset to use for training",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="jinaai/jina-reranker-v2-base-multilingual",
        help="Base model to fine-tune (default: jinaai/jina-reranker-v2-base-multilingual)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Local directory for checkpoints (default: ./output)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
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
        default=0.15,
        help="Fraction of data for validation (default: 0.15)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    hub_model_id = f"{args.org}/{args.model_name}"

    logger.info("=" * 60)
    logger.info("Affiliation Reranker Training")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output: {hub_model_id}")
    logger.info("=" * 60)

    train_dataset, val_dataset, val_triplets = load_and_convert_dataset(
        args.dataset,
        val_split=args.val_split,
    )

    model = load_model(args.base_model, hub_model_id)

    loss = BinaryCrossEntropyLoss(model=model)
    logger.info("Loss function: BinaryCrossEntropyLoss")

    evaluator = create_evaluator(val_triplets, batch_size=args.batch_size)
    logger.info("Evaluating base model before training...")
    base_results = evaluator(model)
    logger.info(f"Base model results: {base_results}")

    training_args = create_training_args(
        output_dir=args.output_dir,
        hub_model_id=hub_model_id,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Final evaluation...")
    final_results = evaluator(model)
    logger.info(f"Final model results: {final_results}")

    logger.info(f"Saving model to {args.output_dir}/final")
    model.save_pretrained(f"{args.output_dir}/final")

    logger.info(f"Pushing to Hub: {hub_model_id}")
    model.push_to_hub(hub_model_id, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model available at: https://huggingface.co/{hub_model_id}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()