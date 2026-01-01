# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers>=4.0.0",
#     "datasets",
#     "torch",
#     "einops",
#     "huggingface_hub",
# ]
# ///
"""
Score AffilGood triplets with the fine-tuned model and push to HuggingFace.

This creates a scored dataset that can be filtered later with different thresholds.

Usage:
    uv run score_affilgood.py --org cometadata
"""

import argparse
import logging
import re

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from sentence_transformers import CrossEncoder

# Regex to match special tags in AffilGood dataset
SPECIAL_TAGS = re.compile(r'\[(MENTION|CITY|COUNTRY|ACRONYM|PARENT|REGION)\]')

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def remove_special_tags(text: str) -> str:
    """Remove [MENTION], [CITY], etc. and normalize whitespace."""
    cleaned = SPECIAL_TAGS.sub('', text)
    return ' '.join(cleaned.split())


def load_affilgood_triplets() -> list[dict]:
    """Load AffilGood dataset and convert to triplet format."""
    logger.info("Loading SIRIS-Lab/affilgood-contrastive-dataset...")
    dataset = load_dataset("SIRIS-Lab/affilgood-contrastive-dataset", split="train")
    logger.info(f"Loaded {len(dataset)} samples from AffilGood dataset")

    triplets = []
    for idx, row in enumerate(dataset):
        anchor = remove_special_tags(row["query"])
        positive = remove_special_tags(row["positive"])

        for neg_idx, negative in enumerate(row["hard_negatives"]):
            triplets.append({
                "source_idx": idx,
                "negative_idx": neg_idx,
                "anchor": anchor,
                "positive": positive,
                "negative": remove_special_tags(negative),
            })

    logger.info(f"Expanded to {len(triplets)} triplets (one per hard_negative)")
    return triplets


def score_triplets(model: CrossEncoder, triplets: list[dict], batch_size: int = 64) -> list[dict]:
    """
    Score each triplet with the model and compute margins.

    Returns triplets with added score information.
    """
    logger.info(f"Scoring {len(triplets)} triplets...")

    # Prepare pairs for scoring
    positive_pairs = [(t["anchor"], t["positive"]) for t in triplets]
    negative_pairs = [(t["anchor"], t["negative"]) for t in triplets]

    # Score in batches
    logger.info("Scoring positive pairs...")
    positive_scores = model.predict(positive_pairs, batch_size=batch_size, show_progress_bar=True)

    logger.info("Scoring negative pairs...")
    negative_scores = model.predict(negative_pairs, batch_size=batch_size, show_progress_bar=True)

    # Add scores to triplets
    scored_triplets = []
    for i, triplet in enumerate(triplets):
        pos_score = float(positive_scores[i])
        neg_score = float(negative_scores[i])
        scored_triplets.append({
            **triplet,
            "positive_score": pos_score,
            "negative_score": neg_score,
            "margin": pos_score - neg_score,
            "model_correct": pos_score > neg_score,
        })

    return scored_triplets


def analyze_distribution(scored_triplets: list[dict]) -> dict:
    """Analyze and log the score distribution."""
    margins = [t["margin"] for t in scored_triplets]
    positive_scores = [t["positive_score"] for t in scored_triplets]
    negative_scores = [t["negative_score"] for t in scored_triplets]
    correct = sum(1 for t in scored_triplets if t["model_correct"])

    # Count by margin buckets
    buckets = {
        "negative": len([m for m in margins if m < 0]),
        "0.0-0.1": len([m for m in margins if 0 <= m < 0.1]),
        "0.1-0.2": len([m for m in margins if 0.1 <= m < 0.2]),
        "0.2-0.3": len([m for m in margins if 0.2 <= m < 0.3]),
        "0.3-0.4": len([m for m in margins if 0.3 <= m < 0.4]),
        "0.4-0.5": len([m for m in margins if 0.4 <= m < 0.5]),
        "0.5+": len([m for m in margins if m >= 0.5]),
    }

    stats = {
        "total": len(scored_triplets),
        "correct": correct,
        "accuracy": correct / len(scored_triplets),
        "mean_positive_score": sum(positive_scores) / len(positive_scores),
        "mean_negative_score": sum(negative_scores) / len(negative_scores),
        "mean_margin": sum(margins) / len(margins),
        "min_margin": min(margins),
        "max_margin": max(margins),
        "buckets": buckets,
    }

    logger.info("=" * 60)
    logger.info("SCORE DISTRIBUTION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Total triplets: {stats['total']}")
    logger.info(f"Model accuracy: {stats['accuracy']:.2%} ({correct}/{len(scored_triplets)})")
    logger.info(f"Mean positive score: {stats['mean_positive_score']:.4f}")
    logger.info(f"Mean negative score: {stats['mean_negative_score']:.4f}")
    logger.info(f"Mean margin: {stats['mean_margin']:.4f}")
    logger.info(f"Min margin: {stats['min_margin']:.4f}")
    logger.info(f"Max margin: {stats['max_margin']:.4f}")
    logger.info("")
    logger.info("Margin distribution:")
    for bucket, count in buckets.items():
        pct = count / len(scored_triplets) * 100
        bar = "#" * int(pct / 2)
        logger.info(f"  {bucket:15s}: {count:6d} ({pct:5.1f}%) {bar}")
    logger.info("=" * 60)

    # Log threshold recommendations
    logger.info("")
    logger.info("FILTERING RECOMMENDATIONS:")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        count = len([m for m in margins if m < threshold])
        logger.info(f"  margin < {threshold}: {count:6d} triplets ({count/len(margins)*100:.1f}%)")
    logger.info("=" * 60)

    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score AffilGood triplets and push to HuggingFace"
    )
    parser.add_argument(
        "--org",
        type=str,
        required=True,
        help="HuggingFace organization to push dataset to",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cometadata/jina-reranker-v2-multilingual-affiliations",
        help="Model to use for scoring (default: fine-tuned model)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="affilgood-scored",
        help="Name for the output dataset (default: affilgood-scored)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for scoring (default: 64)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_id = f"{args.org}/{args.dataset_name}"

    logger.info("=" * 60)
    logger.info("AffilGood Scoring Pipeline")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Output dataset: {dataset_id}")
    logger.info("=" * 60)

    # Load model
    logger.info(f"Loading model: {args.model}")
    model = CrossEncoder(args.model, trust_remote_code=True)

    # Load and expand AffilGood
    triplets = load_affilgood_triplets()

    # Score all triplets
    scored_triplets = score_triplets(model, triplets, batch_size=args.batch_size)

    # Analyze distribution
    stats = analyze_distribution(scored_triplets)

    # Convert to HuggingFace Dataset
    logger.info("Converting to HuggingFace Dataset...")
    dataset = Dataset.from_list(scored_triplets)

    # Push to Hub
    logger.info(f"Pushing to HuggingFace Hub: {dataset_id}")
    dataset.push_to_hub(
        dataset_id,
        private=False,
        commit_message=f"Scored with {args.model}",
    )

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"Dataset available at: https://huggingface.co/datasets/{dataset_id}")
    logger.info("")
    logger.info("To filter, load with:")
    logger.info(f"  ds = load_dataset('{dataset_id}', split='train')")
    logger.info("  filtered = ds.filter(lambda x: x['margin'] < 0.3)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
