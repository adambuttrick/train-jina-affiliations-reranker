# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers>=4.0.0",
#     "datasets",
#     "torch",
#     "einops",
# ]
# ///
"""
Filter AffilGood dataset to keep only disambiguation cases where additional training would help,
while avoiding cases the model already handles well.

Usage:
    uv run filter_affilgood.py --threshold 0.3 --output filtered_affilgood.json
"""

import argparse
import json
import logging
import re

from datasets import load_dataset
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
    for row in dataset:
        anchor = remove_special_tags(row["query"])
        positive = remove_special_tags(row["positive"])

        for negative in row["hard_negatives"]:
            triplets.append({
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
        scored_triplets.append({
            **triplet,
            "positive_score": float(positive_scores[i]),
            "negative_score": float(negative_scores[i]),
            "margin": float(positive_scores[i] - negative_scores[i]),
        })

    return scored_triplets


def analyze_distribution(scored_triplets: list[dict]) -> None:
    """Print statistics about the score distribution."""
    margins = [t["margin"] for t in scored_triplets]
    positive_scores = [t["positive_score"] for t in scored_triplets]
    negative_scores = [t["negative_score"] for t in scored_triplets]

    # Count by margin buckets
    buckets = {
        "negative (model wrong)": len([m for m in margins if m < 0]),
        "0.0 - 0.1": len([m for m in margins if 0 <= m < 0.1]),
        "0.1 - 0.2": len([m for m in margins if 0.1 <= m < 0.2]),
        "0.2 - 0.3": len([m for m in margins if 0.2 <= m < 0.3]),
        "0.3 - 0.4": len([m for m in margins if 0.3 <= m < 0.4]),
        "0.4 - 0.5": len([m for m in margins if 0.4 <= m < 0.5]),
        "0.5+": len([m for m in margins if m >= 0.5]),
    }

    logger.info("=" * 60)
    logger.info("SCORE DISTRIBUTION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Total triplets: {len(scored_triplets)}")
    logger.info(f"Mean positive score: {sum(positive_scores) / len(positive_scores):.4f}")
    logger.info(f"Mean negative score: {sum(negative_scores) / len(negative_scores):.4f}")
    logger.info(f"Mean margin: {sum(margins) / len(margins):.4f}")
    logger.info(f"Min margin: {min(margins):.4f}")
    logger.info(f"Max margin: {max(margins):.4f}")
    logger.info("")
    logger.info("Margin distribution:")
    for bucket, count in buckets.items():
        pct = count / len(scored_triplets) * 100
        bar = "#" * int(pct / 2)
        logger.info(f"  {bucket:25s}: {count:6d} ({pct:5.1f}%) {bar}")
    logger.info("=" * 60)


def filter_by_threshold(scored_triplets: list[dict], threshold: float) -> list[dict]:
    """Keep only triplets where model margin is below threshold."""
    filtered = [t for t in scored_triplets if t["margin"] < threshold]
    logger.info(f"Filtered to {len(filtered)} triplets with margin < {threshold}")
    return filtered


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter AffilGood to cases where fine-tuned model struggles"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cometadata/jina-reranker-v2-multilingual-affiliations",
        help="Model to use for scoring (default: fine-tuned model)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Keep triplets with margin below this threshold (default: 0.3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="filtered_affilgood.json",
        help="Output file for filtered triplets (default: filtered_affilgood.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for scoring (default: 64)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze distribution, don't filter",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("AffilGood Score-Based Filtering")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 60)

    logger.info(f"Loading model: {args.model}")
    model = CrossEncoder(args.model, trust_remote_code=True)
    triplets = load_affilgood_triplets()
    scored_triplets = score_triplets(model, triplets, batch_size=args.batch_size)

    analyze_distribution(scored_triplets)

    if args.analyze_only:
        logger.info("Analyze-only mode, not saving filtered output")
        return

    filtered = filter_by_threshold(scored_triplets, args.threshold)

    output_triplets = [
        {"anchor": t["anchor"], "positive": t["positive"], "negative": t["negative"]}
        for t in filtered
    ]

    with open(args.output, "w") as f:
        json.dump(output_triplets, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(output_triplets)} filtered triplets to {args.output}")

    scored_output = args.output.replace(".json", "_scored.json")
    with open(scored_output, "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved scored version to {scored_output}")


if __name__ == "__main__":
    main()
