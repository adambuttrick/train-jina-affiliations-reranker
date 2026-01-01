# Reranker Training Notes: AffilGood Integration

## Summary

Successfully integrated SIRIS-Lab/affilgood-contrastive-dataset into reranker training to improve disambiguation of confusable institutions. The key insight was using a **small, targeted supplement** of only the hardest cases rather than the full dataset.

## Final Configuration (v4)

```
Model: cometadata/jina-reranker-v2-multilingual-affiliations-v4
Base: jinaai/jina-reranker-v2-base-multilingual
```

### Training Parameters
| Parameter | Value |
|-----------|-------|
| AffilGood ratio | 10% of original dataset |
| Margin threshold | < 0.05 (hardest cases) |
| AffilGood samples | 766 triplets |
| Original samples | 7,665 triplets |
| Total training | 8,431 triplets |
| Epochs | 2 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Loss | BinaryCrossEntropyLoss (standard) |

### Results
| Metric | Base Model | v4 | Change |
|--------|------------|-----|--------|
| Overall Accuracy | 84.5% | 83.4% | -1.1% |
| Overall MRR | 0.9117 | 0.8974 | -0.0143 |
| **Disambiguation Accuracy** | 51.6% | 61.3% | **+9.7%** |
| Disambiguation MRR | 0.7285 | 0.7796 | +0.0511 |

---

## What Worked

### 1. Tiny, Targeted Supplement (10% ratio)
Adding AffilGood at only 10% of the original dataset size prevented the model from overfitting to AffilGood's specific patterns while still learning disambiguation signals.

### 2. Hardest Cases Only (margin < 0.05)
Pre-scoring AffilGood with the fine-tuned model and filtering to margin < 0.05 selected only the ~26,500 triplets where the model was already struggling (out of 135,440 total). This focused training on genuine weaknesses rather than reinforcing what the model already knew.

### 3. Single-Phase Mixed Training
Shuffling original + AffilGood together and training in one phase worked better than:
- Two-phase training (original first, then AffilGood)
- AffilGood-only fine-tuning
- Larger AffilGood ratios

### 4. Standard Loss Function
Using the standard BinaryCrossEntropyLoss without modifications. Weighted loss experiments (giving AffilGood samples 0.5 weight) actually performed worse.

---

## What Didn't Work

### 1. Full AffilGood Dataset
Using all 135,440 AffilGood triplets caused severe regression across all tiers. The model became too specialized for AffilGood's specific disambiguation patterns.

### 2. High AffilGood Ratios (25%, 50%, 100%)
Higher ratios degraded performance on baseline tiers (OCR, abbreviations, hierarchical) while providing diminishing returns on disambiguation.

### 3. Higher Margin Thresholds
- margin < 0.1: Too many samples, diluted the signal
- margin < 0.2: Similar issue
- margin < 0.3: Way too many easy cases included

### 4. Weighted Loss
Attempted to reduce AffilGood influence by weighting those samples at 0.5x:
- Result: +6.5% disambiguation (worse than +9.7% unweighted)
- Result: 13 regressions (worse than 11 unweighted)
- Conclusion: The 10% ratio already provides sufficient balance

### 5. Two-Phase Training
Training on original dataset first, then fine-tuning on AffilGood led to catastrophic forgetting of the original training.

---

## Specific Improvements (7 cases)

### Disambiguation Tier (3 wins)
| Test Case | Description | Before | After |
|-----------|-------------|--------|-------|
| George Washington | GWU vs other Washingtons | Ranked #2 | **Correct** |
| Ohio State proper | OSU vs other Ohio schools | Failed | **Correct** |
| Trinity Dublin vs Cambridge | Irish vs UK Trinity | Failed | **Correct** |

### Abbreviations Tier (2 wins)
| Test Case | Description | Before | After |
|-----------|-------------|--------|-------|
| HKU abbreviation | Hong Kong University | Ranked #2 | **Correct** |
| ENS abbreviation | École Normale Supérieure | Failed | **Correct** |

### Hierarchical Tier (1 win)
| Test Case | Description | Before | After |
|-----------|-------------|--------|-------|
| Research group | Subunit within institution | Failed | **Correct** |

### International Tier (1 win)
| Test Case | Description | Before | After |
|-----------|-------------|--------|-------|
| Tokyo fullwidth | Fullwidth character handling | Failed | **Correct** |

---

## Regressions (11 cases)

### Abbreviations (2)
- Cal informal (UC Berkeley as "Cal")
- Stanford AI Lab

### Hierarchical (3)
- Chicago Booth
- Kellogg School
- Perimeter Institute

### Medical/Hospital (2)
- Penn Medicine
- MSKCC affiliation

### Research Labs (1)
- Argonne National Lab

### International (2)
- Tsinghua Wade-Giles vs Pinyin
- Qinghua Pinyin variant

### Negative Controls (1)
- Berkeley not UCLA (false positive)

---

## Tier-by-Tier Analysis

| Tier | Base | v4 | Change | Notes |
|------|------|-----|--------|-------|
| Baseline | 100% | 100% | 0% | No change (already perfect) |
| OCR/Noise | 100% | 100% | 0% | No change (already perfect) |
| Abbreviations | 80% | 80% | 0% | Net neutral (2 wins, 2 losses) |
| Hierarchical | 77.1% | 71.4% | -5.7% | Some regression |
| Medical/Hospital | 68% | 60% | -8% | Largest regression tier |
| Research Labs | 84% | 80% | -4% | Minor regression |
| International | 87.4% | 86.4% | -1% | Minor regression |
| **Disambiguation** | 51.6% | 61.3% | **+9.7%** | **Primary goal achieved** |
| Negative Controls | 100% | 94.7% | -5.3% | One false positive introduced |
| Ultra-Hard | 96.7% | 96.7% | 0% | No change |

---

## Score Distribution Changes

| Metric | Base | v4 | Interpretation |
|--------|------|-----|----------------|
| Mean Positive Score | 0.456 | 0.471 | Slightly higher confidence on matches |
| Mean Negative Score | 0.065 | 0.135 | Higher scores on negatives (less decisive) |
| Mean Score Gap | 0.391 | 0.336 | Narrower margin between pos/neg |

The model became slightly less "confident" overall, with a narrower gap between positive and negative scores. This is a trade-off for better disambiguation capability.

---

## Data Pipeline

### Pre-scoring (score_affilgood.py)
1. Load SIRIS-Lab/affilgood-contrastive-dataset (52,900 samples)
2. Expand to triplets (one per hard_negative) → 135,440 triplets
3. Score each triplet with fine-tuned model
4. Compute margin = positive_score - negative_score
5. Push to cometadata/affilgood-scored

### Training (train_reranker.py)
1. Load original dataset (cometadata/triplet-loss-for-embedding-affiliations-sample-1)
2. Load pre-scored AffilGood, filter by margin threshold
3. Downsample AffilGood to target ratio
4. Shuffle combined dataset
5. Train with standard CrossEncoderTrainer

---

## Recommendations for Future Work

### To further improve disambiguation:
1. Create more targeted disambiguation training data
2. Add hard negatives specifically for the failing cases (Washington variants, Miami variants)
3. Consider curriculum learning within disambiguation cases

### To reduce regressions:
1. Add regularization to preserve performance on hierarchical/medical tiers
2. Create validation set that covers all tiers equally
3. Consider multi-task learning with tier-specific heads

### To improve medical/hospital:
1. This tier has the worst baseline (68%) and regressed further
2. May need dedicated medical affiliation training data
3. Hospital naming conventions are particularly challenging

---

## Files

| File | Purpose |
|------|---------|
| `train/train_reranker.py` | Main training script |
| `train/score_affilgood.py` | Pre-scores AffilGood for filtering |
| `train/filter_affilgood.py` | Alternative local filtering approach |
| `eval/eval_reranker.py` | Comprehensive 368-case evaluation |

## Datasets

| Dataset | Description |
|---------|-------------|
| `cometadata/triplet-loss-for-embedding-affiliations-sample-1` | Original curated training data (8,069 triplets) |
| `cometadata/affilgood-scored` | Pre-scored AffilGood (135,440 triplets with margins) |
| `SIRIS-Lab/affilgood-contrastive-dataset` | Source disambiguation dataset |

## Models

| Model | Description |
|-------|-------------|
| `jinaai/jina-reranker-v2-base-multilingual` | Base model |
| `cometadata/jina-reranker-v2-multilingual-affiliations` | Original fine-tuned (v1) |
| `cometadata/jina-reranker-v2-multilingual-affiliations-v4` | Best model with AffilGood integration |
