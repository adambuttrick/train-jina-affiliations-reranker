# Jina Affiliation Reranker Training

Fine-tune [jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) for affiliation string matching.

## Dataset

Uses [cometadata/triplet-loss-for-embedding-affiliations-sample-1](https://huggingface.co/datasets/cometadata/triplet-loss-for-embedding-affiliations-sample-1):
- ~8K triplets (anchor, positive, negative), pre-sorted by difficulty
- 80% hard negatives, 20% easy negatives

## Training

### Prerequisites

1. HuggingFace account with write access to your target org
2. HuggingFace CLI logged in: `huggingface-cli login`
3. HuggingFace Pro or Enterprise (for Jobs)

### Command-Line Arguments

```
--org ORG             HuggingFace organization to push the model to (required)
--model-name NAME     Model name (default: jina-reranker-v2-multilingual-affiliations)
--dataset DATASET     Training dataset (default: cometadata/triplet-loss-for-embedding-affiliations-sample-1)
--base-model MODEL    Base model to fine-tune (default: jinaai/jina-reranker-v2-base-multilingual)
--output-dir DIR      Local checkpoint directory (default: ./output)
--epochs N            Training epochs (default: 3)
--batch-size N        Batch size (default: 16)
--learning-rate LR    Learning rate (default: 2e-5)
--eval-steps N        Steps between evaluations (default: 500)
--val-split FRAC      Validation fraction (default: 0.15)
```

### Run Locally

```bash
cd train
uv run train_reranker.py --org cometadata
```

### Run on HuggingFace Jobs

```bash
cd train

# Get your HF token
export HF_TOKEN=$(huggingface-cli whoami --token)

# Run training on L4 GPU
hf jobs uv run \
  --flavor l4x1 \
  --timeout 2h \
  --secret HF_TOKEN=$HF_TOKEN \
  train_reranker.py --org cometadata
```

### Monitor Job

```bash
# List your jobs
hf jobs list

# View logs (replace JOB_ID)
hf jobs logs JOB_ID

# Cancel if needed
hf jobs cancel JOB_ID
```

## Evaluation

The evaluation script benchmarks base vs fine-tuned models across 300 test cases organized in 10 difficulty tiers.

### Run Evaluation

```bash
cd eval
uv run eval_reranker.py
```

### Results

**Overall Performance:**

| Metric | Base Model | Fine-tuned | Δ |
|--------|------------|------------|---|
| Accuracy | 78.3% | 84.3% | +6.0% |
| MRR | 0.873 | 0.913 | +0.040 |

**Performance by Tier:**

| Tier | Cases | Base | Fine-tuned | Δ |
|------|-------|------|------------|---|
| Baseline | 30 | 100.0% | 100.0% | — |
| OCR/Noise | 30 | 100.0% | 100.0% | — |
| Abbreviations | 40 | 60.0% | 80.0% | +20.0% |
| Hierarchical | 35 | 71.4% | 77.1% | +5.7% |
| Medical/Hospital | 25 | 64.0% | 68.0% | +4.0% |
| Research Labs | 25 | 80.0% | 84.0% | +4.0% |
| International | 35 | 82.9% | 91.4% | +8.6% |
| Disambiguation | 31 | 45.2% | 51.6% | +6.5% |
| Negative Controls | 19 | 100.0% | 100.0% | — |
| Ultra-Hard | 30 | 93.3% | 96.7% | +3.3% |

Results are saved to `eval_results.json`.

## Output

Trained model: [cometadata/jina-reranker-v2-multilingual-affiliations](https://huggingface.co/cometadata/jina-reranker-v2-multilingual-affiliations)

### Usage

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cometadata/jina-reranker-v2-multilingual-affiliations",
    trust_remote_code=True,
)

# Score affiliation pairs
pairs = [
    ["University of California, Berkeley", "UC Berkeley"],
    ["University of California, Berkeley", "Berkeley College"],
]
scores = model.predict(pairs)
# Higher score = more likely to be the same institution
```
