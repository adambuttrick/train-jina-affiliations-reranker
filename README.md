# Jina Affiliation Reranker Training

Fine-tune [jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) for affiliation string matching.

## Dataset

Mixed training combining two sources:

[cometadata/triplet-loss-for-embedding-affiliations-sample-1](https://huggingface.co/datasets/cometadata/triplet-loss-for-embedding-affiliations-sample-1)
- ~8K curated triplets (anchor, positive, negative), pre-sorted by difficulty
- 80% hard negatives, 20% easy negatives

[cometadata/affilgood-scored](https://huggingface.co/datasets/cometadata/affilgood-scored)
- AffilGood contrastive pairs, pre-scored with the fine-tuned model
- Filtered to hardest cases (margin < 0.05) where the model struggles
- Added at 10% ratio relative to primary dataset

## Training

### Prerequisites

1. HuggingFace account with write access to your target org
2. HuggingFace CLI logged in: `huggingface-cli login`
3. HuggingFace Pro or Enterprise (for Jobs)

### Command-Line Arguments

```
--org ORG               HuggingFace organization to push the model to (required)
--model-name NAME       Model name (default: jina-reranker-v2-multilingual-affiliations-v4)
--dataset DATASET       Primary dataset (default: cometadata/triplet-loss-for-embedding-affiliations-sample-1)
--affilgood-scored DS   Pre-scored AffilGood dataset (default: cometadata/affilgood-scored)
--margin-threshold N    Keep AffilGood triplets with margin below this (default: 0.05)
--affilgood-ratio N     AffilGood samples as fraction of primary dataset (default: 0.1)
--base-model MODEL      Base model to fine-tune (default: jinaai/jina-reranker-v2-base-multilingual)
--output-dir DIR        Local checkpoint directory (default: ./output)
--epochs N              Training epochs (default: 2)
--batch-size N          Batch size (default: 32)
--learning-rate LR      Learning rate (default: 2e-5)
--eval-steps N          Steps between evaluations (default: 500)
--val-split FRAC        Validation fraction (default: 0.05)
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
