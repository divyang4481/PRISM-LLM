# Developer Notes - PRISM-LLM Dataset Strategy

This document outlines the dataset preparation strategy and the planned phases for model training.

## 📊 Dataset Phases

### Phase 1: Baseline & Hygiene (Current)
**Goal**: Establish a stable pretokenization pipeline and verify basic transformer functionality.
- **Datasets**: 
    - `wikitext-103-raw-v1`: High-quality Wikipedia text for general language modeling.
    - `roneneldan/TinyStories`: Simple, syntactically clean stories for small-model experiments.
- **Format**: Pretokenized `.npy` files (uint16/uint32) with a corresponding `metadata.json`.
- **Storage**: `data_cache/<dataset_name>/`

### Phase 2: Knowledge Distillation & Memory Integration
**Goal**: Integrate the PRISM multi-bank memory system and align it using KD.
- **Datasets**: Expanded WikiText and multi-domain corpora.
- **Complexity**: Introduction of long-context samples (4k+).

### Phase 3: Scaling & Long-Context Benchmarking
**Goal**: Stress-test the prime-aware sparse memory.
- **Datasets**: PG-19 (Project Gutenberg) or specific long-form reasoning datasets.

---

## 🛠️ How to Prepare Datasets

Use the provided preparation script to download, tokenize, and cache datasets locally.

### Command Examples
```bash
# Set PYTHONPATH
$env:PYTHONPATH="src"

# Prepare WikiText-103
python scripts/prepare_dataset.py --dataset wikitext --tokenizer gpt2

# Prepare TinyStories
python scripts/prepare_dataset.py --dataset tinystories --tokenizer gpt2
```

### Cached Layout
```text
data_cache/
  wikitext103/
    train.npy
    val.npy
    metadata.json
  tinystories/
    train.npy
    val.npy
    metadata.json
```

### metadata.json schema
```json
{
  "dataset_name": "wikitext",
  "hf_identifier": "wikitext-103-raw-v1",
  "tokenizer_name": "gpt2",
  "vocab_size": 50257,
  "dtype": "uint16",
  "train_tokens": 100000000,
  "val_tokens": 5000000
}
```
