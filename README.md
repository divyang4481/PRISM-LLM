# 🚀 PRISM-LLM

### **Prime-Residual Integrated Sparse Memory Transformer**

> A bounded-memory large language model architecture designed to achieve **long-context reasoning (4k+) under 2–4 GB VRAM** using **prime-aware sparse memory, semantic compression, and reconstructive latent states**.

---

# 🧠 Motivation

Standard Transformers scale poorly with context:
- KV cache grows **linearly with tokens**
- Memory becomes the bottleneck before compute
- Long-context models require **high-end GPUs**

PRISM-LLM introduces a new paradigm:
> ❌ Store everything
> ✅ Store only what matters — and reconstruct the rest

---

# 🧩 Architecture Overview

## 🔷 Memory System
```
Full Context (4k tokens)
        ↓
[ Recent Window ]  → exact KV
[ Anchor Bank   ]  → sparse tokens (prime-aware)
[ Summary Bank  ]  → chunk representations
[ Latent Bank   ]  → compressed residual state
```

---

# 🏗️ Project Structure

```text
PRISM-LLM/
├── configs/            # Config YAMLs (model, train, data)
├── scripts/            # Entry point scripts (train, eval, kd)
├── src/
│   ├── prism_llm/
│   │   ├── model/      # Transformer and Memory architecture
│   │   ├── train/      # Trainer and KDTrainer logic
│   │   ├── distill/    # KD losses and teacher loading
│   │   ├── data/       # Datasets and Collators
│   │   ├── eval/       # Perplexity and Metrics logic
│   │   ├── utils/      # Configuration and helpers
```

---

# 🚀 Setup (Conda + Poetry)

1. **Create and activate a conda environment**:
   ```bash
   conda create -n prism-llm python=3.10
   conda activate prism-llm
   ```

2. **Install PyTorch via conda**:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

3. **Install dependencies**:
   ```bash
   pip install poetry
   poetry install
   ```

> [!IMPORTANT]
> **Vocab Size Alignment**: Ensure the `vocab_size` in your model config (e.g., `student_tiny.yaml`) matches the tokenizer used in `prepare_dataset.py`. For `gpt2`, use `50257`.

---

# 🏃 Running

First, set the `PYTHONPATH` to include the `src` directory:
```powershell
$env:PYTHONPATH="src"
```

### 🛠️ Standard Training
```bash
python scripts/train.py --model_config configs/model/student_tiny.yaml --train_config configs/train/stage1_smoke.yaml
```

### 🕯️ Knowledge Distillation (KD)
To train a student model using a teacher model:
```bash
python scripts/train_kd.py `
    --student_config configs/model/student_tiny.yaml `
    --teacher_config configs/model/teacher_dense.yaml `
    --train_config configs/train/stage1_smoke.yaml `
    --alpha 0.5 `
    --temperature 2.0
```

### 🧪 Evaluation
To evaluate a trained model checkpoint:
```bash
# Evaluate the smoke test checkpoint
python scripts/eval.py --model_config configs/model/student_tiny.yaml --checkpoint outputs/smoke_test/checkpoint-50.pt
```

### 💨 Smoke Test (Forward Pass only)
```bash
python scripts/smoke_forward.py
```

---

# 🎯 Goals
- ✅ Run on **2–4 GB VRAM**
- ✅ Support **4k–8k context**
- ✅ Maintain **long-range reasoning**
- ✅ Reduce KV memory drastically
- ✅ Be trainable via **Knowledge Distillation**
