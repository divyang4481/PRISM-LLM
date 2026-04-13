# 🚀 PRISM-LLM (Parallel Real-time Integrated Storage & Memory)

**Project Status**: 🟢 **Phase 1 (Baseline Infrastructure) - Complete** | 🟡 **Phase 2 (Memory Architecture) - In Progress**

> [!IMPORTANT]
> **Status Note**: The repository currently implements a high-performance **GQA (Grouped Query Attention) Decoder Baseline**. While the baseline is verified, GPU-accelerated, and producing narrative text, the core "PRISM Memory System" (Recent, Anchor, Summary, Latent banks) is currently in the MVP implementation stage.

## 🚀 Accomplishments (Phase 1)
- **Architecture**: Verified GQA (4:1 ratio) with RoPE and RMSNorm.
- **Pipeline**: Memory-mapped `.npy` pretokenization for instant billion-token loading.
- **Hardware**: Fully optimized for laptop GPUs (RTX 4050 tested) with 15k tok/s throughput.
- **Stability**: Checkpoint resumption and auto-pruning (latest 2 kept).
- **Inference**: Working `scripts/generate.py` for narrative validation.

## 🛠 Experimental Roadmap (Phase 2)
We are currently moving from a standard Dense Decoder to the structured PRISM memory architecture.

### Phase 2A: Memory MVP
1. **RecentMemoryBank**: Fixed sliding-window local context.
2. **AnchorMemoryBank**: Deterministic "prime" token selection and cross-attention.
3. **Memory Router**: Learned gating mechanism to mix local and anchor context.

### Phase 2B: Full PRISM
1. **Summary Bank**: Semantic compression.
2. **Latent Bank**: Reconstructive memory.
3. **Distillation (KD)**: Teaching the memory student using a dense teacher.

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
