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

Inspired by:

- irregular prime-based sampling (anti-aliasing)
- semantic compression
- residual latent reconstruction
- multi-bank memory systems

---

# ⚡ Key Idea

Replace full token history:

[
{K_1, ..., K_T}
]

with a **bounded structured memory**:

[
\mathcal{M} =
\mathcal{R} \cup
\mathcal{A} \cup
\mathcal{S} \cup
\mathcal{Z}
]

| Memory Bank       | Purpose                           |
| ----------------- | --------------------------------- |
| **Recent (R)**    | Exact last tokens (local fluency) |
| **Anchors (A)**   | Sparse prime-aware key tokens     |
| **Summaries (S)** | Compressed semantic chunks        |
| **Latent (Z)**    | Reconstructable old context       |

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

## 🔷 Attention

[
y_t =
g_r A_{rec} +
g_a A_{anc} +
g_s A_{sum} +
g_z A_{lat}
]

Where:

- each memory bank contributes separately
- gating decides importance dynamically

---

## 🔷 Prime-Aware Anchoring

We bias memory selection using prime periodicity:

[
\Pi(i)=
[\cos(2\pi i/p_1), \sin(2\pi i/p_1), ...]
]

This helps:

- avoid periodic compression artifacts
- preserve long-range structure
- improve retrieval stability

---

## 🔷 Compression

Instead of storing all tokens:

[
T = 4096
]

We store:

[
R + A + S + Z \approx 700–800
]

👉 ~80% memory reduction

---

# 🎯 Goals

- ✅ Run on **2–4 GB VRAM**
- ✅ Support **4k–8k context**
- ✅ Maintain **long-range reasoning**
- ✅ Reduce KV memory drastically
- ✅ Be trainable via **Knowledge Distillation**

---

# 🏗️ Project Structure

```text
PRISM-LLM/
├── configs/
├── src/
│   ├── model/
│   │   ├── decoder.py
│   │   ├── attention_gqa.py
│   │   ├── memory/
│   │   │   ├── recent_bank.py
│   │   │   ├── anchor_bank.py
│   │   │   ├── summary_bank.py
│   │   │   ├── latent_bank.py
│   │   │   ├── memory_manager.py
│   ├── distill/
│   │   ├── kd_losses.py
│   │   ├── teacher.py
│   ├── train/
│   │   ├── trainer.py
│   ├── eval/
│   │   ├── long_context_tests.py
│   │   ├── perplexity.py
├── scripts/
│   ├── train_kd.py
│   ├── eval.py
```

---

# ⚙️ Model Configuration

## 🧠 Student (target)

| Parameter   | Value         |
| ----------- | ------------- |
| Layers      | 12            |
| Hidden Size | 640           |
| Heads (Q)   | 10            |
| Heads (KV)  | 2 (GQA)       |
| Context     | 4096          |
| Precision   | 4-bit weights |

## 🧠 Memory Limits

| Bank      | Size |
| --------- | ---- |
| Recent    | 512  |
| Anchors   | 128  |
| Summaries | 64   |
| Latent    | 64   |

---

# 🔥 Training Strategy (KD-based)

## Stage 1 — Baseline KD

- Train dense student with:
  - CE loss
  - logit distillation

## Stage 2 — Memory Compression

- Enable:
  - anchor bank
  - summary bank

- Add:
  - hidden-state KD
  - attention KD

## Stage 3 — Budget Enforcement

- Gradually reduce:
  - anchors
  - summaries

## Stage 4 — Latent Reconstruction

- Add latent memory
- Train reconstruction loss

---

# 📉 Loss Function

[
\mathcal{L} =
\mathcal{L}\_{CE}

- \mathcal{L}\_{KD}
- \mathcal{L}\_{hidden}
- \mathcal{L}\_{attn}
- \mathcal{L}\_{memory}
  ]

---

# 🧪 Evaluation

## Core Metrics

- Perplexity
- Long-range retrieval accuracy
- Memory usage (VRAM)
- Compression ratio

## Special Tests

- Needle-in-haystack retrieval
- Repeated pattern stability
- Summarization accuracy

---

# 📊 Expected Outcome

| Feature     | Standard LLM  | PRISM-LLM  |
| ----------- | ------------- | ---------- |
| Context     | Limited by KV | 4k+        |
| VRAM        | High          | 2–4 GB     |
| Memory      | Linear        | Bounded    |
| Retrieval   | Degrades      | Stable     |
| Compression | None          | Structured |

---

# 🛠️ Roadmap

## ✅ Phase 1 (Week 1–2)

- [ ] Baseline decoder (GQA + RoPE)
- [ ] Tokenizer + dataset pipeline
- [ ] KD training loop

## ✅ Phase 2 (Week 3–4)

- [ ] Anchor bank (prime-aware)
- [ ] Summary bank
- [ ] Memory manager
- [ ] KD (hidden + attention)

## ✅ Phase 3 (Week 5–6)

- [ ] Latent bank
- [ ] Reconstruction module
- [ ] Memory budget control

## ✅ Phase 4 (Week 7–8)

- [ ] Long-context benchmarks
- [ ] VRAM profiling
- [ ] Ablation studies

---

# 🧠 Research Contributions

PRISM-LLM introduces:

1. **Prime-aware sparse memory selection**
2. **Multi-bank bounded memory system**
3. **Semantic-energy compression**
4. **Latent KV reconstruction**
5. **KD for compressed memory alignment**

---

# 🔮 Future Work

- Dynamic memory scaling per query
- Learned prime distributions
- hardware-aware KV compression
- integration with retrieval systems
- multimodal memory extension

---

# 🤝 Contribution

Open for:

- architecture experiments
- training improvements
- efficient kernels
- new compression strategies

---

# ⭐ Final Vision

> PRISM-LLM is not just a smaller LLM.
> It is a **different way to think about memory in transformers**.

---

# ⚡ Next Step (for you)

Tell me and I’ll generate it immediately:

👉 **“Create initial code skeleton”**
👉 **“Give training script + PyTorch code”**
👉 **“Design teacher model + dataset plan”**

We can go from idea → working model very fast now 🚀
