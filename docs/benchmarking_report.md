# PRISM-LLM Benchmarking Report

This document records the efficiency gains and architectural performance of the PRISM-LLM training pipeline.

## 1. Baseline Performance (Short Context)
**Configuration**: `d_model: 256`, `n_layers: 4`, `max_seq_len: 128`

| Metric | GQA (4:1 Ratio) | MHA (Standard) | Delta | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Speed** | 12,504 tok/s | 15,056 tok/s | -17% | Overhead of GQA logic dominates at tiny scales. |
| **Peak VRAM** | 644.9 MB | 649.4 MB | -0.7% | KV Cache is too small to show significant savings. |
| **TFLOPS** | 1.17 | 1.45 | - | - |

## 2. High-Context Scalability (Target: 4k/8k)
*Testing 2048/4096 context...*

## 3. Training Infrastructure
- **Device**: NVIDIA GeForce RTX 4050 Laptop GPU
- **Data Loading**: Memory-mapped `.npy` files (Fast loading, zero RAM hit).
- **Resumption**: Fully supported with latest 2 checkpoints preserved.
