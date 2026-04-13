import torch
import time
import argparse
import yaml
from prism_llm.model.decoder import DecoderForCausalLM
from prism_llm.model.config import ModelConfig
from prism_llm.utils.config_utils import load_config_from_yaml

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_efficiency(m_cfg, device="cuda", num_steps=50, batch_size=4):
    print(f"\n--- Benchmarking Architecture Efficiency ---")
    print(f"Model: d_model={m_cfg.d_model}, layers={m_cfg.n_layers}, heads={m_cfg.n_heads}, kv_heads={m_cfg.n_kv_heads}")
    
    # Initialize model
    model = DecoderForCausalLM(m_cfg).to(device)
    params = count_parameters(model)
    print(f"Total Parameters: {params:,}")

    # Generate dummy input
    input_ids = torch.randint(0, m_cfg.vocab_size, (batch_size, m_cfg.max_seq_len)).to(device)
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        _ = model(input_ids)
    torch.cuda.synchronize()

    # Benchmark Forward + Backward
    print(f"Profiling {num_steps} steps (Forward + Backward)...")
    optimizer = torch.optim.AdamW(model.parameters())
    
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        for _ in range(num_steps):
            optimizer.zero_grad()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"!!! CRASHED: Architecture hit OOM (Out of Memory) at seq_len {m_cfg.max_seq_len}")
            return {"tokens_per_sec": 0, "peak_mem": float('inf'), "params": params, "oom": True}
        raise e
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Metrics
    total_time = end_time - start_time
    ms_per_step = (total_time / num_steps) * 1000
    tokens_per_sec = (batch_size * m_cfg.max_seq_len * num_steps) / total_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB

    # FLOPs Estimation (Rule of thumb for Transformers: 6 * params per token for train)
    flops_per_token = 6 * params
    total_flops = flops_per_token * batch_size * m_cfg.max_seq_len
    tflops_per_sec = (total_flops / (total_time / num_steps)) / 1e12

    print(f"\nRESULTS:")
    print(f"- Speed: {tokens_per_sec:,.0f} tokens/sec")
    print(f"- Latency: {ms_per_step:.2f} ms/step")
    print(f"- Peak VRAM: {peak_mem:.2f} MB")
    print(f"- Est. Compute: {tflops_per_sec:.4f} TFLOPS")
    
    return {
        "tokens_per_sec": tokens_per_sec,
        "peak_mem": peak_mem,
        "params": params,
        "oom": False
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Load config
    m_cfg = load_config_from_yaml(ModelConfig, args.model_config)

    # 1. Benchmark our current GQA config
    res_gqa = benchmark_efficiency(m_cfg, device=args.device, batch_size=args.batch_size)

    # 2. Compare against standard MHA (if we changed kv_heads to match n_heads)
    print("\n" + "="*40)
    print("COMPARING AGAINST STANDARD MHA (Estimated)")
    from dataclasses import replace
    m_cfg_mha = replace(m_cfg, n_kv_heads=m_cfg.n_heads)
    
    res_mha = benchmark_efficiency(m_cfg_mha, device=args.device, batch_size=args.batch_size)

    # Calculate Improvement
    print("\n" + "!"*40)
    print(f"ARCHITECTURE GAINS:")
    
    if res_gqa.get("oom") or res_mha.get("oom"):
        if res_gqa.get("oom") and res_mha.get("oom"):
            print("Both architectures hit OOM. Context window too large for this GPU.")
        elif res_mha.get("oom"):
            print("WIN: GQA survived where MHA hit OOM! (Infinite Improvement)")
        else:
            print("ERROR: GQA hit OOM but MHA survived. Check implementation.")
    else:
        mem_saving = 100 * (1 - res_gqa["peak_mem"] / res_mha["peak_mem"])
        speed_gain = 100 * (res_gqa["tokens_per_sec"] / res_mha["tokens_per_sec"] - 1)
        print(f"Memory Saved by GQA: {mem_saving:.1f}%")
        print(f"Speed Gain by GQA: {speed_gain:.1f}%")
    
    print("!"*40)

if __name__ == "__main__":
    main()
