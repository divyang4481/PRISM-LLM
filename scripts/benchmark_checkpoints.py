import torch
import argparse
import os
import glob
import time
import json
import re
from transformers import GPT2Tokenizer
from prism_llm.model.decoder import DecoderForCausalLM
from prism_llm.model.config import ModelConfig
from prism_llm.utils.config_utils import load_config_from_yaml

DEFAULT_PROMPTS = [
    "Once upon a time in a small village,",
    "The scientist opened the notebook and discovered",
    "In the future, language models will",
    "The captain looked at the map and said",
    "Question: What is the capital of France? Answer:",
]


def generate_and_measure(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    greedy=False,
    device="cuda",
):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]

    start_time = time.time()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs["logits"][:, -1, :]

            if greedy:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                if top_k > 0:
                    indices_to_remove = (
                        logits < torch.topk(logits, top_k)[0][..., -1, None]
                    )
                    logits[indices_to_remove] = float("-inf")

                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    end_time = time.time()
    total_time = end_time - start_time

    generated_length = input_ids.shape[1] - prompt_length
    tokens_per_sec = generated_length / total_time if total_time > 0 else 0

    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return {
        "prompt": prompt,
        "output": output_text,
        "prompt_length": prompt_length,
        "generated_length": generated_length,
        "total_time_sec": total_time,
        "tokens_per_sec": tokens_per_sec,
    }


def get_checkpoints(checkpoint_dir, checkpoints_list):
    checkpoints = []

    if checkpoints_list:
        checkpoints.extend(checkpoints_list)

    if checkpoint_dir and os.path.isdir(checkpoint_dir):
        # Auto-discover checkpoint-*.pt
        pattern = os.path.join(checkpoint_dir, "checkpoint-*.pt")
        found_files = glob.glob(pattern)

        # Sort by step number
        def extract_step(filepath):
            match = re.search(r"checkpoint-(\d+)\.pt", os.path.basename(filepath))
            return int(match.group(1)) if match else -1

        found_files.sort(key=extract_step)
        checkpoints.extend(found_files)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for cp in checkpoints:
        if cp not in seen and os.path.exists(cp):
            seen.add(cp)
            result.append(cp)

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark Checkpoints for PRISM-LLM")
    parser.add_argument(
        "--model_config", type=str, required=True, help="Path to model config"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, help="Directory to auto-discover checkpoints"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        help="Explicit list of checkpoints to evaluate",
    )
    parser.add_argument("--prompt", type=str, help="Single custom prompt to use")
    parser.add_argument(
        "--prompts_file",
        type=str,
        help="Path to a file containing prompts (one per line)",
    )

    # Decoding args
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Where to save results",
    )
    args = parser.parse_args()

    # Determine prompts
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, "r") as f:
            prompts.extend([line.strip() for line in f if line.strip()])

    if not prompts:
        prompts = DEFAULT_PROMPTS

    # Get checkpoints
    checkpoints = get_checkpoints(args.checkpoint_dir, args.checkpoints)
    if not checkpoints:
        print("No checkpoints found. Please provide --checkpoint_dir or --checkpoints.")
        return

    print(f"Found {len(checkpoints)} checkpoints to evaluate.")

    # Setup model/tokenizer
    m_cfg = load_config_from_yaml(ModelConfig, args.model_config)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    for cp_path in checkpoints:
        print(f"\n--- Evaluating {cp_path} ---")
        model = DecoderForCausalLM(m_cfg).to(args.device)

        checkpoint = torch.load(cp_path, map_location=args.device, weights_only=False)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        cp_results = []
        for p in prompts:
            print(f"Prompt: {p[:50]}...")
            res = generate_and_measure(
                model,
                tokenizer,
                p,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                greedy=args.greedy,
                device=args.device,
            )
            print(
                f"  Generated {res['generated_length']} tokens at {res['tokens_per_sec']:.2f} tok/s"
            )
            cp_results.append(res)

        cp_name = os.path.basename(cp_path).replace(".pt", "")
        all_results[cp_name] = cp_results

        # Save individual checkpoint JSON
        cp_json_path = os.path.join(args.output_dir, f"{cp_name}_results.json")
        with open(cp_json_path, "w") as f:
            json.dump(cp_results, f, indent=2)

        # Save individual checkpoint MD
        cp_md_path = os.path.join(args.output_dir, f"{cp_name}_results.md")
        with open(cp_md_path, "w") as f:
            f.write(f"# Benchmark Results: {cp_name}\n\n")
            f.write(f"- **Max New Tokens:** {args.max_new_tokens}\n")
            f.write(f"- **Temperature:** {args.temperature}\n")
            f.write(f"- **Top-K:** {args.top_k}\n")
            f.write(f"- **Greedy:** {args.greedy}\n\n")

            for res in cp_results:
                f.write(f"## Prompt\n> {res['prompt']}\n\n")
                f.write(f"### Output\n{res['output']}\n\n")
                f.write("**Metrics:**\n")
                f.write(f"- Prompt length: {res['prompt_length']} tokens\n")
                f.write(f"- Generated length: {res['generated_length']} tokens\n")
                f.write(f"- Tokens per sec: {res['tokens_per_sec']:.2f}\n")
                f.write(f"- Total time: {res['total_time_sec']:.2f} sec\n\n")
                f.write("---\n\n")

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll benchmarking complete. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
