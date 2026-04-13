import os
import json
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Prepare datasets for PRISM-LLM")
    parser.add_argument("--dataset", type=str, choices=["wikitext", "tinystories"], required=True,
                        help="Which dataset to prepare")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="Hugging Face tokenizer name (e.g., gpt2, meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--cache_dir", type=str, default="data_cache",
                        help="Directory to store tokenized data")
    return parser.parse_args()

def main():
    args = get_args()
    
    # Dataset configurations
    configs = {
        "wikitext": {
            "path": "wikitext",
            "name": "wikitext-103-raw-v1",
            "out_folder": "wikitext103",
        },
        "tinystories": {
            "path": "roneneldan/TinyStories",
            "name": None,
            "out_folder": "tinystories",
        }
    }
    
    cfg = configs[args.dataset]
    output_path = os.path.join(args.cache_dir, cfg["out_folder"])
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Loading dataset: {args.dataset}...")
    ds = load_dataset(cfg["path"], cfg["name"])
    
    print(f"Loading tokenizer: {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_func(examples):
        # Append EOS token to each example to separate them
        texts = [t + tokenizer.eos_token for t in examples["text"]]
        return tokenizer(texts, truncation=False, return_attention_mask=False)

    metadata = {
        "dataset_name": args.dataset,
        "hf_identifier": cfg["name"] if cfg["name"] else cfg["path"],
        "tokenizer_name": args.tokenizer,
        "vocab_size": len(tokenizer),
    }

    dtype = np.uint32 if len(tokenizer) > 65535 else np.uint16

    for split in ["train", "validation"]:
        print(f"Processing {split} split...")
        split_ds = ds[split]
        
        # Tokenize
        tokenized = split_ds.map(
            tokenize_func,
            batched=True,
            remove_columns=split_ds.column_names,
            desc=f"Tokenizing {split}",
            num_proc=os.cpu_count() // 2 if os.cpu_count() > 1 else 1
        )
        
        # Concatenate all tokens efficiently by pre-allocating
        print(f"Concatenating {split} tokens...")
        total_len = sum(len(ids) for ids in tokenized["input_ids"])
        tokens_np = np.empty(total_len, dtype=dtype)
        
        curr = 0
        for ids in tqdm(tokenized["input_ids"], desc=f"Flattening {split}"):
            length = len(ids)
            tokens_np[curr:curr+length] = ids
            curr += length
        
        # Save as .npy
        out_name = "train.npy" if split == "train" else "val.npy"
        out_file = os.path.join(output_path, out_name)
        np.save(out_file, tokens_np)
        
        metadata[f"{split}_tokens"] = len(tokens_np)
        metadata["dtype"] = str(tokens_np.dtype)

    # Save metadata
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nSuccessfully prepared {args.dataset}!")
    print(f"Output saved to: {output_path}")
    print(f"Total tokens: {metadata['train_tokens'] + (metadata.get('validation_tokens', 0))}")

if __name__ == "__main__":
    main()
