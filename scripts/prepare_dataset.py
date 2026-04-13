#!/usr/bin/env python3
"""
Dataset Preparation Script for PRISM-LLM

Downloads, tokenizes, and caches datasets locally.

Examples:
    python scripts/prepare_dataset.py \
      --dataset wikitext103 \
      --output_dir data_cache/wikitext103 \
      --tokenizer gpt2 \
      --split train

    python scripts/prepare_dataset.py \
      --dataset tinystories \
      --output_dir data_cache/tinystories \
      --tokenizer gpt2
"""

import argparse
import logging
from prism_llm.data.prepare import prepare_dataset

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for PRISM-LLM training.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of the dataset (e.g., wikitext103, tinystories)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Local directory to cache the prepared dataset")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="Hugging Face tokenizer to use (e.g., gpt2)")
    parser.add_argument("--split", type=str, default=None,
                        help="Specific split to prepare (e.g., train, validation). If not provided, prepares all splits.")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    prepare_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        split=args.split
    )

if __name__ == "__main__":
    main()
