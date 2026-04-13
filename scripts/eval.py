import argparse
import logging
import torch
from torch.utils.data import DataLoader

from prism_llm.model.config import ModelConfig
from prism_llm.model.decoder import DecoderForCausalLM
from prism_llm.train.config import DataConfig
from prism_llm.train.checkpoint import load_checkpoint
from prism_llm.data.dataset import SyntheticDataset
from prism_llm.data.collator import CausalLMCollator
from prism_llm.eval.perplexity import evaluate_perplexity
from prism_llm.utils import load_config_from_yaml

def main():
    parser = argparse.ArgumentParser(description="Evaluate PRISM-LLM")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint (.pt)")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--data_config", type=str, help="Path to data config YAML (optional)")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configs
    logger.info(f"Loading model config from {args.model_config}")
    m_cfg = load_config_from_yaml(ModelConfig, args.model_config)
    
    if args.data_config:
        logger.info(f"Loading data config from {args.data_config}")
        d_cfg = load_config_from_yaml(DataConfig, args.data_config)
    else:
        logger.info("Using default data config (synthetic)")
        d_cfg = DataConfig(vocab_size=m_cfg.vocab_size, seq_len=m_cfg.max_seq_len)

    # Initialize model
    logger.info("Initializing model...")
    model = DecoderForCausalLM(m_cfg)
    
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}...")
        load_checkpoint(args.checkpoint, model, map_location=str(device))
    
    model.to(device)

    # Initialize dataset
    logger.info(f"Initializing evaluation dataset (type: {d_cfg.dataset_type})...")
    if d_cfg.dataset_type == "synthetic":
        eval_dataset = SyntheticDataset(
            vocab_size=d_cfg.vocab_size,
            seq_len=d_cfg.seq_len,
            num_samples=200 # Fixed size for quick evaluation
        )
    else:
        raise NotImplementedError("Pretokenized dataset loading not fully implemented in CLI yet.")

    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CausalLMCollator()
    )

    # Evaluate
    logger.info("Starting evaluation...")
    results = evaluate_perplexity(model, dataloader, device)
    
    logger.info("-" * 30)
    logger.info("Evaluation Results:")
    logger.info(f"Loss: {results['loss']:.4f}")
    logger.info(f"Perplexity: {results['perplexity']:.4f}")
    logger.info("-" * 30)

if __name__ == "__main__":
    main()
