import argparse
import logging
import torch
from torch.utils.data import DataLoader

from prism_llm.model.config import ModelConfig
from prism_llm.model.decoder import DecoderForCausalLM
from prism_llm.train.config import TrainConfig, DataConfig
from prism_llm.train.trainer import Trainer
from prism_llm.data.dataset import SyntheticDataset, PretokenizedDataset
from prism_llm.data.collator import CausalLMCollator
from prism_llm.utils import load_config_from_yaml

def main():
    parser = argparse.ArgumentParser(description="Train PRISM-LLM")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--train_config", type=str, required=True, help="Path to train config YAML")
    parser.add_argument("--data_config", type=str, help="Path to data config YAML (optional)")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Load configs
    logger.info(f"Loading model config from {args.model_config}")
    m_cfg = load_config_from_yaml(ModelConfig, args.model_config)
    
    logger.info(f"Loading train config from {args.train_config}")
    t_cfg = load_config_from_yaml(TrainConfig, args.train_config)
    
    if args.data_config:
        logger.info(f"Loading data config from {args.data_config}")
        d_cfg = load_config_from_yaml(DataConfig, args.data_config)
    else:
        logger.info("Using default data config (synthetic)")
        d_cfg = DataConfig(vocab_size=m_cfg.vocab_size, seq_len=m_cfg.max_seq_len)

    if args.output_dir:
        t_cfg.output_dir = args.output_dir

    # Initialize model
    logger.info("Initializing model...")
    model = DecoderForCausalLM(m_cfg)

    # Initialize dataset
    logger.info(f"Initializing dataset (type: {d_cfg.dataset_type})...")
    if d_cfg.dataset_type == "synthetic":
        train_dataset = SyntheticDataset(
            vocab_size=d_cfg.vocab_size,
            seq_len=d_cfg.seq_len,
            num_samples=d_cfg.num_samples
        )
        eval_dataset = SyntheticDataset(
            vocab_size=d_cfg.vocab_size,
            seq_len=d_cfg.seq_len,
            num_samples=d_cfg.num_samples // 10
        )
    else:
        # Placeholder for pretokenized logic if needed
        raise NotImplementedError("Pretokenized dataset loading not fully implemented in CLI yet.")

    collator = CausalLMCollator()
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=t_cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=t_cfg.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_config=t_cfg,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
