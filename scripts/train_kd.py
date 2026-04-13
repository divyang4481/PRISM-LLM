import argparse
import logging
import torch
from torch.utils.data import DataLoader

from prism_llm.model.config import ModelConfig
from prism_llm.model.decoder import DecoderForCausalLM
from prism_llm.train.config import TrainConfig
from prism_llm.train.kd_trainer import KDTrainer
from prism_llm.distill.teacher import load_teacher_model
from prism_llm.data.dataset import SyntheticDataset
from prism_llm.data.collator import CausalLMCollator
from prism_llm.utils import load_config_from_yaml

def main():
    parser = argparse.ArgumentParser(description="KD Training for PRISM-LLM")
    parser.add_argument("--student_config", type=str, required=True, help="Path to student model config YAML")
    parser.add_argument("--teacher_config", type=str, required=True, help="Path to teacher model config YAML")
    parser.add_argument("--teacher_checkpoint", type=str, help="Path to teacher model checkpoint (.pt)")
    parser.add_argument("--train_config", type=str, required=True, help="Path to train config YAML")
    parser.add_argument("--alpha", type=float, default=0.5, help="KD loss weight")
    parser.add_argument("--temperature", type=float, default=2.0, help="KD temperature")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load configs
    logger.info(f"Loading student config from {args.student_config}")
    s_m_cfg = load_config_from_yaml(ModelConfig, args.student_config)
    
    logger.info(f"Loading train config from {args.train_config}")
    t_cfg = load_config_from_yaml(TrainConfig, args.train_config)
    
    if args.output_dir:
        t_cfg.output_dir = args.output_dir

    # Initialize student model
    logger.info("Initializing student model...")
    student_model = DecoderForCausalLM(s_m_cfg)

    # Initialize teacher model
    logger.info(f"Loading teacher model from {args.teacher_config}...")
    teacher_model = load_teacher_model(
        config_path=args.teacher_config,
        checkpoint_path=args.teacher_checkpoint,
        device=device
    )

    # Initialize dataset
    logger.info("Initializing dataset (synthetic)...")
    train_dataset = SyntheticDataset(
        vocab_size=s_m_cfg.vocab_size,
        seq_len=s_m_cfg.max_seq_len,
        num_samples=1000
    )
    eval_dataset = SyntheticDataset(
        vocab_size=s_m_cfg.vocab_size,
        seq_len=s_m_cfg.max_seq_len,
        num_samples=100
    )

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

    # Initialize KD Trainer
    trainer = KDTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        train_config=t_cfg,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        alpha=args.alpha,
        temperature=args.temperature,
        device=device
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
