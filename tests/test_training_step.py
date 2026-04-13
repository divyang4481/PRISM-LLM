import torch
from torch.utils.data import DataLoader

from prism_llm.model.config import ModelConfig
from prism_llm.model.decoder import DecoderForCausalLM
from prism_llm.data.dataset import SyntheticDataset
from prism_llm.data.collator import CausalLMCollator
from prism_llm.train.config import TrainConfig
from prism_llm.train.trainer import Trainer

def test_training_step_updates_weights():
    """
    Tests that a single training step successfully updates the model's weights.
    """
    # 1. Setup small model
    config = ModelConfig(
        vocab_size=100,
        max_seq_len=32,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        rope_base=10000.0,
        norm_eps=1e-5,
        bias=False,
        tie_word_embeddings=True
    )
    model = DecoderForCausalLM(config)

    # Store initial weights to compare later
    initial_weights = {name: param.clone().detach() for name, param in model.named_parameters()}

    # 2. Setup tiny dataset
    dataset = SyntheticDataset(vocab_size=100, seq_len=16, num_samples=8)
    collator = CausalLMCollator()
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)

    # 3. Setup trainer
    train_config = TrainConfig(
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        max_steps=2,
        logging_steps=1,
        save_steps=100,
        eval_steps=100
    )

    trainer = Trainer(
        model=model,
        train_config=train_config,
        train_dataloader=dataloader,
        device="cpu"
    )

    # 4. Run one step
    trainer.train()

    # 5. Check weights updated
    weights_changed = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not torch.allclose(param, initial_weights[name]):
                weights_changed = True
                break

    assert weights_changed, "Model weights were not updated after one training step"
