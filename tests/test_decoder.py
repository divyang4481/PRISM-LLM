import torch
from prism_llm.model.config import ModelConfig
from prism_llm.model.decoder import DecoderForCausalLM

def get_tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        max_seq_len=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        rope_base=10000.0,
        norm_eps=1e-5,
        bias=False,
        tie_word_embeddings=True,
    )

def test_decoder_forward():
    config = get_tiny_config()
    model = DecoderForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))

    outputs = model(input_ids, return_hidden_states=True, return_attn_weights=True)

    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 10, config.vocab_size)

    assert "hidden_states" in outputs
    # Embeddings + n_layers blocks
    assert len(outputs["hidden_states"]) == config.n_layers + 1

    assert "attn_weights" in outputs
    assert len(outputs["attn_weights"]) == config.n_layers

def test_decoder_loss():
    config = get_tiny_config()
    model = DecoderForCausalLM(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    labels = input_ids.clone()

    outputs = model(input_ids, labels=labels)

    assert "loss" in outputs
    assert outputs["loss"].dim() == 0  # scalar
    assert outputs["loss"].item() > 0

def test_tied_embeddings():
    config_tied = get_tiny_config()
    config_untied = get_tiny_config()
    config_untied.tie_word_embeddings = False

    model_tied = DecoderForCausalLM(config_tied)
    model_untied = DecoderForCausalLM(config_untied)

    # In tied model, the weights should be the exact same object
    assert model_tied.lm_head.weight is model_tied.model.embeddings.embedding.weight

    # In untied model, they should be different objects
    assert model_untied.lm_head.weight is not model_untied.model.embeddings.embedding.weight
