import torch
from prism_llm.model.config import ModelConfig
from prism_llm.model.embeddings import TokenEmbedding
from prism_llm.model.attention_gqa import GroupedQueryAttention
from prism_llm.model.block import DecoderBlock
from prism_llm.model.decoder import DecoderForCausalLM
from prism_llm.model.rope import RotaryEmbedding

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

def test_embedding_shape():
    config = get_tiny_config()
    embed = TokenEmbedding(config.vocab_size, config.d_model)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = embed(input_ids)
    assert output.shape == (2, 10, 32)

def test_attention_shape():
    config = get_tiny_config()
    attention = GroupedQueryAttention(config)
    hidden_states = torch.randn(2, 10, config.d_model)

    rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_base)
    cos, sin = rope(10)

    output, attn_weights = attention(hidden_states, cos, sin, return_attn_weights=True)

    assert output.shape == (2, 10, config.d_model)
    assert attn_weights.shape == (2, config.n_heads, 10, 10)

def test_block_shape():
    config = get_tiny_config()
    block = DecoderBlock(config)
    hidden_states = torch.randn(2, 10, config.d_model)

    rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_base)
    cos, sin = rope(10)

    output, attn_weights = block(hidden_states, cos, sin, return_attn_weights=True)

    assert output.shape == (2, 10, config.d_model)
    assert attn_weights.shape == (2, config.n_heads, 10, 10)

def test_decoder_logits_shape():
    config = get_tiny_config()
    model = DecoderForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))

    output = model(input_ids)

    assert "logits" in output
    assert output["logits"].shape == (2, 10, config.vocab_size)
