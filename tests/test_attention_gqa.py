import torch
from prism_llm.model.config import ModelConfig
from prism_llm.model.attention_gqa import GroupedQueryAttention
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

def test_causal_masking():
    config = get_tiny_config()
    attention = GroupedQueryAttention(config)

    seq_len = 10
    hidden_states = torch.randn(2, seq_len, config.d_model)

    rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_base)
    cos, sin = rope(seq_len)

    _, attn_weights = attention(hidden_states, cos, sin, return_attn_weights=True)

    # attn_weights shape is [batch, n_heads, seq_len, seq_len]
    assert attn_weights is not None

    # Check that upper triangular part is exactly zero (no attention to future tokens)
    for b in range(2):
        for h in range(config.n_heads):
            # For each position i, it should only attend to positions j <= i
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    # Future attention should be 0.0
                    assert attn_weights[b, h, i, j].item() == 0.0

def test_attention_with_additive_mask():
    config = get_tiny_config()
    attention = GroupedQueryAttention(config)

    seq_len = 4
    hidden_states = torch.randn(1, seq_len, config.d_model)

    rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_base)
    cos, sin = rope(seq_len)

    # Create an additive mask that masks out the last token
    # 0 for keep, -inf for mask
    attention_mask = torch.zeros((1, 1, seq_len, seq_len))
    attention_mask[0, 0, :, 3] = float('-inf')

    _, attn_weights = attention(hidden_states, cos, sin, attention_mask=attention_mask, return_attn_weights=True)

    assert attn_weights is not None

    # Check that attention to the last token is 0
    # For token 0, 1, 2, they cannot attend to token 3 because of causal mask anyway
    # For token 3, it cannot attend to itself because of our custom attention_mask
    assert attn_weights[0, 0, 3, 3].item() == 0.0
