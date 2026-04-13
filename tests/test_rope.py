import torch
import pytest
from prism_llm.model.rope import RotaryEmbedding, apply_rotary_pos_emb

def test_rope_initialization():
    # Should work with even head_dim
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
    assert rope.head_dim == 16
    assert rope.max_seq_len == 64

    # Should fail with odd head_dim
    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim=15, max_seq_len=64)

def test_rope_preserves_shape_and_modifies_values():
    head_dim = 16
    max_seq_len = 32
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=max_seq_len)

    seq_len = 10
    batch_size = 2
    n_heads = 4
    n_kv_heads = 2

    # Create random q and k
    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_kv_heads, seq_len, head_dim)

    # Keep original to check for modifications
    q_orig = q.clone()
    k_orig = k.clone()

    cos, sin = rope(seq_len)

    assert cos.shape == (1, 1, seq_len, head_dim)
    assert sin.shape == (1, 1, seq_len, head_dim)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    # Shapes should be preserved
    assert q_rot.shape == q_orig.shape
    assert k_rot.shape == k_orig.shape

    # Values should be modified
    assert not torch.allclose(q_rot, q_orig)
    assert not torch.allclose(k_rot, k_orig)

def test_rope_consistent_positions():
    head_dim = 16
    max_seq_len = 32
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=max_seq_len)

    cos1, sin1 = rope(10)
    cos2, sin2 = rope(20)

    # The first 10 positions of a longer sequence should match a shorter sequence
    assert torch.allclose(cos1[0, 0, :10, :], cos2[0, 0, :10, :])
    assert torch.allclose(sin1[0, 0, :10, :], sin2[0, 0, :10, :])
