import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import ModelConfig
from .rope import apply_rotary_pos_emb

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_kv_groups = config.n_kv_groups
        self.max_seq_len = config.max_seq_len

        # Projections
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape

        if seq_len > self.max_seq_len:
             raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        # Projections
        # [batch, seq_len, n_heads * head_dim]
        q = self.q_proj(hidden_states)
        # [batch, seq_len, n_kv_heads * head_dim]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        # [batch, seq_len, n_heads, head_dim] -> [batch, n_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # [batch, seq_len, n_kv_heads, head_dim] -> [batch, n_kv_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Grouped-Query Attention: Expand K and V
        # [batch, n_kv_heads, seq_len, head_dim] -> [batch, n_kv_heads, 1, seq_len, head_dim]
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)

        # Expand across groups
        # [batch, n_kv_heads, n_kv_groups, seq_len, head_dim]
        k = k.expand(-1, -1, self.n_kv_groups, -1, -1)
        v = v.expand(-1, -1, self.n_kv_groups, -1, -1)

        # Reshape to match Q's heads
        # [batch, n_heads, seq_len, head_dim]
        k = k.reshape(batch_size, self.n_heads, seq_len, self.head_dim)
        v = v.reshape(batch_size, self.n_heads, seq_len, self.head_dim)

        # Manual Scaled Dot-Product Attention
        # Q @ K^T / sqrt(d)
        # [batch, n_heads, seq_len, head_dim] @ [batch, n_heads, head_dim, seq_len] -> [batch, n_heads, seq_len, seq_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask creation
        # Create a boolean mask where True means "keep" and False means "-inf"
        causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden_states.device).tril()

        # Apply causal mask (convert False positions to -inf)
        attn_weights = attn_weights.masked_fill(~causal_mask, float('-inf'))

        # Optional additive attention mask (e.g., from padding) broadcastable to [batch, 1, seq_len, seq_len]
        if attention_mask is not None:
             # attention_mask is expected to contain 0 for keep and -inf for mask, or similar additive structure
             attn_weights = attn_weights + attention_mask

        # Softmax and Dropout
        attn_weights_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights_probs = self.attn_dropout(attn_weights_probs)

        # Matmul with V
        # [batch, n_heads, seq_len, seq_len] @ [batch, n_heads, seq_len, head_dim] -> [batch, n_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights_probs, v)

        # Reshape and project out
        # [batch, n_heads, seq_len, head_dim] -> [batch, seq_len, n_heads, head_dim] -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights_probs if return_attn_weights else None
