import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import ModelConfig
from .norms import RMSNorm
from .attention_gqa import GroupedQueryAttention
from .mlp import FeedForward
from .memory.memory_manager import MemoryManager

class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Pre-Norm
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.memory = MemoryManager(config)

        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Attention block (Pre-Norm + Residual)
        normed_hidden_states = self.norm1(hidden_states)
        local_attn_output, q, k, v = self.attention(
            hidden_states=normed_hidden_states,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            return_attn_weights=return_attn_weights,
        )
        
        # PRISM Memory Path (Mixing local with anchors)
        # mixed_attn_output is [batch, n_heads, seq_len, head_dim]
        mixed_attn_output = self.memory(normed_hidden_states, local_attn_output, q, k, v, self.attention)
        
        # Final head-merging and projection
        batch_size, n_heads, seq_len, head_dim = mixed_attn_output.shape
        attn_output = mixed_attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model)
        attn_output = self.attention.o_proj(attn_output)
        attn_output = self.attention.resid_dropout(attn_output)
        
        hidden_states = hidden_states + attn_output

        # MLP block (Pre-Norm + Residual)
        normed_hidden_states = self.norm2(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = hidden_states + mlp_output

        return hidden_states, None
