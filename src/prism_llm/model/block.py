import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import ModelConfig
from .norms import RMSNorm
from .attention_gqa import GroupedQueryAttention
from .mlp import FeedForward

class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Pre-Norm
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attention = GroupedQueryAttention(config)

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
        attn_output, attn_weights = self.attention(
            hidden_states=normed_hidden_states,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            return_attn_weights=return_attn_weights,
        )
        hidden_states = hidden_states + attn_output

        # MLP block (Pre-Norm + Residual)
        normed_hidden_states = self.norm2(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = hidden_states + mlp_output

        return hidden_states, attn_weights
