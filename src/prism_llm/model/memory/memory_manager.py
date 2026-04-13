import torch
import torch.nn as nn
from prism_llm.model.memory.recent_bank import RecentMemoryBank
from prism_llm.model.memory.anchor_bank import AnchorMemoryBank

class MemoryManager(nn.Module):
    """
    Orchestrates Recent and Anchor banks in the decoder path.
    Implements the PRISM-MVP Gated Mixing logic.
    """
    def __init__(self, config):
        super().__init__()
        self.recent_bank = RecentMemoryBank(config.memory_window)
        self.anchor_bank = AnchorMemoryBank(config.anchor_interval)
        
        # Learned gate to mix local and anchor context
        self.gate = nn.Linear(config.d_model, 1)
        
    def forward(self, hidden_states, local_attn_out, q, k, v, attention_module):
        """
        hidden_states: [batch, seq_len, d_model]
        local_attn_out: [batch, seq_len, d_model]
        q, k, v: [batch, heads/kv_heads, seq_len, head_dim]
        """
        # 1. Get Anchors
        k_anchors, v_anchors = self.anchor_bank.select_anchors(k, v)
        
        # 1.5 Expand Anchors to match Q's heads (GQA logic)
        k_anchors, v_anchors = attention_module.expand_kv(k_anchors, v_anchors)
        
        # 2. Memory Attention (Attention of current Q on distant Anchors)
        anchor_attn_out = attention_module.scaled_dot_product_attention(
            q, k_anchors, v_anchors, mask=None 
        )
        
        # 3. Gated Mixing
        # hidden_states: [batch, seq_len, d_model]
        # gate: [batch, seq_len, 1] -> [batch, 1, seq_len, 1] for broadcasting
        g = torch.sigmoid(self.gate(hidden_states)).transpose(1, 2).unsqueeze(-1)
        
        # Both local_attn_out and anchor_attn_out are [batch, n_heads, seq_len, head_dim]
        mixed_out = g * local_attn_out + (1 - g) * anchor_attn_out
        
        return mixed_out
