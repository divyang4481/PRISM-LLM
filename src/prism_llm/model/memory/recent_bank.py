import torch
import torch.nn as nn

class RecentMemoryBank(nn.Module):
    """
    Standard sliding-window memory bank for local context.
    Ensures the model always has access to the most recent W tokens' KV states.
    """
    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size
        
        # Note: In a stateless training pass, the "sliding window" is often 
        # enforced via an attention mask. However, for the PRISM multi-bank 
        # architecture, we define it as its own entity to allow the 
        # MemoryRouter to mix it with Anchor and Summary banks.
        
    def forward(self, k: torch.Tensor, v: torch.Tensor):
        """
        k, v: [batch, n_kv_heads, seq_len, head_dim]
        Returns the k, v tensors truncated/mask-ready for the local window.
        """
        seq_len = k.size(-2)
        if seq_len <= self.window_size:
            return k, v
        
        # For training on a full sequence, we return the full KV, 
        # but the attention logic will apply the window mask.
        # For inference, the bank will maintain a cache (implemented in Inference subclass).
        return k, v

    def get_window_mask(self, seq_len: int, device: torch.device):
        """
        Generates a 2D sliding window causal mask.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        # Add sliding window constraint: mask out anything older than window_size
        window_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-self.window_size).bool()
        return mask | window_mask
