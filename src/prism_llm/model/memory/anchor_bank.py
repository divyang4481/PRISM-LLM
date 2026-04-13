import torch
import torch.nn as nn

class AnchorMemoryBank(nn.Module):
    """
    Selectively preserves 'anchor' tokens from the distant past.
    MVP Heuristic: Keep BOS token + every N-th token.
    """
    def __init__(self, anchor_interval: int = 16):
        super().__init__()
        self.anchor_interval = anchor_interval

    def select_anchors(self, k: torch.Tensor, v: torch.Tensor):
        """
        k, v: [batch, n_kv_heads, seq_len, head_dim]
        Returns k_anchors, v_anchors
        """
        seq_len = k.size(-2)
        if seq_len < self.anchor_interval:
            return k, v

        # Heuristic 2: Keep every N-th token
        # Using a range and ensuring no duplicates
        indices_list = [0]
        for i in range(self.anchor_interval, seq_len, self.anchor_interval):
            indices_list.append(i)
        
        indices = torch.tensor(indices_list, device=k.device, dtype=torch.long)
        
        k_anchors = k.index_select(-2, indices)
        v_anchors = v.index_select(-2, indices)
        
        return k_anchors, v_anchors
