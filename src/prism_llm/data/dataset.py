import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Union

class SyntheticDataset(Dataset):
    """
    A simple synthetic dataset that generates random token sequences.
    Useful for smoke testing the pipeline.
    """
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Generate random tokens. Note: seq_len + 1 because the collator
        # might need it if we didn't do shifting in the model, but since
        # the model handles shifting, we just generate exactly seq_len tokens.
        return torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)

class PretokenizedDataset(Dataset):
    """
    A dataset that loads pre-tokenized integer sequences from a numpy array
    or list of lists and provides fixed-length chunks.
    """
    def __init__(self, data: Union[np.ndarray, List[List[int]], torch.Tensor], seq_len: int):
        if isinstance(data, list):
            # Flatten if it's a list of lists, or assume it's a flat list
            if data and isinstance(data[0], list):
                flat_data = [item for sublist in data for item in sublist]
                self.data = torch.tensor(flat_data, dtype=torch.long)
            else:
                self.data = torch.tensor(data, dtype=torch.long)
        elif isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).long()
        elif isinstance(data, torch.Tensor):
            self.data = data.long()
        else:
            raise ValueError("Unsupported data type")

        # Flatten the data to a 1D tensor
        if self.data.dim() > 1:
            self.data = self.data.view(-1)

        self.seq_len = seq_len
        self.num_samples = len(self.data) // self.seq_len

        # Truncate to exact multiple of seq_len
        self.data = self.data[:self.num_samples * self.seq_len]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        return self.data[start_idx:end_idx]
