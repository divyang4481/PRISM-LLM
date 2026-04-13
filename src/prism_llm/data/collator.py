import torch
from typing import List, Dict

class CausalLMCollator:
    """
    Collator for causal language modeling.
    Since the model handles label shifting internally, this just
    returns inputs as both input_ids and labels.
    """
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # If examples are already of the same length, stack them directly
        # For variable length, we would need to pad

        # Check if padding is needed
        lengths = [len(ex) for ex in examples]
        max_len = max(lengths)

        if all(length == max_len for length in lengths):
            # All same length, just stack
            input_ids = torch.stack(examples)
            labels = input_ids.clone()
        else:
            # Need to pad
            input_ids = torch.full((len(examples), max_len), self.pad_token_id, dtype=torch.long)
            labels = torch.full((len(examples), max_len), -100, dtype=torch.long)

            for i, ex in enumerate(examples):
                input_ids[i, :len(ex)] = ex
                labels[i, :len(ex)] = ex

        return {
            "input_ids": input_ids,
            "labels": labels
        }
