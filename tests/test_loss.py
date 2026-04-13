import torch

from prism_llm.model.config import ModelConfig
from prism_llm.model.decoder import DecoderForCausalLM

def test_loss_ignores_index_and_shifts():
    """
    Tests that the causal LM loss function correctly shifts tokens
    and ignores the padding index (-100).
    """
    config = ModelConfig(
        vocab_size=100,
        max_seq_len=32,
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        rope_base=10000.0,
        norm_eps=1e-5,
        bias=False,
        tie_word_embeddings=True
    )
    model = DecoderForCausalLM(config)
    model.eval()

    # Create input with some padding (represented by -100 in labels)
    # Batch size 1, seq len 5
    input_ids = torch.tensor([[10, 20, 30, 0, 0]])
    labels = torch.tensor([[10, 20, 30, -100, -100]])

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    loss = outputs["loss"]

    # To verify the ignore_index and shift works, we can calculate loss manually
    logits = outputs["logits"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    manual_loss = loss_fct(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))

    assert torch.isclose(loss, manual_loss)

    # The active labels are shift_labels: [20, 30, -100, -100]
    # So only 2 tokens (20, 30) should contribute to the loss.
    # We can check this by zeroing out the logits for the padded positions
    # and ensuring the loss doesn't change
    shift_logits_zeroed = shift_logits.clone()
    shift_logits_zeroed[0, 2:] = 0.0 # Zero out logits where label is -100

    zeroed_loss = loss_fct(shift_logits_zeroed.view(-1, config.vocab_size), shift_labels.view(-1))
    assert torch.isclose(loss, zeroed_loss)
