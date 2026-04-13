import torch
import torch.nn.functional as F

def kl_distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Standard KL divergence loss for logit distillation.
    """
    # Scale logits by temperature
    p_s = F.log_softmax(student_logits / temperature, dim=-1)
    p_t = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL Divergence
    loss = F.kl_div(p_s, p_t, reduction="batchmean") * (temperature ** 2)
    return loss

def hidden_state_mse_loss(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
    """
    MSE loss for hidden state alignment.
    Assumes student and teacher have the same hidden dimension.
    """
    return F.mse_loss(student_hidden, teacher_hidden)
