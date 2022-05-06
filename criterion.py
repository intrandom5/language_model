import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, tokenizer):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            reduction="mean",
            ignore_index=tokenizer.pad_token,
        )
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        max_target_length = targets.size(1)
        max_logits_length = logits.size(1)
        
        if max_logits_length > max_target_length:
            logits = logits[:, :max_target_length, :]
        elif max_target_length > max_logits_length:
            targets = targets[:, :max_logits_length]
            
        logits = logits.contiguous().view(-1, logits.size(-1))
        
        return self.cross_entropy_loss(
            logits.contiguous().view(-1, logits.size(-1)),
            targets.contiguous().view(-1),
        )
