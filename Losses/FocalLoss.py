import torch
import torch.nn as nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self,
                 gamma: float = 1e-3,
                 alpha: float = 0.0,
                 num_classes: int = 2,
                 ignore_id: int = 0,
                 print_step: int = 25
                 ) -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.Tensor([alpha, 1 - alpha])
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.print_step = print_step
        self.print_counter = 0

    def get_loss(self, y: torch.Tensor, y_gt: torch.Tensor):
        y_gt[y_gt == self.ignore_id] = 0
        if y.dim() > 2:
            y = y.view(y.size(0), y.size(1), -1)
            y = y.transpose(1, 2)
            y = y.contiguous().view(-1, y.size(2))
        y_gt = y_gt.view(-1, 1)

        log_probits = F.log_softmax(y, dim=-1)
        log_probits = log_probits.gather(1, y_gt)
        log_probits = log_probits.view(-1)
        probits = log_probits.detach().exp()

        return (-1 * self.alpha * torch.exp(self.gamma * (1 - probits)) * log_probits).sum() / \
               ((self.alpha.data > 0).sum())

    def forward(self, logits, labels, **kwargs):
        loss = self.get_loss(logits, labels)

        if (self.print_counter % self.print_step) == 0:
            print(f"Loss: {loss.data.cpu().item()}:.4f")
        self.print_counter += 1

        return loss
