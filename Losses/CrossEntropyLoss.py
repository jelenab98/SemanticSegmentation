from typing import List

import torch.nn as nn
from torch.nn import functional as F

__all__ = ["CrossEntropyLoss", "WeightedCrossEntropyLoss"]


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int = 2,
                 ignore_id: int = 0,
                 print_step: int = 25
                 ) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.print_counter = 0
        self.print_step = print_step

    def get_loss(self, y, y_gt):
        return F.cross_entropy(input=y, target=y_gt, ignore_index=self.ignore_id)

    def forward(self, logits, labels, **kwargs):
        loss = self.get_loss(logits, labels)
        if (self.print_counter % self.print_step) == 0:
            print(f"Loss: {loss.data.cpu().item():.4f}")
        self.print_counter += 1
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int = 2,
                 ignore_id: int = 0,
                 print_step: int = 20,
                 weights: List = None,
                 ) -> None:
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.print_counter = 0
        self.print_step = print_step
        self.weights = weights

    def get_loss(self, y, y_gt):
        return F.cross_entropy(input=y, target=y_gt, ignore_index=self.ignore_id, weight=self.weights)

    def forward(self, logits, labels, **kwargs):
        loss = self.get_loss(logits, labels)
        if (self.print_counter % self.print_step) == 0:
            print(f"Loss: {loss.data.cpu().item():.4f}")
        self.print_counter += 1
        return loss
