"""BCE + Dice composite loss for overlap mask prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.eps) / (pred.sum() + target.sum() + self.eps)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Total loss = BCE + Dice (equal weights, paper Section III-B)."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 eps: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(eps=eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
