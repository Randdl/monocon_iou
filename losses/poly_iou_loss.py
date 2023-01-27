import os
import sys
import torch
import torch.nn as nn

from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from losses.utils import weighted_loss
from losses.fixed_polygon_iou_loss import batch_poly_diou_loss


class IOULoss(nn.Module):
    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,) -> torch.Tensor:
        pred = pred.view(-1, 4, 2).contiguous()
        target = target.view(-1, 4, 2).contiguous()
        return self.loss_weight * batch_poly_diou_loss(pred, target, a=0).mean()
