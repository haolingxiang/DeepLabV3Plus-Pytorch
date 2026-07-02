import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

        if isinstance(alpha, (float, int)):
            self.register_buffer('alpha', torch.tensor(float(alpha)))
            self.per_class_alpha = False
        else:
            self.register_buffer('alpha', alpha.float())
            self.per_class_alpha = True

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.per_class_alpha:
            alpha_factor = self.alpha.to(inputs.device)[targets]
            alpha_factor = alpha_factor.masked_fill(targets == self.ignore_index, 0.)
            focal_loss = alpha_factor * focal_loss
        else:
            focal_loss = self.alpha * focal_loss

        valid = targets != self.ignore_index
        if not valid.any():
            return focal_loss.sum() * 0.0

        focal_loss = focal_loss[valid]
        if self.size_average:
            return focal_loss.mean()
        return focal_loss.sum()
