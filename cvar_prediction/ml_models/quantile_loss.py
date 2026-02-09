import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def forward(self, y_pred, y_true):
        e = y_true - y_pred
        return torch.mean(torch.max(self.q*e, (self.q-1)*e))
