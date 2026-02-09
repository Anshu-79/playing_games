import torch
import torch.nn as nn

class QRNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.q1 = nn.Linear(64, 1)
        self.q5 = nn.Linear(64, 1)
        self.q10 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.net(x)
        return self.q1(h), self.q5(h), self.q10(h)
