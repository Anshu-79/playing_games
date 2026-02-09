import torch
import torch.nn as nn


class LSTMQuantile(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )

        self.q1 = nn.Linear(64, 1)
        self.q5 = nn.Linear(64, 1)
        self.q10 = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = self.fc(h)
        return self.q1(h), self.q5(h), self.q10(h)
