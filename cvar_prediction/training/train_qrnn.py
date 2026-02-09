import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import logging

from ml_models.qrnn import QRNN
from ml_models.quantile_loss import QuantileLoss
from features.feature_engineering import build_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QRNN-TRAIN")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
df = pd.read_csv("data/raw/nifty.csv")

# Build features
df = build_features(df)

features = ["log_return", "vol_5", "vol_20", "vol_60", "drawdown"]
X = df[features].values
y = df["log_return"].values

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

model = QRNN(X_train.shape[1]).to(device)

loss1 = QuantileLoss(0.01)
loss5 = QuantileLoss(0.05)
loss10 = QuantileLoss(0.10)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    total = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        q1, q5, q10 = model(xb)
        l = loss1(q1, yb) + loss5(q5, yb) + loss10(q10, yb)
        l.backward()
        optimizer.step()
        total += l.item()
    logger.info(f"Epoch {epoch+1}, Loss {total/len(loader):.6f}")

# Save model
torch.save(model.state_dict(), "experiments/qrnn_model.pt")
logger.info("QRNN model saved")
