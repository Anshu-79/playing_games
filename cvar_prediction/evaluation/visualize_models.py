import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from ml_models.qrnn import QRNN
from ml_models.lstm_quantile import LSTMQuantile
from risk_models.caviar import train_caviar, generate_caviar
from features.feature_engineering import build_features
from features.windowing import make_sequences
from evaluation.cvar_validation import tail_mae

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EVAL-VIZ")

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# Load data
# --------------------
df = pd.read_csv("data/raw/nifty.csv")
df = build_features(df)

features = ["log_return", "vol_5", "vol_20", "vol_60", "drawdown"]
X = df[features].values
y = df["log_return"].values
returns = y.copy()

# --------------------
# Load QRNN
# --------------------
qrnn = QRNN(len(features)).to(device)
qrnn.load_state_dict(torch.load("experiments/qrnn_model.pt", map_location=device))
qrnn.eval()

with torch.no_grad():
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    q1_qrnn, q5_qrnn, q10_qrnn = qrnn(X_t)

q5_qrnn = q5_qrnn.cpu().numpy().flatten()

# --------------------
# Load LSTM
# --------------------
X_seq, y_seq = make_sequences(X, y, window=60)

lstm = LSTMQuantile(input_dim=X_seq.shape[-1]).to(device)
lstm.load_state_dict(
    torch.load("experiments/lstm_quantile_model.pt", map_location=device)
)
lstm.eval()

with torch.no_grad():
    X_seq_t = torch.tensor(X_seq, dtype=torch.float32).to(device)
    q1_lstm, q5_lstm, q10_lstm = lstm(X_seq_t)

q5_lstm = q5_lstm.cpu().numpy().flatten()

# Align lengths
qrnn_var = q5_qrnn[-len(q5_lstm) :]
returns_aligned = returns[-len(q5_lstm) :]

# --------------------
# CaViAR
# --------------------
params = train_caviar(returns, alpha=0.05)
caviar_var = generate_caviar(params, returns, alpha=0.05)
caviar_var = caviar_var[-len(q5_lstm) :]


# --------------------
# CVaR computation
# --------------------
def compute_cvar(ret, var):
    cvar = []
    for i in range(len(ret)):
        tail = ret[: i + 1][ret[: i + 1] <= var[i]]
        if len(tail) > 5:
            cvar.append(tail.mean())
        else:
            cvar.append(np.nan)
    return np.array(cvar)


qrnn_cvar = compute_cvar(returns_aligned, qrnn_var)
lstm_cvar = compute_cvar(returns_aligned, q5_lstm)
caviar_cvar = compute_cvar(returns_aligned, caviar_var)

# --------------------
# Metrics
# --------------------
logger.info(f"QRNN Tail MAE: {tail_mae(returns_aligned, qrnn_var, qrnn_cvar)}")
logger.info(f"LSTM Tail MAE: {tail_mae(returns_aligned, q5_lstm, lstm_cvar)}")
logger.info(f"CaViAR Tail MAE: {tail_mae(returns_aligned, caviar_var, caviar_cvar)}")

# --------------------
# Visualization
# --------------------
plt.figure(figsize=(14, 6))
plt.plot(returns_aligned, label="Returns", alpha=0.4)
plt.plot(qrnn_var, label="QRNN VaR (5%)")
plt.plot(q5_lstm, label="LSTM VaR (5%)")
plt.plot(caviar_var, label="CaViAR VaR (5%)")
plt.legend()
plt.title("VaR Comparison")
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(qrnn_cvar, label="QRNN CVaR")
plt.plot(lstm_cvar, label="LSTM CVaR")
plt.plot(caviar_cvar, label="CaViAR CVaR")
plt.legend()
plt.title("CVaR Comparison")
plt.show()

# --------------------
# Save evaluation
# --------------------
out = pd.DataFrame(
    {
        "returns": returns_aligned,
        "VaR_QRNN": qrnn_var,
        "VaR_LSTM": q5_lstm,
        "VaR_CaViAR": caviar_var,
        "CVaR_QRNN": qrnn_cvar,
        "CVaR_LSTM": lstm_cvar,
        "CVaR_CaViAR": caviar_cvar,
    }
)

out.to_csv("experiments/model_comparison.csv", index=False)
logger.info("Model comparison saved to experiments/model_comparison.csv")
