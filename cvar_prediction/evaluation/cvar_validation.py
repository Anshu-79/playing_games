import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VAL")

def compute_tail_metrics(returns, var_series, true_cvar, pred_cvar):
    """
    Proper CVaR tail evaluation.
    """

    # --- alignment safety ---
    n = min(len(returns), len(var_series), len(true_cvar), len(pred_cvar))
    returns = returns[-n:]
    var_series = var_series[-n:]
    true_cvar = true_cvar[-n:]
    pred_cvar = pred_cvar[-n:]

    # --- tail region ---
    losses = -returns
    tail_mask = losses >= var_series
    tail_mask = tail_mask & (losses > 0)   # enforce loss-only tail

    tail_count = int(tail_mask.sum())

    logger.info(f"Tail samples: {tail_count}/{n}")

    if tail_count == 0:
        raise RuntimeError("No tail events detected — invalid CVaR evaluation")

    # --- tail data ---
    tail_true_cvar = true_cvar[tail_mask]
    tail_pred_cvar = pred_cvar[tail_mask]
    tail_returns = returns[tail_mask]

    # --- metrics ---
    tail_mae = np.mean(np.abs(tail_pred_cvar - tail_true_cvar))
    tail_rmse = np.sqrt(np.mean((tail_pred_cvar - tail_true_cvar) ** 2))
    exceedance_loss = np.mean(np.maximum(0, tail_returns - tail_pred_cvar))
    calibration_error = np.mean(tail_pred_cvar - tail_true_cvar)

    metrics = {
        "tail_samples": tail_count,
        "tail_mae": float(tail_mae),
        "tail_rmse": float(tail_rmse),
        "exceedance_loss": float(exceedance_loss),
        "calibration_error": float(calibration_error),
    }

    return metrics
