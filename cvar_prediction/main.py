import logging
import pandas as pd
import numpy as np
import torch

from risk_models.caviar import train_caviar, generate_caviar
from features.feature_engineering import build_features
from evaluation.cvar_validation import tail_mae

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)

logger = logging.getLogger("PIPELINE")

logger.info("Starting CVaR pipeline")

# 1. Load data
df = pd.read_csv('data/raw/nifty.csv')
logger.info("Data loaded")

# 2. Feature engineering
df = build_features(df)
logger.info("Features built")

returns = df['log_return'].values

# 3. Train CaViAR
params = train_caviar(returns, alpha=0.05)
logger.info(f"CaViAR trained: {params}")

# 4. Generate VaR
var_series = generate_caviar(params, returns, alpha=0.05)
logger.info("VaR series generated")

# 5. Compute CVaR
cvar_series = []
for t in range(len(returns)):
    tail = returns[:t+1][returns[:t+1] <= var_series[t]]
    if len(tail) > 5:
        cvar_series.append(tail.mean())
    else:
        cvar_series.append(np.nan)

cvar_series = np.array(cvar_series)
logger.info("CVaR series computed")

# 6. Validation
mae = tail_mae(returns, var_series, cvar_series)
logger.info(f"Tail MAE (CVaR): {mae}")

# 7. Save results
out = pd.DataFrame({
    'returns': returns,
    'VaR': var_series,
    'CVaR': cvar_series
})

out.to_csv('experiments/results.csv', index=False)
logger.info("Pipeline results saved")

logger.info("Pipeline completed successfully")
