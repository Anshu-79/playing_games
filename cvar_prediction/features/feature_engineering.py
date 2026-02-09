import numpy as np
import pandas as pd

def build_features(df):
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    
    r = df['log_return']
    
    df['vol_5'] = r.rolling(5).std()
    df['vol_20'] = r.rolling(20).std()
    df['vol_60'] = r.rolling(60).std()
    
    df['skew_20'] = r.rolling(20).skew()
    df['kurt_20'] = r.rolling(20).kurt()
    
    df['momentum_5'] = r.rolling(5).mean()
    df['momentum_20'] = r.rolling(20).mean()
    
    df['cum_return'] = (1+r).cumprod()
    df['drawdown'] = df['cum_return']/df['cum_return'].cummax() - 1
    
    return df.dropna().reset_index(drop=True)
