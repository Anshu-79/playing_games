import numpy as np
from scipy.optimize import minimize

def train_caviar(returns, alpha=0.05):
    def loss(params):
        b0,b1,b2,b3 = params
        T = len(returns)
        VaR = np.zeros(T)
        VaR[0] = np.quantile(returns, alpha)
        for t in range(1,T):
            r = returns[t-1]
            VaR[t] = b0 + b1*VaR[t-1] + b2*max(r,0) + b3*min(r,0)
        u = returns - VaR
        return np.sum(u*(alpha - (u<0)))

    res = minimize(loss, [0,0.9,0.1,0.1], method='Nelder-Mead')
    return res.x




def generate_caviar(params, returns, alpha=0.05):
    b0,b1,b2,b3 = params
    T = len(returns)
    VaR = np.zeros(T)
    VaR[0] = np.quantile(returns, alpha)
    for t in range(1,T):
        r = returns[t-1]
        VaR[t] = b0 + b1*VaR[t-1] + b2*max(r,0) + b3*min(r,0)
    return VaR
