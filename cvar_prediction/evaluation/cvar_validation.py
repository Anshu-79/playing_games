import numpy as np


def tail_mae(returns, var, cvar):
    errors = []
    for t in range(len(returns)):
        tail = returns[:t+1][returns[:t+1] <= var[t]]
        if len(tail) > 5:
            errors.append(abs(tail.mean() - cvar[t]))
    return np.mean(errors)
