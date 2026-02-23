import numpy as np

# ⭐ LIGHTWEIGHT prediction (no training)
def lstm_predict(close_prices):

    if len(close_prices) < 10:
        return close_prices[-1]

    last_prices = close_prices[-10:]
    trend = np.mean(np.diff(last_prices))

    return close_prices[-1] + trend