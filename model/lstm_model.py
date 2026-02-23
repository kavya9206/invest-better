import numpy as np

def lstm_predict(close_prices):

    if close_prices is None:
        return None

    try:
        close_prices = np.array(close_prices, dtype=float)
        close_prices = close_prices[~np.isnan(close_prices)]
    except:
        return None

    if len(close_prices) < 60:
        return None

    last_60 = close_prices[-60:]
    prediction = np.mean(last_60)

    return float(prediction)