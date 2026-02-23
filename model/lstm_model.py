import numpy as np


def lstm_predict(close_prices):
    """
    Streamlit-safe lightweight prediction.
    Works without TensorFlow.
    """

    if close_prices is None:
        return None

    try:
        close_prices = np.array(close_prices, dtype=float)
        close_prices = close_prices[~np.isnan(close_prices)]
    except:
        return None

    if len(close_prices) < 60:
        return None

    # simple stable prediction (mean of last 60)
    last_60 = close_prices[-60:]
    prediction = np.mean(last_60)

    return float(prediction)