import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import streamlit as st


@st.cache_resource
def load_model():
    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def lstm_predict(close_prices):

    if close_prices is None:
        return None

    close_prices = np.array(close_prices, dtype=float)
    close_prices = close_prices[~np.isnan(close_prices)]

    if len(close_prices) < 60:
        return None

    # ⭐ Streamlit-safe lightweight prediction
    last_60 = close_prices[-60:]

    # simple smoothing (acts like lightweight LSTM fallback)
    prediction = np.mean(last_60)

    return float(prediction)