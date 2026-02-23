import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


# ---------- Load model (cached) ----------
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


# ---------- LSTM prediction ----------
def lstm_predict(close_prices):

    # ✅ Safety check
    if close_prices is None or len(close_prices) < 60:
        return None

    close_prices = np.array(close_prices)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)

    # ✅ Another safety check
    if len(X) == 0:
        return None

    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = load_model()

    # ⚠️ Lightweight training (kept minimal for Streamlit)
    model.fit(X, y, epochs=1, batch_size=32, verbose=0)

    # ---------- Predict ----------
    last_60 = scaled[-60:].reshape(1, 60, 1)
    pred = model.predict(last_60, verbose=0)

    prediction = scaler.inverse_transform(pred)[0][0]

    return float(prediction)