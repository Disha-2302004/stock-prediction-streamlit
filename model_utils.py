import pandas as pd
import numpy as np
import yfinance as yf
import joblib

from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator


# ---------------- LOAD MODEL ----------------
def load_model():
    return joblib.load("xgboost_model.pkl")


# ---------------- FETCH DATA ----------------
def fetch_data(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    return df


# ---------------- FEATURE ENGINEERING ----------------
def engineer_features(df):
    df = df.copy()

    close = df["Close"]

    df["sma5"] = SMAIndicator(close, window=5).sma_indicator()
    df["sma10"] = SMAIndicator(close, window=10).sma_indicator()
    df["ema5"] = EMAIndicator(close, window=5).ema_indicator()
    df["rsi"] = RSIIndicator(close, window=14).rsi()

    df["vwap"] = (df["High"] + df["Low"] + close) / 3
    df["returns"] = close.pct_change()
    df["volatility"] = df["returns"].rolling(10).std()

    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)

    return df


# ---------------- PREDICTION ----------------
def predict_price(model, df):
    feature_cols = [
        "Close",
        "sma5",
        "sma10",
        "ema5",
        "rsi",
        "vwap",
        "volatility"
    ]

    X = df[feature_cols].iloc[-1:].values
    predicted_price = float(model.predict(X)[0])

    current_price = float(df["Close"].iloc[-1])
    signal = 1 if predicted_price > current_price else 0

    return predicted_price, signal


# ---------------- DASHBOARD METRICS ----------------
def calculate_dashboard_metrics(df, predicted_price):
    current_price = float(df["Close"].iloc[-1])

    price_change = predicted_price - current_price
    pct_change = (price_change / current_price) * 100

    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "pct_change": pct_change,
        "rsi": float(df["rsi"].iloc[-1]),
        "sma5": float(df["sma5"].iloc[-1]),
        "sma10": float(df["sma10"].iloc[-1]),
        "volatility": float(df["volatility"].iloc[-1])
    }
