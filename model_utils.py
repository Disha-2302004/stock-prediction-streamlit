from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
import pandas as pd

def engineer_features(df):
    df = df.copy()

    # 🔥 FORCE ALL PRICE COLUMNS TO 1D SERIES
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    open_ = df["Open"].squeeze()
    volume = df["Volume"].squeeze()

    # Core features
    df["return"] = close.pct_change()
    df["volatility"] = df["return"].rolling(5).std()

    # ✅ FIXED VWAP (now all 1D)
    df["vwap"] = (high + low + close) / 3

    # Moving averages
    df["sma5"] = SMAIndicator(close=close, window=5).sma_indicator()
    df["sma20"] = SMAIndicator(close=close, window=20).sma_indicator()
    df["momentum"] = df["sma5"] - df["sma20"]

    # Indicators
    df["rsi"] = RSIIndicator(close=close, window=14).rsi()
    df["macd"] = MACD(close=close).macd()

    # Targets
    df["target_price"] = close.shift(-1)
    df["direction"] = (df["target_price"] > close).astype(int)

    df.dropna(inplace=True)
    return df
