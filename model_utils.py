import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator


# ---------------- FETCH DATA ----------------
def fetch_data(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)

    # ✅ FIX 1: Flatten MultiIndex columns (CRITICAL)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    # ✅ FIX 2: Ensure numeric 1D Series
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    return df


# ---------------- FEATURE ENGINEERING ----------------
def engineer_features(df):
    df = df.copy()

    # ✅ FIX 3: force Close to strict 1D
    close = df["Close"].to_numpy().flatten()

    close_series = pd.Series(close)

    df["sma5"] = SMAIndicator(close_series, window=5).sma_indicator()
    df["sma10"] = SMAIndicator(close_series, window=10).sma_indicator()
    df["rsi"] = RSIIndicator(close_series, window=14).rsi()

    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)

    return df


# ---------------- SIGNAL LOGIC ----------------
def generate_signal(df):
    latest = df.iloc[-1]

    buy = latest["sma5"] > latest["sma10"] and latest["rsi"] < 70
    signal = "BUY" if buy else "SELL"

    return {
        "signal": signal,
        "current_price": float(latest["Close"]),
        "rsi": float(latest["rsi"]),
        "sma5": float(latest["sma5"]),
        "sma10": float(latest["sma10"])
    }
