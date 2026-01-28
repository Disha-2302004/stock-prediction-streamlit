import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator


# ---------------- FETCH DATA ----------------
def fetch_data(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)

    # Force numeric + 1D safety
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------- FEATURE ENGINEERING ----------------
def engineer_features(df):
    df = df.copy()

    # 🔴 CRITICAL FIX: force 1D array
    close = df["Close"].values.flatten()

    df["sma5"] = SMAIndicator(pd.Series(close), window=5).sma_indicator()
    df["sma10"] = SMAIndicator(pd.Series(close), window=10).sma_indicator()
    df["rsi"] = RSIIndicator(pd.Series(close), window=14).rsi()

    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)

    return df


# ---------------- SIGNAL LOGIC ----------------
def generate_signal(df):
    latest = df.iloc[-1]

    buy_condition = (
        latest["sma5"] > latest["sma10"]
        and latest["rsi"] < 70
    )

    signal = "BUY" if buy_condition else "SELL"

    return {
        "signal": signal,
        "current_price": float(latest["Close"]),
        "rsi": float(latest["rsi"]),
        "sma5": float(latest["sma5"]),
        "sma10": float(latest["sma10"])
    }
