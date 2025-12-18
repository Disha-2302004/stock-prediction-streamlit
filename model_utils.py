import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

TICKERS = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]

def fetch_data(ticker):
    df = yf.download(ticker, start="2022-01-01", end=datetime.today())
    df.reset_index(inplace=True)
    return df

def engineer_features(df):
    df["return"] = df["Close"].pct_change()
    df["volatility"] = df["return"].rolling(5).std()
    df["vwap"] = (df["High"] + df["Low"] + df["Close"]) / 3

    df["sma5"] = SMAIndicator(df["Close"], 5).sma_indicator()
    df["sma20"] = SMAIndicator(df["Close"], 20).sma_indicator()
    df["momentum"] = df["sma5"] - df["sma20"]

    df["rsi"] = RSIIndicator(df["Close"]).rsi()
    df["macd"] = MACD(df["Close"]).macd()

    df["target_price"] = df["Close"].shift(-1)
    df["direction"] = (df["target_price"] > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df

def train_model(df):
    features = ["Open","High","Low","Close","Volume","vwap","return",
                "volatility","sma5","sma20","momentum","rsi","macd"]

    X = df[features]
    y_reg = df["target_price"]
    y_cls = df["direction"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05)
    cls = xgb.XGBClassifier(n_estimators=300)

    reg.fit(X_scaled, y_reg)
    cls.fit(X_scaled, y_cls)

    return reg, cls, scaler, features

def predict_next(df, reg, cls, scaler, features):
    last_row = df.iloc[-1:][features]
    scaled = scaler.transform(last_row)

    price = reg.predict(scaled)[0]
    direction = cls.predict(scaled)[0]

    return price, direction
