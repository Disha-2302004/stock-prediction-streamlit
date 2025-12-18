import streamlit as st
import plotly.graph_objects as go
from model_utils import *

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

st.title("📈 Multi-Company Stock Prediction Dashboard")

# Sidebar
ticker = st.sidebar.selectbox("Select Company", TICKERS)

st.sidebar.info("ML Model: XGBoost\n\nTimeframe: Daily")

# Load Data
df = fetch_data(ticker)
df_feat = engineer_features(df)

# Train model
with st.spinner("Training model..."):
    reg, cls, scaler, features = train_model(df_feat)

# Prediction
pred_price, pred_dir = predict_next(df_feat, reg, cls, scaler, features)

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"{df_feat['Close'].iloc[-1]:.2f}")
col2.metric("Predicted Next Price", f"{pred_price:.2f}")
col3.metric("Signal", "📈 BUY" if pred_dir==1 else "📉 SELL")

# Line Chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_feat["Date"],
    y=df_feat["Close"],
    name="Close Price"
))
st.plotly_chart(fig, use_container_width=True)

# Show raw data
with st.expander("Show Raw Data"):
    st.dataframe(df_feat.tail(50))
