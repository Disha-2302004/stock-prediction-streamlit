import streamlit as st
import plotly.graph_objects as go

from model_utils import (
    load_model,
    fetch_data,
    engineer_features,
    predict_price,
    calculate_dashboard_metrics
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Professional Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: white; }
[data-testid="stSidebar"] { background-color: #161b22; }
.metric-box {
    background: #161b22;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 0 15px rgba(0,255,255,0.2);
}
.metric-title { color: #9ca3af; font-size: 15px; }
.metric-value { font-size: 28px; font-weight: bold; color: #00f2ff; }
.buy { color: #00ff9c; font-size: 30px; font-weight: bold; }
.sell { color: #ff4d4d; font-size: 30px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>📈 Stock Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#9ca3af;'>AI-Driven Buy / Sell Decision System</p>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("---")

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX"]
ticker = st.sidebar.selectbox("Select Company", TICKERS)
predict_btn = st.sidebar.button("🔮 Predict")

# ---------------- LOAD MODEL ----------------
model = load_model()

# ---------------- MAIN ----------------
if predict_btn:

    with st.spinner("Running prediction..."):
        df = fetch_data(ticker)
        df_feat = engineer_features(df)
        pred_price, signal = predict_price(model, df_feat)
        metrics = calculate_dashboard_metrics(df_feat, pred_price)

    signal_text = "BUY" if signal == 1 else "SELL"
    signal_class = "buy" if signal == 1 else "sell"

    # ---------------- METRICS ----------------
    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Current Price</div>
        <div class="metric-value">${metrics['current_price']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Predicted Price</div>
        <div class="metric-value">${metrics['predicted_price']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Expected Change</div>
        <div class="metric-value">{metrics['pct_change']:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    c4.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Signal</div>
        <div class="{signal_class}">{signal_text}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------------- CHART ----------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_feat["Date"],
        y=df_feat["Close"],
        name="Close Price",
        line=dict(color="#00f2ff", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df_feat["Date"],
        y=df_feat["sma5"],
        name="SMA 5",
        line=dict(color="#ffaa00", dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=df_feat["Date"],
        y=df_feat["sma10"],
        name="SMA 10",
        line=dict(color="#ff4d4d", dash="dot")
    ))

    fig.update_layout(
        title=f"{ticker} Price & Indicators",
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- INDICATORS ----------------
    st.subheader("📊 Key Indicators")
    i1, i2, i3 = st.columns(3)
    i1.metric("RSI", f"{metrics['rsi']:.2f}")
    i2.metric("Volatility", f"{metrics['volatility']:.4f}")
    i3.metric("EMA/SMA Trend", "Bullish" if metrics["sma5"] > metrics["sma10"] else "Bearish")

    with st.expander("📁 View Data"):
        st.dataframe(df_feat.tail(30), use_container_width=True)

else:
    st.info("👈 Select a company and click **Predict** to generate insights")
