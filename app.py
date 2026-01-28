import streamlit as st
import plotly.graph_objects as go
from model_utils import fetch_data, engineer_features, generate_signal


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Professional Stock Dashboard",
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
st.markdown("<h1 style='text-align:center;'>📈 Stock Trading Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#9ca3af;'>Technical Indicator Based Buy / Sell System</p>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("---")

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX"]
ticker = st.sidebar.selectbox("Select Company", TICKERS)

predict_btn = st.sidebar.button("🔮 Generate Signal")

# ---------------- MAIN ----------------
if predict_btn:

    with st.spinner("Analyzing market data..."):
        df = fetch_data(ticker)
        df_feat = engineer_features(df)
        result = generate_signal(df_feat)

    # ---------------- METRICS ----------------
    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Current Price</div>
        <div class="metric-value">${result['current_price']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">RSI</div>
        <div class="metric-value">{result['rsi']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">SMA 5</div>
        <div class="metric-value">{result['sma5']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    c4.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Signal</div>
        <div class="{result['signal'].lower()}">{result['signal']}</div>
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
        title=f"{ticker} Price Trend & Indicators",
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📁 View Recent Data"):
        st.dataframe(df_feat.tail(30), use_container_width=True)

else:
    st.info("👈 Select a company and click **Generate Signal**")
