from pathlib import Path

# Final cleaned version of the screener script as a string

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Auto-refresh every 5 minutes (300,000 ms) ---
st_autorefresh(interval=300000, key="datarefresh")

st.set_page_config(page_title="📈 NASDAQ Gainers Screener", layout="wide")
st.title("📈 NASDAQ Gainers Screener")
st.caption("Alerts on intraday gainers with RSI > 70, Market Cap < $50M, and custom fundamentals.")

@st.cache_data(ttl=3600)
def get_nasdaq_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[3]
    return table["Ticker"].tolist()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(ttl=300)
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d", interval="1m")
        info = stock.info

        if hist.empty:
            return None

        open_price = hist["Open"][0]
        current_price = hist["Close"][-1]
        gain_pct = ((current_price - open_price) / open_price) * 100

        rsi_series = calculate_rsi(hist["Close"])
        rsi = rsi_series.iloc[-1] if not rsi_series.empty else None

        market_cap = info.get("marketCap", 0) or 0
        cash = info.get("totalCash", 0) or 0
        shares_outstanding = info.get("sharesOutstanding", 0) or 0
        float_shares = info.get("floatShares", 0) or 0
        cash_burn = info.get("operatingCashflow", 0) or 0

        # Custom fields
        months_of_cash = round((cash / abs(cash_burn) * 12), 1) if cash_burn else None
        offering_ability = "High" if float_shares and float_shares / shares_outstanding > 0.7 else "Low"
        dilution_risk = "High" if market_cap < 30000000 and cash < 5000000 else "Low"
        cash_need = "Urgent" if months_of_cash and months_of_cash < 3 else "Moderate"

        return {
            "Ticker": ticker,
            "Gain %": round(gain_pct, 2),
            "RSI": round(rsi, 2) if rsi else None,
            "Market Cap": market_cap,
            "Offering Ability": offering_ability,
            "Dilution Risk": dilution_risk,
            "Cash Need": cash_need,
            "Months of Cash Left": months_of_cash,
            "Cash Position ($)": cash,
            "Float Shares": float_shares,
            "Total Outstanding Shares": shares_outstanding,
        }
    except Exception:
        return None

# --- Main Screener Logic ---
tickers = get_nasdaq_tickers()
results = []

progress = st.progress(0)
for i, ticker in enumerate(tickers):
    data = fetch_stock_data(ticker)
    if data:
        if data["Gain %"] >= 30 and data["RSI"] and data["RSI"] > 70 and data["Market Cap"] < 50000000:
            results.append(data)
    progress.progress((i + 1) / len(tickers))

# --- Display Results ---
if results:
    df = pd.DataFrame(results)
    df = df[["Ticker", "Offering Ability", "Dilution Risk", "Cash Need",
             "Gain %", "RSI", "Market Cap", "Months of Cash Left",
             "Float Shares", "Total Outstanding Shares", "Cash Position ($)"]]

    def highlight_offering(val):
        return "background-color: orange" if val == "High" else ""
    def highlight_dilution(val):
        return "background-color: red" if val == "High" else ""
    def highlight_cash_need(val):
        return "background-color: red" if val == "Urgent" else ""

    styled_df = (
        df.style
        .applymap(highlight_offering, subset=["Offering Ability"])
        .applymap(highlight_dilution, subset=["Dilution Risk"])
        .applymap(highlight_cash_need, subset=["Cash Need"])
    )

    st.success(f"✅ Found {len(df)} matching stocks.")
    st.dataframe(styled_df, use_container_width=True)

    # --- Download CSV ---
    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "screener_results.csv", "text/csv")

    # --- Plot Charts ---
    st.subheader("📉 Charts")
    for ticker in df["Ticker"]:
        with st.expander(f"Chart for {ticker}"):
            try:
                chart_data = yf.Ticker(ticker).history(period="5d", interval="15m")
                if chart_data.empty:
                    st.warning("No chart data available.")
                    continue

                fig = go.Figure(data=[
                    go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['Open'],
                        high=chart_data['High'],
                        low=chart_data['Low'],
                        close=chart_data['Close'],
                        name="Candlesticks"
                    )
                ])

                fig.update_layout(
                    title=f"{ticker} - 5D Candlestick Chart",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning(f"Could not load chart for {ticker}.")
else:
    st.warning("No stocks met the criteria today.")


