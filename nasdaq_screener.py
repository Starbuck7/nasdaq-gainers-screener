import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from yahoo_fin import stock_info as si

# Auto-refresh every 5 minutes
st.set_page_config(page_title="📈 NASDAQ Stock Screener", layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="refresh")

# --- HEADER ---
st.title("🚨 Real-Time NASDAQ Stock Screener")
st.markdown("""
Scan for **stocks with:**
- Daily gain ≥ 30%
- RSI > 70
- Market cap < $50M
""")

# --- RSI FUNCTION ---
def calculate_rsi(series, period=14):
    if series is None or series.empty:
        return pd.Series(dtype=float)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

#Color Styling to Columns
def highlight_offering(val):
    return "background-color: orange" if val == "High" else ""

def highlight_dilution(val):
    return "background-color: red" if val == "High" else ""

def highlight_cash_need(val):
    return "background-color: red" if val == "Urgent" else ""

# --- LOAD NASDAQ TICKERS ---
@st.cache_data
def get_nasdaq_gainers():
    try:
        df = si.get_day_gainers()
        nasdaq_gainers = df[df['Exchange'] == 'NASDAQ']['Symbol'].tolist()
        return nasdaq_gainers
    except Exception as e:
        st.error(f"Error fetching gainers: {e}")
        return []

# Load Tickers
tickers = get_nasdaq_gainers()
st.sidebar.write(f"Loaded {len(tickers)} NASDAQ gainers today.")

# --- RUN SCAN ---
run_scan = st.button("🔍 Run Scan")
results = []

if run_scan:
    with st.spinner("Scanning stocks (please wait 1-2 mins)..."):
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d", interval="5m")
                info = stock.info
                if hist.empty or "Open" not in hist.columns or "Close" not in hist.columns:
                    continue
                open_price = hist["Open"][0]
                close_price = hist["Close"].iloc[-1]
                gain_pct = ((close_price - open_price) / open_price) * 100
                if gain_pct < 30:
                    continue

                rsi_hist = stock.history(period="15d", interval="1d")
                rsi_series = calculate_rsi(rsi_hist["Close"])
                rsi = rsi_series.iloc[-1] if not rsi_series.empty else 0
                if rsi <= 70:
                    continue

                market_cap = info.get("marketCap", 0)
                if market_cap is None or market_cap >= 50_000_000:
                    continue

                cash = info.get("totalCash", None)
                operating_expenses = info.get("totalOperatingExpenses", None)
                shares_outstanding = info.get("sharesOutstanding", None)
                float_shares = info.get("floatShares", None)
                months_cash_left = None
                if cash and operating_expenses and operating_expenses > 0:
                    months_cash_left = round(cash / (operating_expenses / 12), 1)

                offering_ability = "High" if float_shares and float_shares > 0.5 * shares_outstanding else "Low"
                dilution_risk = "High" if cash and cash < 10_000_000 and float_shares and float_shares > 0.7 * shares_outstanding else "Moderate"
                cash_need = "Urgent" if months_cash_left and months_cash_left < 3 else "Moderate"

                results.append({
                    "Offering Ability": offering_ability,
                    "Dilution Risk": dilution_risk,
                    "Cash Need": cash_need,
                    "Ticker": ticker,
                    "Price": round(close_price, 2),
                    "Gain %": round(gain_pct, 2),
                    "RSI": round(rsi, 2),
                    "Market Cap": market_cap,
                    "Cash ($)": cash,
                    "Months Cash Left": months_cash_left,
                    "Float Shares": float_shares,
                    "Total Outstanding Shares": shares_outstanding,
                    "Cash Position ($)": cash
                })

            except Exception as e:
                st.warning(f"Error processing {ticker}: {e}")
                continue

    if results:
        df = pd.DataFrame(results)
        st.success(f"✅ Found {len(df)} matching stock(s).")
        df = df[["Ticker", "Offering Ability", "Dilution Risk", "Cash Need",
                 "Gain %", "RSI", "Market Cap", "Months Cash Left",
                 "Float Shares", "Total Outstanding Shares", "Cash Position ($)"]]

        styled_df = (
            df.style
            .applymap(highlight_offering, subset=["Offering Ability"])
            .applymap(highlight_dilution, subset=["Dilution Risk"])
            .applymap(highlight_cash_need, subset=["Cash Need"])
        )

        st.dataframe(styled_df, use_container_width=True)

        # Download
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "screener_results.csv", "text/csv")

        # Charts
        st.subheader("📉 Price Charts")
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
                        title=f"{ticker} - 5 Day Candlestick Chart",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        xaxis_rangeslider_visible=False,
                        height=400,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.warning(f"Could not load chart for {ticker}")
    else:
        st.warning("No matching stocks found.")
