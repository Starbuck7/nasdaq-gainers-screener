import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh


# --- SETTINGS ---
st.set_page_config(page_title="NASDAQ Screener", layout="wide")

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
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Refresh every 5 minutes (300,000 ms)
st_autorefresh(interval=5 * 60 * 1000, key="refresh")

# --- LOAD NASDAQ TICKERS ---
@st.cache_data
def load_nasdaq():
    url = 'https://datahub.io/core/nasdaq-listings/r/nasdaq-listed.csv'
    df = pd.read_csv(url)
    return df['Symbol'].tolist()

tickers = load_nasdaq()
st.sidebar.write(f"Loaded {len(tickers)} NASDAQ tickers.")

# --- FILTERING ---
run_scan = st.button("🔍 Run Scan")

results = []
if run_scan:
    with st.spinner("Scanning stocks (please wait 1-2 mins)..."):
        for ticker in tickers[:100]:  # Limit for demo
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d", interval="5m")
                if len(hist) < 2:
                    continue

                open_price = hist['Open'][0]
                current_price = hist['Close'][-1]
                gain_pct = ((current_price - open_price) / open_price) * 100
                rsi_series = calculate_rsi(hist['Close'])
                rsi = rsi_series.iloc[-1] if not rsi_series.empty else None
                info = stock.info
                market_cap = info.get("marketCap", 0)

                if gain_pct >= 30 and rsi and rsi > 70 and market_cap and market_cap < 5e7:
                    results.append({
                        "Ticker": ticker,
                        "Price": round(current_price, 2),
                        "Gain %": round(gain_pct, 2),
                        "RSI": round(rsi, 2),
                        "Market Cap": market_cap
                    })

            except Exception as e:
                continue
    
    # Display results
    if results:
        df = pd.DataFrame(results)
        st.success(f"Found {len(df)} matching stocks!")
        st.dataframe(df)

        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, file_name="stock_alerts.csv")
    else:
        st.warning("No matching stocks found.")
