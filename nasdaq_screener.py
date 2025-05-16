import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

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

#Add chart as subheader
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

# --- RSI FUNCTION ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def human_format(num):
    if num is None:
        return "-"
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000:
            return f"{num:.2f}"
        num /= 1000
    return f"{num:.2f}T"

#Color Styling to Columns
def highlight_offering(val):
    if val == "High":
        return "background-color: #d4edda; color: #155724"  # green
    return "background-color: #f8d7da; color: #721c24"      # red

def highlight_dilution(val):
    if val == "High":
        return "background-color: #f8d7da; color: #721c24"  # red
    return "background-color: #fff3cd; color: #856404"      # yellow

def highlight_cash_need(val):
    if val == "Urgent":
        return "background-color: #f8d7da; color: #721c24"  # red
    return "background-color: #d4edda; color: #155724"      # green
    
# --- LOAD NASDAQ TICKERS ---
@st.cache_data
def load_nasdaq():
    url = 'https://datahub.io/core/nasdaq-listings/r/nasdaq-listed.csv'
    df = pd.read_csv(url)
    return df["Ticker"].tolist()

tickers = load_nasdaq()
st.sidebar.write(f"Loaded {len(tickers)} NASDAQ tickers.")

# --- FILTERING ---
run_scan = st.button("🔍 Run Scan")

results = []
if run_scan:
    with st.spinner("Scanning stocks (please wait 1-2 mins)..."):
        for ticker in tickers:  # Limit for demo
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d", interval="5m")
                info = stock.info
                if hist.empty:
                    continue    
            except Exception as e:
                continue
                
# Prices & RSI
    open_price = hist["Open"][0]
    current_price = hist['Close'][-1]
    gain_pct = ((current_price - open_price) / open_price) * 100
    rsi_series = calculate_rsi(hist['Close'])
    rsi = rsi_series.iloc[-1] if not rsi_series.empty else None
            
# Fundamentals
    market_cap = info.get("marketCap", 0)
    cash = info.get("totalCash", None)
    operating_expenses = info.get("totalOperatingExpenses", None)
    shares_outstanding = info.get("sharesOutstanding", None)
    float_shares = info.get("floatShares", None)

# --- Custom Risk Metrics ---
    offering_ability = "High" if float_shares and float_shares > 0.5 * shares_outstanding else "Low"
    dilution_risk = "High" if cash and cash < 10_000_000 and float_shares and float_shares > 0.7 * shares_outstanding else "Moderate"
    cash_need = "Urgent" if months_cash_left and months_cash_left < 3 else "Moderate"

# Calculate Months of Cash Left
    months_cash_left = None
    if cash and operating_expenses and operating_expenses > 0:
        months_cash_left = round(cash / (operating_expenses / 12), 1)

# Apply filters
    if gain_pct >= 30 and rsi and rsi > 70 and market_cap and market_cap < 5e7:
        results.append({
            "Offering Ability": offering_ability,
            "Dilution Risk": dilution_risk,
            "Cash Need": cash_need,
            "Ticker": ticker,
            "Price": round(current_price, 2),
            "Gain %": round(gain_pct, 2),
            "RSI": round(rsi, 2),
            "Market Cap": market_cap,
            "Cash ($)": cash,
            "Months Cash Left": months_cash_left,
            "Float": float_shares,
            "Shares Outstanding": shares_outstanding,
        })
    
              
    # Display results
    if results:
        df = pd.DataFrame(results)
       
        # Format numbers
        df["Cash ($)"] = df["Cash ($)"].apply(human_format)
        df["Market Cap"] = df["Market Cap"].apply(human_format)
        df["Float"] = df["Float"].apply(human_format)
        df["Shares Outstanding"] = df["Shares Outstanding"].apply(human_format)
        df["Gain %"] = df["Gain %"].apply(lambda x: f"{x:.2f}%")
        df["RSI"] = df["RSI"].apply(lambda x: f"{x:.1f}" if x is not None else "-")

        styled_df = (
            df.style
            .applymap(highlight_offering, subset=["Offering Ability"])
            .applymap(highlight_dilution, subset=["Dilution Risk"])
            .applymap(highlight_cash_need, subset=["Cash Need"])
        )
         
        st.success(f"Found {len(df)} matching stocks!")
        st.dataframe(styled_df, use_container_width=True)
      
    # Optional: Add CSV export
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "screener_results.csv", "text/csv")
    else:
        st.warning("No matching stocks found.")
