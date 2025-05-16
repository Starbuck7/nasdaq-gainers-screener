import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="📈 NASDAQ Stock Screener", layout="wide")
st.title("📈 NASDAQ Stock Screener")

# --- Load NASDAQ tickers from public CSV ---
@st.cache_data(ttl=24*3600)
def load_nasdaq_tickers():
    url = "https://datahub.io/core/nasdaq-listings/r/nasdaq-listed-symbols.csv"
    df = pd.read_csv(url)
    # Filter out test tickers or invalid entries if needed
    tickers = df['Symbol'].tolist()
    return tickers

# --- RSI Calculation ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- UI Controls ---
st.sidebar.header("📊 Filter Criteria")
st.sidebar.write("Only analyzing today's top NASDAQ gainers.")
run_scan = st.sidebar.button("🔍 Run Screener")

if run_scan:
    tickers = load_nasdaq_tickers()
    st.write(f"Total NASDAQ tickers loaded: {len(tickers)}")

    results = []

    for i, ticker in enumerate(tickers):
        # Show progress
        if i % 50 == 0:
            st.write(f"Processing {i} / {len(tickers)} tickers...")

        try:
            stock = yf.Ticker(ticker)

            # Get last 2 days daily data for gain calculation
            hist = stock.history(period="3d", interval="1d")
            if hist.shape[0] < 2:
                continue
            open_price = hist['Open'].iloc[-2]
            close_price = hist['Close'].iloc[-1]
            gain_pct = ((close_price - open_price) / open_price) * 100

            if gain_pct < 30:
                continue

            # RSI calculation on last 15 days
            hist_15d = stock.history(period="15d", interval="1d")
            if hist_15d.empty or 'Close' not in hist_15d:
                continue

            rsi_series = calculate_rsi(hist_15d['Close'])
            rsi = rsi_series.iloc[-1] if not rsi_series.empty else 0

            if rsi <= 70:
                continue

            info = stock.info
            market_cap = info.get('marketCap', 0)
            if market_cap is None or market_cap >= 50_000_000:
                continue

            # Optional additional info
            cash = info.get('totalCash', None)
            operating_expenses = info.get('totalOperatingExpenses', None)
            shares_outstanding = info.get('sharesOutstanding', None)
            float_shares = info.get('floatShares', None)

            months_cash_left = None
            if cash and operating_expenses and operating_expenses > 0:
                months_cash_left = round(cash / (operating_expenses / 12), 1)

            offering_ability = "High" if float_shares and shares_outstanding and float_shares > 0.5 * shares_outstanding else "Low"
            dilution_risk = "High" if cash and cash < 10_000_000 and float_shares and shares_outstanding and float_shares > 0.7 * shares_outstanding else "Moderate"
            cash_need = "Urgent" if months_cash_left and months_cash_left < 3 else "Moderate"

            results.append({
                "Ticker": ticker,
                "Price": round(close_price, 2),
                "Gain %": round(gain_pct, 2),
                "RSI": round(rsi, 2),
                "Market Cap": market_cap,
                "Cash ($)": cash,
                "Months Cash Left": months_cash_left,
                "Float Shares": float_shares,
                "Shares Outstanding": shares_outstanding,
                "Offering Ability": offering_ability,
                "Dilution Risk": dilution_risk,
                "Cash Need": cash_need,
            })

        except Exception as e:
            st.warning(f"Error processing {ticker}: {e}")
            continue

    if results:
        df = pd.DataFrame(results)
        st.success(f"✅ Found {len(df)} matching stock(s).")

        def highlight_offering(val): return "background-color: orange" if val == "High" else ""
        def highlight_dilution(val): return "background-color: red" if val == "High" else ""
        def highlight_cash_need(val): return "background-color: red" if val == "Urgent" else ""

        styled_df = (
            df.style
            .applymap(highlight_offering, subset=["Offering Ability"])
            .applymap(highlight_dilution, subset=["Dilution Risk"])
            .applymap(highlight_cash_need, subset=["Cash Need"])
        )

        st.dataframe(styled_df, use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "screener_results.csv", "text/csv")

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
                            close=chart_data['Close']
                        )
                    ])
                    fig.update_layout(
                        title=f"{ticker} - 5 Day Candlestick Chart",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        xaxis_rangeslider_visible=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.warning(f"Could not load chart for {ticker}")
    else:
        st.warning("No matching stocks found.")
