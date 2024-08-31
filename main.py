# import libraries
import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from datetime import date, datetime, timedelta
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Helper functions
def human_format(num):
    if num is None:
        return "N/A"
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return f'{num:.1f}{" KMBT"[magnitude]}'

# Check Stationarity
def check_stationarity(data):
    result = adfuller(data)
    return result[1] < 0.05  # p-value < 0.05 means data is stationary

# Function to difference the data to make it stationary
def difference_data(data):
    return np.diff(data)

# Function to fit the ARIMA model with manually set parameters
def fit_arima_model(hist, periods):
    prices = hist['Close'].values

    # Ensure the data is stationary
    if not check_stationarity(prices):
        prices = difference_data(prices)

    try:
        # Manually setting ARIMA parameters (e.g., p=5, d=1, q=0)
        model = ARIMA(prices, order=(5, 1, 0))  # Adjust the order as needed
        model_fit = model.fit()

        # Predict future periods
        forecast = model_fit.forecast(steps=periods)

        # Adjust forecast to account for differencing (if applicable)
        if len(hist) > 1:
            forecast = np.cumsum(forecast) + hist['Close'].iloc[-1]

        return forecast, model_fit
    except Exception as e:
        st.sidebar.error(f"ARIMA model failed to predict: {e}")
        return np.full(periods, hist['Close'].iloc[-1]), None  # Fallback to a flat forecast if ARIMA fails

# Discounted Cash Flow (DCF)
def calculate_dcf(stock_data, growth_rate, discount_rate):
    # Begrenzung der Wachstumsrate auf ein realistisches Intervall
    growth_rate = max(min(growth_rate, 0.15), 0.02)  # z.B. 2% bis 15%

    eps = stock_data.info.get('trailingEps', 0)
    future_cash_flows = []
    for i in range(1, 6):
        future_cash_flows.append(eps * (1 + growth_rate) ** i)
    
    # Terminal Growth Rate festlegen, z.B. 2.5%
    terminal_growth_rate = 0.025
    terminal_value = (eps * (1 + terminal_growth_rate) ** 6) / (discount_rate - terminal_growth_rate)
    
    dcf_value = 0
    for i, cash_flow in enumerate(future_cash_flows):
        dcf_value += cash_flow / (1 + discount_rate) ** (i + 1)
    dcf_value += terminal_value / (1 + discount_rate) ** 5
    return dcf_value

# Streamlit interface
st.title("Stock-Price Prediction and Valuation")
st.write("Enter your ticker symbol and select the growth rate for DCF Valuation")

# Input for the ticker symbol (Default set to NVDA)
ticker = st.text_input("Ticker symbol", "NVDA")

# Select growth rate
growth_rate = st.slider("Growth Rate (%)", min_value=-20.0, max_value=20.0, value=5.0) / 100

# Fixed discount rate
discount_rate = 0.10

# Use YTD data
start_date = datetime(datetime.now().year, 1, 1)
end_date = datetime.combine(date.today(), datetime.min.time())

# Shorter prediction horizon = 30 days
prediction_horizon = 30

# Fetch data from Yahoo Finance
if ticker:
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(start=start_date, end=end_date)

    # Sidebar for stock info and additional metrics
    st.sidebar.title("Stock-Metrics")
    st.sidebar.write(f"**Ticker:** {ticker}")
    st.sidebar.write(f"**Name:** {stock_data.info.get('longName', 'N/A')}")
    st.sidebar.write(f"**Sector:** {stock_data.info.get('sector', 'N/A')}")
    st.sidebar.write(f"**Industry:** {stock_data.info.get('industry', 'N/A')}")
    st.sidebar.write(f"**Market Cap:** {human_format(stock_data.info.get('marketCap'))}")
    st.sidebar.write(f"**Dividend Yield:** {stock_data.info.get('dividendYield', 0) * 100:.2f}%")
    st.sidebar.write(f"**Previous Close:** ${stock_data.info.get('previousClose', 'N/A')}")
    st.sidebar.write(f"**Open:** ${stock_data.info.get('open', 'N/A')}")
    st.sidebar.write(f"**52 Week High:** ${stock_data.info.get('fiftyTwoWeekHigh', 'N/A')}")
    st.sidebar.write(f"**52 Week Low:** ${stock_data.info.get('fiftyTwoWeekLow', 'N/A')}")

    st.sidebar.write(f"**Selected Growth Rate:** {growth_rate * 100:.2f}%")
    st.sidebar.write(f"**Fixed Discount Rate:** {discount_rate * 100:.2f}%")

    # Calculate DCF valuation and current stock price
    if not hist.empty:
        dcf_value = calculate_dcf(stock_data, growth_rate, discount_rate)
        st.sidebar.write(f"**DCF Valuation:** ${dcf_value:.2f}")

        current_price = hist['Close'][-1]
        dcf_percent_diff = ((dcf_value - current_price) / current_price) * 100

        if dcf_percent_diff >= 0:
            st.sidebar.markdown(f"**Undervalued by:** <span style='color: green;'>{abs(dcf_percent_diff):.2f}%</span>",
                                unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"**Overvalued by:** <span style='color: red;'>{abs(dcf_percent_diff):.2f}%</span>",
                                unsafe_allow_html=True)

    # Fit ARIMA model and predict next 30 days
    forecast, arima_model = fit_arima_model(hist, prediction_horizon)

    if arima_model:
        # Evaluation metrics
        arima_aic = arima_model.aic
        arima_bic = arima_model.bic
        final_forecast_price = forecast[-1]

        st.sidebar.subheader("ARIMA Model Performance")
        st.sidebar.write(f"**Predicted Price after 30 Days:** ${final_forecast_price:.2f}")
        st.sidebar.write(f"**AIC:** {arima_aic:.2f}")
        st.sidebar.write(f"**BIC:** {arima_bic:.2f}")

        # Calculate the percentage difference between ARIMA predicted price and the current price
        arima_percent_diff = ((final_forecast_price - current_price) / current_price) * 100

        if arima_percent_diff >= 0:
            st.sidebar.markdown(
                f"**Price Increase in 30 Days:** <span style='color: green;'>{abs(arima_percent_diff):.2f}%</span>",
                unsafe_allow_html=True)
        else:
            st.sidebar.markdown(
                f"**Price Decrease in 30 Days:** <span style='color: red;'>{abs(arima_percent_diff):.2f}%</span>",
                unsafe_allow_html=True)

    # Create figure
    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close Price'))

    # Add ARIMA prediction in graph
    future_dates = [end_date + timedelta(days=i) for i in range(1, prediction_horizon + 1)]
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Predicted Price', line=dict(dash='dash')))

    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Price (YTD) with 30-Day Prediction',
        xaxis_title='Date',
        yaxis_title='Close Price (USD)',
        template='plotly_dark'
    )

    # Display in Streamlit
    st.plotly_chart(fig)

