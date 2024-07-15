import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Helper function to format large numbers
def human_format(num):
    if num is None:
        return "N/A"
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return f'{num:.1f}{" KMBT"[magnitude]}'

# Function to train a linear regression model
def train_regression_model(hist):
    # Extracting features (days since start)
    days_since_start = np.array(range(len(hist))).reshape(-1, 1)
    prices = hist['Close'].values.reshape(-1, 1)

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(days_since_start, prices, test_size=0.2, random_state=42)

    # Training the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicting on test set
    y_pred = model.predict(X_test)

    # Evaluating model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

# Function to calculate Discounted Cash Flow (DCF) valuation
def calculate_dcf(stock_data, growth_rate, discount_rate):
    # Fetch relevant financial metrics
    eps = stock_data.info.get('trailingEps', 0)

    # Estimate future cash flows (here, using EPS growth)
    future_cash_flows = []
    for i in range(1, 6):
        future_cash_flows.append(eps * (1 + growth_rate)**i)

    # Calculate terminal value using Gordon Growth Model (also known as perpetuity growth method)
    terminal_value = (eps * (1 + growth_rate)**6) / (discount_rate - growth_rate)

    # Discount all cash flows to present value
    dcf_value = 0
    for i, cash_flow in enumerate(future_cash_flows):
        dcf_value += cash_flow / (1 + discount_rate)**(i + 1)
    dcf_value += terminal_value / (1 + discount_rate)**5

    return dcf_value

# Set up the Streamlit interface
st.title("Stock Chart Viewer with DCF Valuation")
st.write("Enter a ticker symbol and select parameters to view the stock chart and valuation using a Discounted Cash Flow (DCF) model.")

# Input for the ticker symbol
ticker = st.text_input("Ticker symbol", "AAPL")

# Date range input for the time frame
date_range = st.date_input(
    "Select date range",
    value=[date(2020, 1, 1), date.today()],
    min_value=date(2000, 1, 1),
    max_value=date.today()
)

# Growth rate and discount rate inputs
growth_rate = st.number_input("Growth Rate (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
discount_rate = st.number_input("Discount Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

# Ensure the date range contains exactly two dates
if len(date_range) == 2:
    start_date, end_date = date_range

    # Fetch data from Yahoo Finance
    if ticker:
        stock_data = yf.Ticker(ticker)
        hist = stock_data.history(start=start_date, end=end_date)

        # Sidebar for stock info and additional metrics
        st.sidebar.title("Stock Information and Metrics")
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

        # Calculate DCF valuation and current stock price
        if not hist.empty:
            dcf_value = calculate_dcf(stock_data, growth_rate / 100, discount_rate / 100)
            st.sidebar.write(f"**DCF Valuation:** ${dcf_value:.2f}")

            current_price = hist['Close'][-1]
            dcf_percent_diff = ((dcf_value - current_price) / current_price) * 100

            if dcf_percent_diff >= 0:
                st.sidebar.markdown(f"**Undervalued by:** <span style='color: green;'>{abs(dcf_percent_diff):.2f}%</span>", unsafe_allow_html=True)
            else:
                st.sidebar.markdown(f"**Overvalued by:** <span style='color: red;'>{abs(dcf_percent_diff):.2f}%</span>", unsafe_allow_html=True)

        # Train regression model and predict future prices
        model, mse, r2 = train_regression_model(hist)

        # Display regression model performance
        st.sidebar.subheader("Regression Model Performance")
        st.sidebar.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.sidebar.write(f"**R^2 Score:** {r2:.2f}")

        # Calculate and display stock metrics
        if not hist.empty:
            start_price = hist['Close'][0]
            end_price = hist['Close'][-1]
            stock_return = ((end_price - start_price) / start_price) * 100
            pe_ratio = stock_data.info.get('forwardPE', 'N/A')
            next_dividend_date = stock_data.info.get('dividendDate', 'N/A')

            st.sidebar.subheader("Stock Metrics")
            st.sidebar.write(f"**Stock Return:** {stock_return:.2f}%")
            st.sidebar.write(f"**P/E Ratio:** {pe_ratio}")
            st.sidebar.write(f"**Next Dividend Date:** {next_dividend_date}")

        # Create the Plotly figure
        fig = go.Figure()

        # Add the historical data
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close Price'))

        # Predict future prices for the next 30 days
        future_dates = [end_date + timedelta(days=i) for i in range(1, 31)]
        future_days = np.array(range(len(hist), len(hist) + 30)).reshape(-1, 1)
        future_predictions = model.predict(future_days)

        # Add future predictions to the plot
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Predicted Price', line=dict(dash='dash')))

        # Update layout
        fig.update_layout(
            title=f'{ticker} Stock Price with Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Close Price (USD)',
            template='plotly_dark'  # You can change the template to your preference
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Debt vs. Equity Mosaic Chart
        debt = stock_data.info.get('totalDebt', 0)
        equity = stock_data.info.get('totalStockholderEquity', 0)
        if debt and equity:
            fig_mosaic = go.Figure(
                data=[go.Pie(labels=['Debt', 'Equity'], values=[debt, equity], hole=0.4)]
            )
            fig_mosaic.update_layout(
                title_text='Debt vs. Equity',
                annotations=[dict(text='Debt vs. Equity', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            st.plotly_chart(fig_mosaic)

    else:
        st.write("No data available for the selected time frame or ticker symbol.")

else:
    st.write("Please select a valid date range.")