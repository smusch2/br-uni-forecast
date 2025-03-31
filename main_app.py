# main_app.py

import streamlit as st
import pandas as pd
from models.sarima_model import run_sarima
from utils.plot_utils import plot_forecast

st.set_page_config(page_title="Univariate Forecast App", layout="wide")
st.title("📈 Univariate Forecast App")

st.markdown("""
Upload a time series and generate forecasts using SARIMA.
""")

uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])
if uploaded_file:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(df.head())

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    series = df.iloc[:, 0]

    steps = st.sidebar.number_input("Steps to Forecast", min_value=1, max_value=52, value=6)
    train, test = series[:-steps], series[-steps:]

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    forecast, error = run_sarima(train, test, order, seasonal_order, steps)
    st.metric("Mean Squared Error", f"{error:.2f}")

    fig = plot_forecast(train, test, forecast)
    st.pyplot(fig)

