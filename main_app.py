# main_app.py

import streamlit as st
import pandas as pd

from models.sarima_model import run_sarima
from models.prophet_model import run_prophet
from utils.plot_forecast import plot_forecast

# Upload data
uploaded_file = st.file_uploader("Upload your time series CSV")
if uploaded_file:
    data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    st.line_chart(data)

    # Model selection
    model_choice = st.selectbox("Choose a model", ["SARIMA", "Prophet"])

    if model_choice == "SARIMA":
        forecast, conf_int = run_sarima(data)
        plot = plot_forecast(data, forecast, conf_int)
    elif model_choice == "Prophet":
        forecast_df = run_prophet(data)
        forecast = forecast_df.set_index("ds")["yhat"]
        conf_int = forecast_df[["yhat_lower", "yhat_upper"]]
        plot = plot_forecast(data, forecast, conf_int)

    st.pyplot(plot)
