# main_app.py

# cd /Users/sammusch/Desktop/Uni-Streamlit-2
# python3 -m streamlit run main_app.py

import streamlit as st
import pandas as pd
from models.sarima_model import run_sarima
from models.arima_model import run_arima
from models.arma_model import run_arma
from models.voting_model import run_voting_ensemble
from utils.plot_forecast import plot_forecast
import matplotlib.pyplot as plt



st.set_page_config(page_title="Univariate Forecast App", layout="wide")
st.title("\U0001F4C8 Univariate Forecast App")

# Custom font style
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: "Segoe UI", "Aptos", sans-serif;
}
</style>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = pd.Series(y_true), pd.Series(y_pred)
    return (abs((y_true - y_pred) / y_true.replace(0, 1))).mean() * 100

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.write(df.head())

    if 'date' not in df.columns or len(df.columns) < 2:
        st.error("Please ensure your file has a 'date' column and at least one value column.")
    else:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        value_col = df.columns[0]
        series = df[value_col]

        # Optional Smoothing
        st.sidebar.subheader("Optional Smoothing - BEFORE model")
        smoothing_method = st.sidebar.selectbox("Smoothing Method", ["None", "Moving Average"])
        if smoothing_method != "None":
            smoothing_window = st.sidebar.slider("Smoothing Window (k)", 2, 30, 6)
            if smoothing_method == "Moving Average":
                series = series.rolling(window=smoothing_window).mean().dropna()

        # Forecast Settings
        st.sidebar.subheader("Forecast Settings")
        pred_steps = st.sidebar.number_input("Steps to Forecast", min_value=1, max_value=52, value=6, step=1)
        model_choice = st.sidebar.selectbox("Choose Forecasting Model", ["ARMA", "ARIMA", "SARIMA"]) # , "Voting Ensemble"

        m = None
        if model_choice in ["SARIMA", "Voting Ensemble"]:
            m = st.sidebar.selectbox("Seasonal Period (m)", [1, 4, 6, 12], index=3)

        # Run model only when Go is clicked
        if st.button("\U0001F680 Go!", type="primary"):
            with st.spinner("Running model..."):
                train = series.iloc[:-pred_steps]
                test = series.iloc[-pred_steps:]

                if model_choice == "SARIMA":
                    forecast, error, param_text = run_sarima(train, test, m=m)
                elif model_choice == "ARIMA":
                    forecast, error, param_text = run_arima(train, test)
                elif model_choice == "ARMA":
                    forecast, error, param_text = run_arma(train, test)
                elif model_choice == "Voting Ensemble":
                    forecast, error, param_text = run_voting_ensemble(train, test, m=m)

                if forecast is None:
                    st.error("\u26A0\ufe0f Model failed to produce a forecast. Try a different model or seasonal setting.")
                    st.stop()

                st.session_state["model_forecast"] = forecast
                st.session_state["model_test"] = test
                st.session_state["model_train"] = train
                st.session_state["model_error"] = error
                st.session_state["model_params"] = param_text

# Show results if model has been run
if "model_forecast" in st.session_state:
    forecast = st.session_state["model_forecast"]
    train = st.session_state["model_train"]
    test = st.session_state["model_test"]
    error = st.session_state["model_error"]
    param_text = st.session_state["model_params"]

    full_series = pd.concat([train, test])

    # Date Range Selector
    st.subheader("Select Date Range to Plot")
    min_date = full_series.index.min()
    max_date = full_series.index.max()
    date_range = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        full_series = full_series[(full_series.index >= start_date) & (full_series.index <= end_date)]
        forecast = forecast[(forecast.index >= start_date) & (forecast.index <= end_date)]

    # Plot customization
    st.sidebar.subheader("Plot Customization")
    plot_title = st.sidebar.text_input("Plot Title", value=f"{model_choice} Forecast vs. Actuals")
    xlabel = st.sidebar.text_input("X-axis Label", value="Date")
    ylabel = st.sidebar.text_input("Y-axis Label", value="Value")
    line_width = st.sidebar.slider("Line Width", 1, 5, value=2)
    series_color = st.sidebar.color_picker("Observed Line Color", "#1f77b4")
    forecast_color = st.sidebar.color_picker("Forecast Line Color", "#d62728")
    show_grid = st.sidebar.checkbox("Show Grid", value=False)

    # Plot
    fig = plot_forecast(
        full_series=full_series,
        forecast=forecast,
        title=plot_title,
        xlabel=xlabel,
        ylabel=ylabel,
        line_width=line_width,
        series_color=series_color,
        forecast_color=forecast_color,
        show_grid=show_grid
    )
    #st.plotly_chart(fig, use_container_width=True)
    st.pyplot(fig)

    # Metrics
    col1, col2 = st.columns(2)
    col1.metric(label="Mean Squared Error (MSE)", value=f"{error:.2f}")
    mape = mean_absolute_percentage_error(test, forecast)
    col2.metric(label="Mean Absolute % Error (MAPE)", value=f"{mape:.2f}%")

    if param_text:
        st.markdown("### Model Parameters")
        st.markdown(param_text)
