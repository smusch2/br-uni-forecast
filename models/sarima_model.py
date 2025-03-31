#models/sarima_model.py
def run_sarima(data, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    results = model.fit()
    forecast = results.get_forecast(steps=12)
    return forecast.predicted_mean, forecast.conf_int()
