#models/sarima_model.py

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

def run_sarima(train, test, order, seasonal_order, steps):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=steps)
    error = mean_squared_error(test, forecast)
    return forecast, error
