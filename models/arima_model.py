import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import itertools
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def run_arima(train, test):
    # Define ranges for p, d, q
    p = d = q = range(0, 4)
    pdq = list(itertools.product(p, d, q))

    best_score = float("inf")
    best_order = None
    best_forecast = None

    for order in pdq:
        try:
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            error = mean_squared_error(test, forecast)
            if error < best_score:
                best_score = error
                best_order = order
                best_forecast = forecast
        except Exception:
            continue

    param_text = f"""
    **Selected ARIMA Order (p,d,q)**: {best_order}  

    **Descriptions**  
    - **p**: Number of lag observations (AR)  
    - **d**: Number of times data was differenced (trend removal)  
    - **q**: Size of the moving average window (MA)
    """

    return best_forecast, best_score, param_text
