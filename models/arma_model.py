import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import itertools
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def run_arma(train, test):
    # ARMA is ARIMA with d=0
    p = q = range(0, 3)
    pdq = list(itertools.product(p, q))
    
    best_score = float("inf")
    best_order = None
    best_forecast = None

    for order in pdq:
        try:
            model = ARIMA(train, order=(order[0], 0, order[1]))
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
    **Selected ARMA Order (p,q)**: {best_order} (with d=0)  

    **Descriptions**  
    - **p**: Number of autoregressive terms  
    - **q**: Number of moving average terms  
    - **Note**: ARMA is suitable for stationary time series (differencing not applied).
    """

    return best_forecast, best_score, param_text
