import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import itertools
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def run_sarima(train, test, m=52):
    # Define ranges
    p = d = q = range(0, 4)
    P = D = Q = range(0, 4)

    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = list(itertools.product(P, D, Q))

    best_score = float("inf")
    best_order = None
    best_seasonal_order = None
    best_forecast = None

    for order in pdq:
        for seasonal_order in seasonal_pdq:
            try:
                model = SARIMAX(train, order=order, seasonal_order=(seasonal_order[0], seasonal_order[1], seasonal_order[2], m))
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(steps=len(test))
                error = mean_squared_error(test, forecast)
                if error < best_score:
                    best_score = error
                    best_order = order
                    best_seasonal_order = seasonal_order
                    best_forecast = forecast
            except Exception:
                continue

    param_text = f"""
    **Selected SARIMA Order**  
    - Non-seasonal (p,d,q): {best_order}  
    - Seasonal (P,D,Q,m): ({best_seasonal_order[0]}, {best_seasonal_order[1]}, {best_seasonal_order[2]}, {m})

    **Descriptions**  
    - **p/P**: Autoregressive terms  
    - **d/D**: Differencing terms  
    - **q/Q**: Moving average terms  
    - **m**: Seasonal period (e.g., 12 for monthly, 4 for quarterly)
    """

    if best_forecast is None:
        return None, None, "SARIMA failed to fit any models. Try a different seasonal period (m) or use another model."

    param_text = f"""
    **Selected SARIMA Order**  
    - Non-seasonal (p,d,q): {best_order}  
    - Seasonal (P,D,Q,m): ({best_seasonal_order[0]}, {best_seasonal_order[1]}, {best_seasonal_order[2]}, {m})

    **Descriptions**  
    - **p/P**: Autoregressive terms  
    - **d/D**: Differencing terms  
    - **q/Q**: Moving average terms  
    - **m**: Seasonal period (e.g., 12 for monthly, 4 for quarterly)
    """

    return best_forecast, best_score, param_text


    return best_forecast, best_score, param_text
