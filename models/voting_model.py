from models.arima_model import run_arima
from models.sarima_model import run_sarima
from models.arma_model import run_arma
import pandas as pd

def run_voting_ensemble(train, test, m):
    forecasts = []
    model_errors = []

    # ARIMA
    forecast_arima, error_arima, _ = run_arima(train, test)
    if forecast_arima is not None:
        forecasts.append(forecast_arima)
        model_errors.append(("ARIMA", error_arima))

    # SARIMA
    forecast_sarima, error_sarima, _ = run_sarima(train, test, m=m)
    if forecast_sarima is not None:
        forecasts.append(forecast_sarima)
        model_errors.append(("SARIMA", error_sarima))

    # ARMA
    forecast_arma, error_arma, _ = run_arma(train, test)
    if forecast_arma is not None:
        forecasts.append(forecast_arma)
        model_errors.append(("ARMA", error_arma))

    if not forecasts:
        return None, None, "Voting Ensemble failed: no base models were successfully fit."

    # Average forecasts
    forecasts_df = pd.concat(forecasts, axis=1)
    ensemble_forecast = forecasts_df.mean(axis=1)
    error_ensemble = ((ensemble_forecast - test) ** 2).mean()

    param_text = f"**Voting Ensemble Forecast**\n\n"
    param_text += f"- Models included: {', '.join(name for name, _ in model_errors)}\n"
    param_text += f"- Ensemble MSE: {error_ensemble:.2f}\n"
    param_text += "\n**Individual Model Errors:**\n"
    for name, err in model_errors:
        param_text += f"- {name} MSE: {err:.2f}\n"

    return ensemble_forecast, error_ensemble, param_text

