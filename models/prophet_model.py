# models/prophet_model.py
def run_prophet(data):
    from prophet import Prophet
    import pandas as pd

    df = pd.DataFrame({'ds': data.index, 'y': data.values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
