#utils/plot_forecast.py

import matplotlib.pyplot as plt
def plot_forecast(actual, forecast, conf_int=None):
    plt.figure(figsize=(10, 5))
    actual.plot(label='Actual')
    forecast.plot(label='Forecast')
    if conf_int is not None:
        plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    plt.legend()
    return plt
