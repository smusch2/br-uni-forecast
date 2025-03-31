#utils/plot_forecast.py

import matplotlib.pyplot as plt

def plot_forecast(train, test, forecast):
    fig, ax = plt.subplots(figsize=(10, 4))
    train.plot(ax=ax, label='Train')
    test.plot(ax=ax, label='Test')
    forecast.plot(ax=ax, label='Forecast', style='--')
    ax.legend()
    return fig
