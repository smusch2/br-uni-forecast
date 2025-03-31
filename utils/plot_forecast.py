#utils/plot_forecast.py

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#import streamlit as st

def plot_forecast(
    full_series, forecast,
    title="Forecast vs. Actuals",
    xlabel="",
    ylabel="Value",
    line_width=2,
    series_color="#1f77b4",
    forecast_color="#d62728",
    show_grid=False
):
    fig, ax = plt.subplots(figsize=(12, 5))

    full_series.plot(ax=ax, label='Actual Data', color=series_color, linewidth=line_width)
    forecast.plot(ax=ax, label='Forecast', color=forecast_color, linewidth=line_width) # linestyle='--',

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(show_grid)

    return fig


"""
import plotly.graph_objects as go

def plot_forecast(
    full_series, forecast,
    title="Forecast vs. Actuals",
    xlabel="",
    ylabel="Value",
    line_width=2,
    series_color="#1f77b4",
    forecast_color="#d62728",
    show_grid=False
):
    fig = go.Figure()


    # Observed data
    fig.add_trace(go.Scatter(
        x=full_series.index, y=full_series.values,
        mode="lines", name="Observed",
        line=dict(color=series_color, width=line_width)
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values,
        mode="lines", name="Forecast",
        line=dict(color=forecast_color, width=line_width, dash="dash")
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        showlegend=True,
        template="plotly_white",
        xaxis=dict(showgrid=show_grid),
        yaxis=dict(showgrid=show_grid),
        height=500,
        paper_bgcolor="white",       # ensures full background is white
        plot_bgcolor="white",        # ensures plot area is white
        font=dict(
            family="Segoe UI, Aptos, sans-serif",
            size=14,
            color="black"            # 👈 force text color to black
        ),
        title_font=dict(
            size=18,
            color="black"
        ),
        legend=dict(
            font=dict(color="black")
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    return fig
"""