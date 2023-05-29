import plotly.express as px
import plotly.graph_objects as go

import numpy as np


def plot_data(X_train, y_train, X_test, y_test, title):
    """Plot the data.
    Parameters
    ----------
    X_train : numpy.ndarray
        Training input data.
    y_train : numpy.ndarray
        Training target data.
    X_test : numpy.ndarray
        Test input data.
    y_test : numpy.ndarray
        Test target data.
    title : str
        Title of the plot.
    Returns
    -------
    plotly.graph_objects.Figure
        A plotly figure.
    """
    fig = go.Figure(
        [
            go.Scatter(
                name="Train",
                x=X_train,
                y=y_train,
                mode="markers",
                line=dict(color="red"),
            ),
            go.Scatter(
                name="Test",
                x=X_test,
                y=y_test,
                mode="markers",
                line=dict(color="blue"),
            ),
        ]
    )

    fig.update_layout(yaxis_title="y", xaxis_title="X", title=title, hovermode="x")
    fig.update_layout(autosize=False, width=1000, height=1000)

    return fig


def plot_regression(X_train, y_train, X_test, y_test, title, line_title, scatter_title):
    indices = np.argsort(X_test)
    X_test = X_test[indices]
    y_test = y_test[indices]
    fig = go.Figure(
        [
            go.Scatter(name=scatter_title, x=X_train, y=y_train, mode="markers"),
            go.Scatter(
                name=line_title,
                x=X_test,
                y=y_test,
                mode="lines",
                line=dict(color="rgb(31, 119, 180)"),
            ),
        ]
    )

    fig.update_layout(yaxis_title="y", xaxis_title="x", title=title, hovermode="x")
    fig.update_layout(autosize=False, width=1000, height=1000)

    return fig


def plot_bayesian_regression(
    X_train, y_train, X_test, y_test, y_std, title="regression"
):
    indices = np.argsort(X_test)
    X_test = X_test[indices]
    y_test = y_test[indices]
    y_std = y_std[indices]
    fig = go.Figure(
        [
            go.Scatter(name="Training data", x=X_train, y=y_train, mode="markers"),
            go.Scatter(
                name="MAP",
                x=X_test,
                y=y_test,
                mode="lines",
                line=dict(color="rgb(31, 119, 180)"),
            ),
            go.Scatter(
                name="Upper Bound",
                x=X_test,
                y=y_test + y_std,
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                name="Lower Bound",
                x=X_test,
                y=y_test - y_std,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
                showlegend=False,
            ),
        ]
    )
    fig.update_layout(yaxis_title="y", xaxis_title="x", title=title, hovermode="x")
    fig.update_layout(autosize=False, width=1000, height=1000)

    return fig
