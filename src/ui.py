"""Cached loaders and chart styling shared by every page."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import model as ml
from src.processing import load_raw, preprocess

# Outcomes: categorical slots 1–3 (validated all-pairs, CVD-safe)
OUTCOME_COLORS = {"Graduate": "#2a78d6", "Dropout": "#eb6834", "Enrolled": "#1baf7a"}
# Risk levels are a state → reserved status palette, always shown with labels
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
DIVERGING = [[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#d03b3b"]]  # positive = red
OUTCOME_ORDER = ["Graduate", "Enrolled", "Dropout"]
RISK_ORDER = ["Low", "Medium", "High"]


@st.cache_data(show_spinner="Loading and preprocessing data…")
def get_data() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    raw = load_raw()
    df, log = preprocess(raw)
    return raw, df, log


@st.cache_resource(show_spinner="Training models (one-time, ~10 s)…")
def get_models() -> ml.TrainResult:
    _, df, _ = get_data()
    return ml.train_all(df)


def style(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        margin=dict(l=10, r=10, t=72 if fig.layout.title.text else 30, b=10),
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        title=dict(font_size=15, yref="container", y=0.985, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        showlegend=legend,
        bargap=0.25,
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    fig.update_traces(selector=dict(type="bar"), marker_line=dict(width=1.5, color="white"))
    return fig


def pct(x: float) -> str:
    return f"{x:.1%}"


def sidebar_footer() -> None:
    with st.sidebar:
        st.divider()
        st.caption(
            "**Domain:** Education  \n"
            "**Dataset:** UCI *Predict Students' Dropout and Academic Success* (4,424 students)  \n"
            "**Tools:** Python · KNIME · Tableau · Streamlit"
        )
