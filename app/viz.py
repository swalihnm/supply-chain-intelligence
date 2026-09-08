"""Chart theme and builders for the dashboard.

PALETTE
-------
Categorical slots are assigned in fixed order and never cycled. The four in use
were validated with the data-viz validator against both surfaces:

  light  worst adjacent CVD dE 9.1, normal-vision dE 22.9  - PASS
  dark   worst adjacent CVD dE 8.4, normal-vision dE 19.8  - PASS

Light mode raises a sub-3:1 contrast warning on aqua and yellow. That is not
dismissable: every chart here ships a table view beside it, which is the
documented relief.

FORM CHOICES
------------
The model comparison is a sorted horizontal BAR, not a grouped bar across folds.
Nine models x six folds as grouped bars would be 54 marks and would exceed the
categorical cap several times over; the question ("which model is best") is a
magnitude comparison across categories, and a sorted bar answers it directly.
Fold-level stability is a separate small line chart restricted to four models.

There are no dual-axis charts anywhere. Cost and service level are shown as one
scatter with cost on y and achieved service on x, which is the relationship
itself rather than two series sharing an axis.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# --- palette ---------------------------------------------------------------
LIGHT = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "s1": "#2a78d6",
    "s2": "#eb6834",
    "s3": "#1baf7a",
    "s4": "#eda100",
    "critical": "#d03b3b",
    "good": "#0ca30c",
}

DARK = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "s1": "#3987e5",
    "s2": "#d95926",
    "s3": "#199e70",
    "s4": "#c98500",
    "critical": "#d03b3b",
    "good": "#0ca30c",
}


def palette(dark: bool = False) -> dict:
    return DARK if dark else LIGHT


def base_layout(p: dict, title: str = "", height: int = 380) -> dict:
    """Recessive chrome: hairline grid, no chart-junk border, muted axis ink."""
    return {
        "title": {"text": title, "font": {"size": 15, "color": p["text"]}, "x": 0.0},
        "height": height,
        "paper_bgcolor": p["surface"],
        "plot_bgcolor": p["surface"],
        "font": {"color": p["text_secondary"], "size": 12},
        "margin": {"l": 60, "r": 24, "t": 44 if title else 16, "b": 48},
        "xaxis": {
            "gridcolor": p["grid"],
            "zeroline": False,
            "linecolor": p["axis"],
            "tickfont": {"color": p["muted"]},
        },
        "yaxis": {
            "gridcolor": p["grid"],
            "zeroline": False,
            "linecolor": p["axis"],
            "tickfont": {"color": p["muted"]},
        },
        "hovermode": "closest",
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"color": p["text_secondary"]},
        },
    }


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------


def service_level_frontier(
    fill_rates: np.ndarray,
    costs: np.ndarray,
    targets: np.ndarray,
    baseline_fill: float | None = None,
    baseline_cost: float | None = None,
    recommended_idx: int | None = None,
    dark: bool = False,
) -> go.Figure:
    """Cost against achieved service level - the chart the project exists for.

    Service on x and cost on y, so the shape of the curve IS the trade-off: the
    steepening slope toward the right is the non-linearity in the "what happens
    at 98% instead of 95%" question.
    """
    p = palette(dark)
    fig = go.Figure()

    order = np.argsort(fill_rates)
    fig.add_trace(
        go.Scatter(
            x=fill_rates[order] * 100,
            y=costs[order],
            mode="lines+markers+text",
            name="Optimized policy",
            line={"color": p["s1"], "width": 2},
            marker={"size": 10, "color": p["s1"], "line": {"width": 2, "color": p["surface"]}},
            text=[f"{t:.0%}" for t in targets[order]],
            textposition="top center",
            textfont={"color": p["text_secondary"], "size": 10},
            hovertemplate="target %{text}<br>achieved %{x:.2f}%<br>cost $%{y:,.0f}<extra></extra>",
        )
    )

    if baseline_fill is not None and baseline_cost is not None:
        fig.add_trace(
            go.Scatter(
                x=[baseline_fill * 100],
                y=[baseline_cost],
                mode="markers+text",
                name="Reconstructed baseline",
                marker={
                    "size": 14,
                    "color": p["s2"],
                    "symbol": "diamond",
                    "line": {"width": 2, "color": p["surface"]},
                },
                text=["current"],
                textposition="middle right",
                textfont={"color": p["s2"], "size": 11},
                hovertemplate="baseline<br>fill %{x:.2f}%<br>cost $%{y:,.0f}<extra></extra>",
            )
        )

    if recommended_idx is not None and recommended_idx < len(fill_rates):
        fig.add_trace(
            go.Scatter(
                x=[fill_rates[recommended_idx] * 100],
                y=[costs[recommended_idx]],
                mode="markers",
                name="Recommended",
                marker={
                    "size": 18,
                    "color": "rgba(0,0,0,0)",
                    "line": {"width": 2.5, "color": p["good"]},
                },
                hovertemplate="recommended operating point<extra></extra>",
            )
        )

    lay = base_layout(p, "", 420)
    lay["xaxis"]["title"] = "Achieved fill rate (%)"
    lay["yaxis"]["title"] = "Total cost ($ over 28 days)"
    fig.update_layout(**lay)
    return fig


def forecast_chart(
    days: np.ndarray,
    actual: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    band_label: str = "50-95% band",
    dark: bool = False,
) -> go.Figure:
    """Actuals against the forecast, with the quantile band behind them."""
    p = palette(dark)
    fig = go.Figure()

    if lower is not None and upper is not None:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([days, days[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor="rgba(235,104,52,0.16)",
                line={"width": 0},
                name=band_label,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=days,
            y=point,
            mode="lines",
            name="Forecast",
            line={"color": p["s2"], "width": 2},
            hovertemplate="day %{x}<br>forecast %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=actual,
            mode="lines+markers",
            name="Actual",
            line={"color": p["s1"], "width": 2},
            marker={"size": 8, "line": {"width": 2, "color": p["surface"]}},
            hovertemplate="day %{x}<br>actual %{y:.0f}<extra></extra>",
        )
    )

    lay = base_layout(p, "", 380)
    lay["xaxis"]["title"] = "Day index"
    lay["yaxis"]["title"] = "Units"
    lay["hovermode"] = "x unified"
    fig.update_layout(**lay)
    return fig


def calibration_chart(nominal: np.ndarray, empirical: np.ndarray, dark: bool = False) -> go.Figure:
    """Do the quantiles mean what they say?

    The diagonal is perfect calibration. Points below it mean the quantile is
    too low and the optimizer will under-stock relative to its service target.
    """
    p = palette(dark)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0.4, 1.0],
            y=[0.4, 1.0],
            mode="lines",
            name="Perfect calibration",
            line={"color": p["muted"], "width": 1.5, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=nominal,
            y=empirical,
            mode="markers+text",
            name="Observed coverage",
            marker={"size": 12, "color": p["s1"], "line": {"width": 2, "color": p["surface"]}},
            text=[f"{e:.3f}" for e in empirical],
            textposition="bottom right",
            textfont={"color": p["text_secondary"], "size": 10},
            hovertemplate="nominal %{x:.3f}<br>empirical %{y:.3f}<extra></extra>",
        )
    )

    lay = base_layout(p, "", 380)
    lay["xaxis"]["title"] = "Nominal quantile"
    lay["yaxis"]["title"] = "Empirical coverage"
    lay["xaxis"]["range"] = [0.4, 1.02]
    lay["yaxis"]["range"] = [0.4, 1.02]
    fig.update_layout(**lay)
    return fig


def model_ranking(models: list[str], values: np.ndarray, dark: bool = False) -> go.Figure:
    """Sorted horizontal bar. One series, so no legend - the title names it."""
    p = palette(dark)
    order = np.argsort(values)[::-1]
    names = [models[i] for i in order]
    vals = values[order]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker={"color": p["s1"], "line": {"width": 2, "color": p["surface"]}},
            text=[f"{v:.4f}" for v in vals],
            textposition="outside",
            textfont={"color": p["text_secondary"], "size": 11},
            hovertemplate="%{y}<br>WRMSSE %{x:.4f}<extra></extra>",
        )
    )
    lay = base_layout(p, "", max(280, 34 * len(models) + 80))
    lay["xaxis"]["title"] = "WRMSSE (lower is better)"
    lay["margin"]["l"] = 150
    lay["margin"]["r"] = 70
    fig.update_layout(**lay)
    return fig


def fold_stability(
    folds: list[str], series_by_model: dict[str, np.ndarray], dark: bool = False
) -> go.Figure:
    """Per-fold WRMSSE for at most four models - the categorical cap."""
    p = palette(dark)
    slots = [p["s1"], p["s2"], p["s3"], p["s4"]]
    fig = go.Figure()
    for i, (name, vals) in enumerate(list(series_by_model.items())[:4]):
        fig.add_trace(
            go.Scatter(
                x=folds,
                y=vals,
                mode="lines+markers",
                name=name,
                line={"color": slots[i], "width": 2},
                marker={"size": 8, "line": {"width": 2, "color": p["surface"]}},
                hovertemplate=f"{name}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
            )
        )
    lay = base_layout(p, "", 340)
    lay["xaxis"]["title"] = "Fold"
    lay["yaxis"]["title"] = "WRMSSE"
    fig.update_layout(**lay)
    return fig


def cost_breakdown(
    labels: list[str],
    holding: np.ndarray,
    stockout: np.ndarray,
    ordering: np.ndarray,
    dark: bool = False,
) -> go.Figure:
    """Stacked cost composition. A 2px surface gap separates the segments."""
    p = palette(dark)
    fig = go.Figure()
    for name, vals, colour in [
        ("Holding", holding, p["s1"]),
        ("Stockout", stockout, p["s2"]),
        ("Ordering", ordering, p["s3"]),
    ]:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=vals,
                name=name,
                marker={"color": colour, "line": {"width": 2, "color": p["surface"]}},
                hovertemplate=f"{name}: $%{{y:,.0f}}<extra></extra>",
            )
        )
    lay = base_layout(p, "", 360)
    lay["yaxis"]["title"] = "Cost ($ over 28 days)"
    fig.update_layout(barmode="stack", **lay)
    return fig
