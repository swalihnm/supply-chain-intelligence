"""Render the README figures as PNG.

Usage:  uv run python scripts/make_figures.py

Produces docs/frontier.png — the service-level-versus-cost chart.

WHY THIS EXISTS
---------------
The dashboard already draws this interactively, but almost nobody reading the
repo will start a container to see it. The spec is blunt about this: the
frontier is the single most useful chart in the project for an interview, and a
README that only describes it has not delivered it.

Colours are the same validated categorical palette the dashboard uses
(app/viz.py) - worst adjacent CVD dE 9.1, normal-vision dE 22.9 on the light
surface. Identity is never carried by colour alone: both series are directly
labelled as well as being in the legend.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from app.viz import LIGHT  # noqa: E402
from sci.config import REPO_ROOT, settings  # noqa: E402
from sci.log import get_logger  # noqa: E402

log = get_logger(__name__)

DOCS = REPO_ROOT / "docs"


def _style(ax) -> None:
    """Recessive chrome: hairline grid, no top/right spines, muted tick ink."""
    ax.set_facecolor(LIGHT["surface"])
    ax.grid(True, color=LIGHT["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LIGHT["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=LIGHT["muted"], labelsize=10)


def frontier() -> None:
    path = settings.artifacts_dir / "service_level_frontier.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found — run scripts/run_optimizer.py first.")

    df = pl.read_csv(path)
    # is_not_null() alone is NOT enough: the baseline row carries NaN, not null,
    # and Polars treats those as different. Rendering the chart without the
    # is_not_nan() drew the baseline as an eighth point ON the frontier line,
    # labelled "nan%", and put the recommended-point ring on it.
    pts = df.filter(
        pl.col("target_service_level").is_not_null() & pl.col("target_service_level").is_not_nan()
    ).sort("target_service_level")
    base = df.filter(pl.col("policy") == "reconstructed_baseline")

    x = pts["fill_rate"].to_numpy() * 100
    y = pts["total_cost"].to_numpy()
    targets = pts["target_service_level"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=170)
    fig.patch.set_facecolor(LIGHT["surface"])
    _style(ax)

    ax.plot(
        x,
        y,
        color=LIGHT["s1"],
        linewidth=2.0,
        zorder=3,
        # "Optimized policy" would be ambiguous - the recommended star is an
        # optimized policy too. This line is specifically what ONE uniform
        # target buys.
        label="Uniform service-level target",
    )
    ax.scatter(
        x,
        y,
        s=78,
        color=LIGHT["s1"],
        edgecolors=LIGHT["surface"],
        linewidths=2.0,
        zorder=4,
    )

    # Direct labels on every point - there are only eight, and the target level
    # is the thing a reader wants to map back to.
    for xi, yi, t in zip(x, y, targets, strict=True):
        ax.annotate(
            f"{t:.1%}".replace(".0%", "%"),
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
            color=LIGHT["text_secondary"],
            zorder=5,
        )

    if base.height:
        bx = float(base["fill_rate"][0]) * 100
        by = float(base["total_cost"][0])
        ax.scatter(
            [bx],
            [by],
            s=190,
            marker="D",
            color=LIGHT["s2"],
            edgecolors=LIGHT["surface"],
            linewidths=2.0,
            zorder=6,
            label="Reconstructed baseline",
        )
        # Label below-left: the frontier and the 99% ring both crowd the space
        # to the upper right of this point.
        ax.annotate(
            "current operating point",
            (bx, by),
            textcoords="offset points",
            xytext=(-14, -26),
            ha="right",
            fontsize=10,
            color=LIGHT["s2"],
            fontweight="bold",
            zorder=8,
            arrowprops={"arrowstyle": "-", "color": LIGHT["s2"], "linewidth": 1.2},
        )

    # The RECOMMENDED policy is not a point on this curve. The curve is what a
    # single uniform target buys; the recommendation allocates a service level
    # per series, so it sits off the curve entirely. Marking a point on the
    # uniform frontier as "recommended" would quietly misrepresent the result.
    cf_path = settings.artifacts_dir / "counterfactual.csv"
    if cf_path.exists():
        cf = pl.read_csv(cf_path)
        alloc = cf.filter(pl.col("policy") == "cost_optimal_allocation")
        if alloc.height:
            ax_ = float(alloc["fill_rate"][0]) * 100
            ay_ = float(alloc["total_cost"][0])
            ax.scatter(
                [ax_],
                [ay_],
                s=340,
                marker="*",
                color=LIGHT["good"],
                edgecolors=LIGHT["surface"],
                linewidths=1.8,
                zorder=9,
                label="Per-series allocation (recommended)",
            )
            ax.annotate(
                "cheaper AND higher service",
                (ax_, ay_),
                textcoords="offset points",
                xytext=(6, 24),
                ha="left",
                fontsize=10,
                color=LIGHT["good"],
                fontweight="bold",
                zorder=9,
                arrowprops={"arrowstyle": "-", "color": LIGHT["good"], "linewidth": 1.2},
            )

    ax.set_xlabel("Achieved fill rate (%)", fontsize=11, color=LIGHT["text_secondary"])
    ax.set_ylabel("Total cost, 28 days ($)", fontsize=11, color=LIGHT["text_secondary"])
    ax.yaxis.set_major_formatter(lambda v, _: f"${v / 1000:,.0f}k")

    # Headroom so the title, the subtitle and the topmost point label cannot
    # collide - the first render had all three overlapping.
    ax.margins(x=0.08)
    ax.set_ylim(top=max(y) * 1.10)

    fig.suptitle(
        "Service level vs cost",
        fontsize=16,
        color=LIGHT["text"],
        x=0.055,
        ha="left",
        y=0.995,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.915,
        "Cost FALLS as service rises — a lost unit costs ~26x holding one over the "
        "exposure window, so there is no\ninterior optimum. That is why the policy "
        "allocates per series rather than setting one target for the catalogue.",
        fontsize=9.5,
        color=LIGHT["text_secondary"],
        va="top",
        ha="left",
    )

    leg = ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=10,
        labelcolor=LIGHT["text_secondary"],
    )
    for t in leg.get_texts():
        t.set_color(LIGHT["text_secondary"])

    # Leave room at the top for suptitle + subtitle set in figure coords.
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = DOCS / "frontier.png"
    fig.savefig(out, facecolor=LIGHT["surface"], bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s (%.0f KB)", out, out.stat().st_size / 1024)


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    frontier()


if __name__ == "__main__":
    main()
