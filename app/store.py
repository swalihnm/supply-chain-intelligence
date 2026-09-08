"""Artifact loading for the serving layer.

Loaded once at process start and shared. Everything here is read-only, so it is
safe across FastAPI's threadpool and Streamlit's reruns.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

from sci.config import settings

BUNDLE_NAME = "serving_bundle.npz"


class ArtifactsMissing(RuntimeError):
    pass


@dataclass
class Store:
    series: pl.DataFrame
    history: np.ndarray
    demand: np.ndarray
    price: np.ndarray
    median_price: np.ndarray
    actual_test: np.ndarray
    forecasts: dict[str, np.ndarray]
    warmup_days: int
    valid_start_d: int
    valid_end_d: int
    tables: dict[str, pl.DataFrame]

    # ---- lookups ---------------------------------------------------------
    def index_of(self, store_id: str, item_id: str) -> int:
        hit = self.series.with_row_index("idx").filter(
            (pl.col("store_id") == store_id) & (pl.col("item_id") == item_id)
        )
        if hit.height == 0:
            raise KeyError(f"no series for store_id={store_id!r} item_id={item_id!r}")
        return int(hit["idx"][0])

    def meta(self, idx: int) -> dict:
        return self.series.row(idx, named=True)

    @property
    def n_series(self) -> int:
        return self.series.height

    def store_ids(self) -> list[str]:
        return sorted(self.series["store_id"].unique().to_list())

    def item_ids(self, store_id: str | None = None, limit: int | None = None) -> list[str]:
        df = self.series
        if store_id:
            df = df.filter(pl.col("store_id") == store_id)
        vals = df["item_id"].unique().sort().to_list()
        return vals[:limit] if limit else vals

    def has_forecast(self) -> bool:
        return "point" in self.forecasts

    def quantile_levels(self) -> list[float]:
        return sorted(float(k[1:]) for k in self.forecasts if k.startswith("q") and k != "point")


def _load_tables(art: Path) -> dict[str, pl.DataFrame]:
    """Every CSV artifact the dashboard might want, loaded if present."""
    names = [
        "backtest_summary",
        "backtest_folds",
        "backtest_by_sb_class",
        "backtest_by_abc_class",
        "calibration",
        "calibration_by_segment",
        "pinball",
        "feature_importance",
        "counterfactual",
        "counterfactual_ci",
        "service_level_frontier",
        "cost_sensitivity",
        "shadow_prices",
        "policy_test",
        "abc_xyz_segmentation",
        "stockout_proxy_detection",
    ]
    out = {}
    for n in names:
        p = art / f"{n}.csv"
        if p.exists():
            try:
                out[n] = pl.read_csv(p, infer_schema_length=10000)
            except Exception:  # noqa: BLE001 - a malformed artifact must not kill startup
                continue
    return out


@lru_cache(maxsize=1)
def get_store() -> Store:
    art = settings.artifacts_dir
    bundle = art / BUNDLE_NAME
    series_path = art / "serving_series.parquet"

    if not bundle.exists() or not series_path.exists():
        raise ArtifactsMissing(
            f"Serving artifacts not found in {art}.\n"
            "Build them with:\n"
            "  uv run python scripts/run_model_ladder.py\n"
            "  uv run python scripts/run_optimizer.py\n"
            "  uv run python scripts/build_serving_bundle.py"
        )

    z = np.load(bundle)
    forecasts = {k.removeprefix("forecast_"): z[k] for k in z.files if k.startswith("forecast_")}
    return Store(
        series=pl.read_parquet(series_path),
        history=z["history"],
        demand=z["demand"],
        price=z["price"],
        median_price=z["median_price"],
        actual_test=z["actual_test"],
        forecasts=forecasts,
        warmup_days=int(z["warmup_days"][0]),
        valid_start_d=int(z["valid_start_d"][0]),
        valid_end_d=int(z["valid_end_d"][0]),
        tables=_load_tables(art),
    )
