"""Copy the artifacts the container actually serves into the build context.

Usage:  uv run python scripts/stage_deploy.py

SCI_DATA_DIR points outside the repo (it holds a 163 MB warehouse and ~500 MB of
matrices), so the Docker build context cannot reach it. This copies only what
the API and dashboard read - the serving bundle plus the result tables - into
deploy/data/artifacts/, which is what the Dockerfile COPYs.

Deliberately excluded: the DuckDB warehouse, staging parquet, raw pulls, and
panel_matrices.npz. None are needed to serve, and shipping them would turn a
~35 MB image layer into a gigabyte.
"""

from __future__ import annotations

import shutil

from sci.config import REPO_ROOT, settings
from sci.log import get_logger

log = get_logger(__name__)

DEPLOY_ARTIFACTS = REPO_ROOT / "deploy" / "data" / "artifacts"

REQUIRED = ["serving_bundle.npz", "serving_series.parquet"]
OPTIONAL = [
    "serving_calendar.parquet",
    "backtest_summary.csv",
    "backtest_folds.csv",
    "backtest_by_sb_class.csv",
    "backtest_by_abc_class.csv",
    "calibration.csv",
    "calibration_by_segment.csv",
    "pinball.csv",
    "feature_importance.csv",
    "counterfactual.csv",
    "counterfactual_ci.csv",
    "service_level_frontier.csv",
    "cost_sensitivity.csv",
    "shadow_prices.csv",
    "policy_test.csv",
    "abc_xyz_segmentation.csv",
    "stockout_proxy_detection.csv",
    "reconciliation_accuracy.csv",
    "reconciliation_coherence.csv",
    "hierarchy_levels.csv",
]


def main() -> None:
    src = settings.artifacts_dir
    DEPLOY_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    missing = [n for n in REQUIRED if not (src / n).exists()]
    if missing:
        raise SystemExit(
            f"Missing required artifacts in {src}: {missing}\n"
            "Run scripts/build_serving_bundle.py first."
        )

    total = 0
    for name in REQUIRED + OPTIONAL:
        p = src / name
        if not p.exists():
            log.info("  skip (absent) %s", name)
            continue
        shutil.copy2(p, DEPLOY_ARTIFACTS / name)
        total += p.stat().st_size
        log.info("  staged %-36s %6.1f MB", name, p.stat().st_size / 1e6)

    log.info("staged %.1f MB into %s", total / 1e6, DEPLOY_ARTIFACTS)


if __name__ == "__main__":
    main()
