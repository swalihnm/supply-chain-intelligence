"""Run the full model ladder and write every artifact week 3 needs.

Usage:  uv run python scripts/run_model_ladder.py

Produces in $SCI_DATA_DIR/artifacts/:
    backtest_folds.csv            per-model, per-fold accuracy
    backtest_summary.csv          CV means, ranked
    backtest_by_sb_class.csv      error decomposition by series class
    backtest_by_abc_class.csv     error decomposition by revenue class
    feature_importance.csv        global model gain shares
    calibration.csv               nominal vs empirical quantile coverage
    calibration_by_segment.csv    the same, per series class
    pinball.csv                   pinball loss by level and method
    forecast_test.npz             TEST-fold point + quantile forecasts

COMPUTE BUDGET
--------------
Quantile models are fitted on three folds, not five: cv4 and cv5 to assess
calibration, and the locked test fold because the optimizer consumes it. Point
models and baselines run on every fold. Fitting five quantile levels on five
folds would roughly triple the runtime to buy a third and fourth estimate of a
calibration gap that is already stable across two.

THE TEST FOLD
-------------
This script scores the locked test set. That is the one permitted touch defined
in evaluation/protocol.py. Re-running it after seeing the number and changing a
model would break the protocol - if that becomes necessary, the honest move is
to say so in the results.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from evaluation import metrics
from evaluation.backtest import error_by_segment, run_ladder, summarise
from evaluation.protocol import FIRST_D, QUANTILES, cv_folds, describe, test_fold
from features import panel_matrix
from models import global_ml, quantile
from models.baselines import REGISTRY as BASELINES
from sci.config import settings
from sci.log import get_logger

log = get_logger(__name__)

QUANTILE_FOLDS = ("cv4", "cv5", "test")
NUM_BOOST_ROUND = 300


def main() -> None:
    settings.ensure_dirs()
    art = settings.artifacts_dir
    print(describe())

    pm = panel_matrix.build()
    folds = cv_folds() + [test_fold()]

    # ---------------------------------------------------------------- baselines
    log.info("=== baselines ===")
    base_df, base_results = run_ladder(pm, BASELINES, include_test=True)

    # --------------------------------------------------------------- global ML
    log.info("=== global model ===")
    rows = []
    global_results = []
    importances = []
    quantile_forecasts: dict[str, quantile.QuantileForecast] = {}

    for fold in folds:
        log.info("fold %s", fold)
        dm = global_ml._training_matrix(pm, fold, seed=0)

        want_q = fold.name in QUANTILE_FOLDS
        if want_q:
            qf = quantile.train_quantiles(
                pm, fold, quantiles=QUANTILES, num_boost_round=NUM_BOOST_ROUND, dm=dm
            )
            quantile_forecasts[fold.name] = qf
            pred = qf.point
            model = qf.models[QUANTILES[0]]
            point_importance = None
        else:
            model = global_ml.train(pm, fold, num_boost_round=NUM_BOOST_ROUND, dm=dm)
            pred = global_ml.predict(model, pm, fold)
            point_importance = model.importance

        del dm

        # score the point forecast
        t_end = fold.train_end_d - FIRST_D + 1
        train_raw = pm.units[:, :t_end]
        price_train = pm.price[:, :t_end]
        v_lo = fold.valid_start_d - FIRST_D
        actual = pm.units[:, v_lo : v_lo + fold.horizon]

        scale_sq = metrics.naive_scale_squared(train_raw)
        scale_abs = metrics.naive_scale_absolute(train_raw)
        weights = metrics.m5_weights(train_raw, price_train)
        per_rmsse = metrics.rmsse(actual, np.clip(pred, 0, None), scale_sq)
        per_mase = metrics.mase(actual, np.clip(pred, 0, None), scale_abs)

        rows.append(
            {
                "model": "global_lgbm",
                "fold": fold.name,
                "wrmsse": metrics.wrmsse(per_rmsse, weights),
                "rmsse_mean": float(np.nanmean(per_rmsse)),
                "mase_mean": float(np.nanmean(per_mase)),
                "n_series_scored": int(np.isfinite(per_rmsse).sum()),
                "fit_seconds": 0.0,
            }
        )
        log.info("global_lgbm %-5s WRMSSE %.4f", fold.name, rows[-1]["wrmsse"])

        from evaluation.backtest import FoldResult

        global_results.append(
            FoldResult(
                model="global_lgbm",
                fold=fold.name,
                wrmsse=rows[-1]["wrmsse"],
                rmsse_mean=rows[-1]["rmsse_mean"],
                mase_mean=rows[-1]["mase_mean"],
                n_series_scored=rows[-1]["n_series_scored"],
                fit_seconds=0.0,
                per_series_rmsse=per_rmsse,
                weights=weights,
            )
        )
        if point_importance:
            importances.append(
                pl.DataFrame(
                    {
                        "fold": fold.name,
                        "feature": list(point_importance),
                        "gain_share": list(point_importance.values()),
                    }
                )
            )

    # ------------------------------------------------------------------ combine
    all_df = pl.concat([base_df, pl.DataFrame(rows)], how="vertical_relaxed")
    all_results = base_results + global_results

    all_df.write_csv(art / "backtest_folds.csv")
    summary = summarise(all_df)
    summary.write_csv(art / "backtest_summary.csv")
    print("\n=== CV SUMMARY ===")
    print(summary)

    print("\n=== TEST FOLD (scored once) ===")
    print(all_df.filter(pl.col("fold") == "test").sort("wrmsse"))

    error_by_segment(pm, all_results, "sb_class").write_csv(art / "backtest_by_sb_class.csv")
    error_by_segment(pm, all_results, "abc_class").write_csv(art / "backtest_by_abc_class.csv")

    if importances:
        pl.concat(importances).write_csv(art / "feature_importance.csv")

    # -------------------------------------------------------------- calibration
    cal_rows, cal_seg_rows, pin_rows = [], [], []
    sb = pm.series["sb_class"].to_numpy()

    for fold in folds:
        if fold.name not in quantile_forecasts:
            continue
        qf = quantile_forecasts[fold.name]
        v_lo = fold.valid_start_d - FIRST_D
        actual = pm.units[:, v_lo : v_lo + fold.horizon]
        t_end = fold.train_end_d - FIRST_D + 1
        weights = metrics.m5_weights(pm.units[:, :t_end], pm.price[:, :t_end])

        cal = quantile.calibration_report(actual, qf.by_quantile, label="lgbm_quantile")
        cal_rows.append(cal.with_columns(pl.lit(fold.name).alias("fold")))

        seg = quantile.calibration_by_segment(actual, qf.by_quantile, sb)
        cal_seg_rows.append(seg.with_columns(pl.lit(fold.name).alias("fold")))

        pin = quantile.pinball_report(
            actual, qf.by_quantile, weights=weights, label="lgbm_quantile"
        )
        pin_rows.append(pin.with_columns(pl.lit(fold.name).alias("fold")))

    # Conformal, calibrated on cv5 and applied to the test fold.
    if "cv5" in quantile_forecasts and "test" in quantile_forecasts:
        cv5, tst = cv_folds()[-1], test_fold()
        a_lo = cv5.valid_start_d - FIRST_D
        actual_cal = pm.units[:, a_lo : a_lo + cv5.horizon]
        t_lo = tst.valid_start_d - FIRST_D
        actual_test = pm.units[:, t_lo : t_lo + tst.horizon]

        conf = quantile.conformal_quantiles(
            point_cal=quantile_forecasts["cv5"].point,
            actual_cal=actual_cal,
            point_new=quantile_forecasts["test"].point,
            quantiles=QUANTILES,
            by_group=sb,
        )
        cal_rows.append(
            quantile.calibration_report(actual_test, conf, label="conformal").with_columns(
                pl.lit("test").alias("fold")
            )
        )
        pin_rows.append(
            quantile.pinball_report(actual_test, conf, label="conformal").with_columns(
                pl.lit("test").alias("fold")
            )
        )

    if cal_rows:
        cal_all = pl.concat(cal_rows, how="vertical_relaxed")
        cal_all.write_csv(art / "calibration.csv")
        print("\n=== QUANTILE CALIBRATION ===")
        print(cal_all)
        pl.concat(cal_seg_rows, how="vertical_relaxed").write_csv(
            art / "calibration_by_segment.csv"
        )
        pl.concat(pin_rows, how="vertical_relaxed").write_csv(art / "pinball.csv")

    # ------------------------------------------------- forecasts for the optimizer
    if "test" in quantile_forecasts:
        qf = quantile_forecasts["test"]
        np.savez_compressed(
            art / "forecast_test.npz",
            point=qf.point,
            **{f"q{q}": v for q, v in qf.by_quantile.items()},
        )
        log.info("wrote forecast_test.npz for the optimizer")

    log.info("done - artifacts in %s", art)


if __name__ == "__main__":
    main()
