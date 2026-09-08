"""Hierarchy construction and reconciliation correctness.

The load-bearing test is `test_matrix_free_matches_dense_closed_form`: the
production solver never builds S'W^-1S, so it is verified against the textbook
dense formula on a hierarchy small enough for that formula to be computed
directly.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from models import hierarchy, reconcile


@pytest.fixture
def small_series() -> pl.DataFrame:
    """2 items x 2 stores across 2 states - small enough to solve densely."""
    rows = []
    for item, dept, cat in [("i1", "d1", "c1"), ("i2", "d2", "c2")]:
        for store, state in [("s1", "CA"), ("s2", "TX")]:
            rows.append(
                {
                    "item_id": item,
                    "store_id": store,
                    "dept_id": dept,
                    "cat_id": cat,
                    "state_id": state,
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def h(small_series):
    return hierarchy.build(small_series)


# --- hierarchy construction -----------------------------------------------


def test_each_bottom_series_appears_once_per_level(h):
    """Every column of S must have exactly one 1 per level - a series belongs
    to exactly one state, one store, one category, and so on."""
    S = h.S.toarray()
    for level in h.level_names:
        block = S[h.rows_for(level)]
        assert (block.sum(axis=0) == 1).all(), f"level {level} does not partition the bottom"


def test_total_level_sums_everything(h):
    b = np.arange(1, h.n_bottom + 1, dtype=float)[:, None]
    agg = h.aggregate(b)
    assert agg[h.rows_for("total")][0, 0] == pytest.approx(b.sum())


def test_bottom_block_is_the_identity(h):
    block = h.S[h.rows_for("item_store")].toarray()
    np.testing.assert_allclose(block, np.eye(h.n_bottom))


def test_store_level_matches_a_manual_groupby(h, small_series):
    b = np.array([[1.0], [2.0], [3.0], [4.0]])
    agg = h.aggregate(b)
    stores = small_series["store_id"].to_numpy()
    for i, label in enumerate(h.node_labels[h.rows_for("store")]):
        store = label.split(":")[1]
        expected = b[stores == store].sum()
        assert agg[h.rows_for("store").start + i, 0] == pytest.approx(expected)


def test_full_m5_level_counts(h):
    counts = dict(
        zip(
            h.level_names,
            [h.rows_for(n).stop - h.rows_for(n).start for n in h.level_names],
            strict=True,
        )
    )
    assert counts["total"] == 1
    assert counts["state"] == 2
    assert counts["item_store"] == 4


def test_sparsity_is_exactly_one_entry_per_level_per_series(h):
    assert h.S.nnz == len(h.level_names) * h.n_bottom


# --- reconciliation correctness --------------------------------------------


def _dense_reference(h, base, w_diag):
    """The textbook closed form: b = (S'W^-1 S)^-1 S'W^-1 yhat."""
    S = h.S.toarray()
    w_inv = np.diag(1.0 / w_diag)
    A = S.T @ w_inv @ S
    rhs = S.T @ w_inv @ base
    return np.linalg.solve(A, rhs)


@pytest.mark.parametrize("method", ["ols", "wls_struct", "wls_var"])
def test_matrix_free_matches_dense_closed_form(h, method):
    """The whole reason the production solver is trustworthy.

    reconcile() uses preconditioned CG against an operator and never forms
    S'W^-1S. Here the dense matrix IS formed and inverted directly, and the two
    must agree.
    """
    rng = np.random.default_rng(0)
    base = rng.uniform(1, 20, size=(h.n_nodes, 3))
    mse = rng.uniform(0.5, 5.0, size=h.n_nodes)

    w = reconcile._weights(h, method, mse)
    expected = _dense_reference(h, base, w)

    got = reconcile.reconcile(h, base, method=method, residual_mse=mse, tol=1e-12).bottom

    # The production path clips negatives; the reference does not. Compare only
    # where the reference is non-negative so the clip cannot mask a real error.
    mask = expected >= 0
    np.testing.assert_allclose(got[mask], expected[mask], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("method", reconcile.METHODS)
def test_reconciled_forecasts_are_coherent(h, method):
    """The property the whole exercise exists to buy."""
    rng = np.random.default_rng(1)
    base = rng.uniform(1, 20, size=(h.n_nodes, 4))
    mse = rng.uniform(0.5, 5.0, size=h.n_nodes)

    r = reconcile.reconcile(h, base, method=method, residual_mse=mse)
    full = r.full(h)
    assert reconcile.coherence_error(h, full) < 1e-6


def test_base_forecasts_are_NOT_coherent(h):
    """Sanity check on the test above: independent forecasts really do violate
    the adding-up constraints, so a passing coherence test means something."""
    rng = np.random.default_rng(2)
    base = rng.uniform(1, 20, size=(h.n_nodes, 4))
    assert reconcile.coherence_error(h, base) > 1.0


@pytest.mark.parametrize("method", ["ols", "wls_struct"])
def test_reconciliation_is_idempotent_on_coherent_input(h, method):
    """Reconciling an already-coherent set must return it unchanged."""
    rng = np.random.default_rng(3)
    bottom = rng.uniform(1, 10, size=(h.n_bottom, 3))
    coherent = h.aggregate(bottom)

    r = reconcile.reconcile(h, coherent, method=method, residual_mse=np.ones(h.n_nodes))
    np.testing.assert_allclose(r.bottom, bottom, rtol=1e-6, atol=1e-8)


def test_bottom_up_ignores_the_aggregate_forecasts(h):
    rng = np.random.default_rng(4)
    base = rng.uniform(1, 20, size=(h.n_nodes, 3))
    r = reconcile.reconcile(h, base, method="bottom_up")
    np.testing.assert_allclose(r.bottom, base[h.rows_for("item_store")])


def test_reconciled_output_is_never_negative(h):
    rng = np.random.default_rng(5)
    # Aggregates far below what the bottom implies -> pushes some bottoms down.
    base = rng.uniform(1, 5, size=(h.n_nodes, 3))
    base[h.rows_for("total")] = 0.01
    r = reconcile.reconcile(h, base, method="ols")
    assert (r.bottom >= 0).all()


def test_unknown_method_raises(h):
    with pytest.raises(ValueError, match="unknown method"):
        reconcile._weights(h, "not_a_method", None)


def test_wls_var_requires_residuals(h):
    with pytest.raises(ValueError, match="residual_mse"):
        reconcile._weights(h, "wls_var", None)


def test_mismatched_base_shape_raises(h):
    with pytest.raises(ValueError, match="hierarchy has"):
        reconcile.reconcile(h, np.zeros((h.n_nodes + 5, 2)), method="ols")


# --- the real hierarchy ----------------------------------------------------


def test_real_hierarchy_matches_published_m5_structure():
    from sci.config import settings

    if not (settings.artifacts_dir / "panel_matrices.npz").exists():
        pytest.skip("panel matrices not built")

    from features import panel_matrix

    pm = panel_matrix.build()
    hh = hierarchy.build(pm.series)

    expected = {
        "total": 1,
        "state": 3,
        "store": 10,
        "category": 3,
        "department": 7,
        "state_category": 9,
        "state_department": 21,
        "store_category": 30,
        "store_department": 70,
        "item": 3049,
        "item_state": 9147,
        "item_store": 30490,
    }
    for level, n in expected.items():
        got = hh.rows_for(level).stop - hh.rows_for(level).start
        assert got == n, f"{level}: expected {n} nodes, got {got}"
    assert hh.n_nodes == sum(expected.values()) == 42840
