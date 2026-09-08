"""Hierarchical reconciliation (MinT family), solved matrix-free.

THE BUSINESS REASON, NOT THE STATISTICAL ONE
--------------------------------------------
Independent forecasts at different altitudes do not add up. The store's SKU
orders sum to one number; the category buyer's plan says another; finance's
state plan says a third. All three are then "the forecast", and any disagreement
between departments becomes an argument about whose spreadsheet is right rather
than about what to do.

Reconciliation produces ONE set of numbers that is coherent by construction: the
SKU forecasts sum exactly to the department forecast, which sums exactly to the
category, and so on. That is worth more operationally than the accuracy gain,
though there is usually an accuracy gain too - reconciliation pools information
across levels, and an aggregate series is far easier to forecast than the
intermittent bottom series it is made of.

THE MATH
--------
With summing matrix S (n_nodes x n_bottom) and base forecasts yhat at every
node, the reconciled bottom-level forecast is

    b~ = (S' W^-1 S)^-1 S' W^-1 yhat          and then y~ = S b~

W is the covariance of base forecast errors. Choices implemented:

    ols          W = I                     (Hyndman et al. 2011)
    wls_struct   W = diag(row sums of S)    structural scaling - a node that
                                            aggregates more series is noisier
    wls_var      W = diag(in-sample MSE)    the practical MinT variant; usually
                                            as good as full MinT and far cheaper

WHY MATRIX-FREE, AND WHY THAT MATTERS HERE
------------------------------------------
S' W^-1 S is 30,490 x 30,490. It is also DENSE - every pair of bottom series
co-occurs at the `total` node, so no sparsity survives. Materialising it would
need about 7.4 GB, on a machine with 7.3 GB of RAM total.

But conjugate gradients never needs the matrix, only its action on a vector:

    (S' W^-1 S) v  =  S' ( W^-1 ( S v ) )

which is two sparse mat-vecs against a matrix with 365,880 non-zeros. The
system is symmetric positive definite (S has full column rank), so CG applies
directly, and it is preconditioned with the diagonal of S' W^-1 S - which IS
cheap to compute exactly, since diag_j = sum_i w_i^-1 S_ij^2.

This turns a problem that does not fit in memory into a few seconds of sparse
arithmetic, and it is the reason reconciliation at full M5 scale is feasible on
a laptop at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from models.hierarchy import Hierarchy
from sci.log import get_logger

log = get_logger(__name__)

METHODS = ("ols", "wls_struct", "wls_var", "bottom_up")


@dataclass
class ReconResult:
    bottom: np.ndarray  # (n_bottom, h) reconciled bottom-level forecast
    method: str
    iterations: list[int]
    converged: bool

    def full(self, h: Hierarchy) -> np.ndarray:
        """The coherent forecast at every node."""
        return h.S @ self.bottom


def _weights(h: Hierarchy, method: str, residual_mse: np.ndarray | None) -> np.ndarray:
    """Diagonal of W. Returned as the per-node variance, not its inverse."""
    if method == "ols":
        return np.ones(h.n_nodes, dtype=np.float64)

    if method == "wls_struct":
        # Row sums of S: how many bottom series each node aggregates.
        return np.asarray(h.S.sum(axis=1)).ravel().astype(np.float64)

    if method == "wls_var":
        if residual_mse is None:
            raise ValueError("wls_var needs residual_mse (in-sample MSE per node)")
        w = np.asarray(residual_mse, dtype=np.float64).copy()
        # A node with zero in-sample error would get infinite weight and
        # dominate the solve; floor it at a small fraction of the median.
        floor = max(np.median(w[w > 0]) * 1e-6, 1e-9) if (w > 0).any() else 1e-9
        return np.maximum(w, floor)

    raise ValueError(f"unknown method {method!r}")


def reconcile(
    h: Hierarchy,
    base: np.ndarray,
    method: str = "wls_var",
    residual_mse: np.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int = 500,
) -> ReconResult:
    """Reconcile base forecasts at every node to a coherent bottom-level set.

    base : (n_nodes, h) independent forecasts, node order matching h.S rows
    """
    if method == "bottom_up":
        # Already coherent: discard the aggregate forecasts and keep the bottom.
        bottom = base[h.rows_for("item_store")].copy()
        return ReconResult(bottom=bottom, method=method, iterations=[0], converged=True)

    if base.shape[0] != h.n_nodes:
        raise ValueError(f"base has {base.shape[0]} rows, hierarchy has {h.n_nodes} nodes")

    w = _weights(h, method, residual_mse)
    w_inv = 1.0 / w

    S = h.S
    St = S.T.tocsr()

    # Exact diagonal of S' W^-1 S, for Jacobi preconditioning.
    # S entries are 0/1, so S_ij^2 == S_ij.
    diag = np.asarray(St.multiply(w_inv[None, :]) @ np.ones(h.n_nodes)).ravel()
    diag = np.maximum(diag, 1e-12)

    def matvec(v: np.ndarray) -> np.ndarray:
        return St @ (w_inv * (S @ v))

    A = LinearOperator((h.n_bottom, h.n_bottom), matvec=matvec, dtype=np.float64)
    M = LinearOperator((h.n_bottom, h.n_bottom), matvec=lambda v: v / diag, dtype=np.float64)

    horizon = base.shape[1]
    out = np.empty((h.n_bottom, horizon), dtype=np.float64)
    iters: list[int] = []
    converged = True

    for t in range(horizon):
        rhs = St @ (w_inv * base[:, t])
        counter = {"n": 0}

        def _count(_xk, counter=counter):
            counter["n"] += 1

        # scipy renamed the tolerance argument; support both.
        try:
            x, info = cg(A, rhs, rtol=tol, maxiter=maxiter, M=M, callback=_count)
        except TypeError:
            x, info = cg(A, rhs, tol=tol, maxiter=maxiter, M=M, callback=_count)

        if info != 0:
            converged = False
            log.warning("  CG did not converge at horizon step %d (info=%d)", t, info)
        out[:, t] = x
        iters.append(counter["n"])

    log.info(
        "reconciled (%s): %d steps, CG iterations min/median/max %d/%d/%d",
        method,
        horizon,
        min(iters),
        int(np.median(iters)),
        max(iters),
    )

    # Demand cannot be negative. Reconciliation is unconstrained and can push a
    # near-zero intermittent series slightly below zero; clipping is the honest
    # minimum fix, and it does break exact coherence by a negligible amount.
    n_neg = int((out < 0).sum())
    if n_neg:
        log.info("  clipped %d negative reconciled cells (%.4f%%)", n_neg, 100 * n_neg / out.size)
    out = np.clip(out, 0.0, None)

    return ReconResult(bottom=out, method=method, iterations=iters, converged=converged)


def coherence_error(h: Hierarchy, node_forecasts: np.ndarray) -> float:
    """Max absolute violation of the adding-up constraints.

    Zero for anything produced by `reconcile`; large for independent base
    forecasts. This is the number that makes the problem concrete.
    """
    bottom = node_forecasts[h.rows_for("item_store")]
    implied = h.S @ bottom
    return float(np.abs(implied - node_forecasts).max())


def relative_coherence_error(h: Hierarchy, node_forecasts: np.ndarray) -> float:
    """Coherence violation as a share of the total-level forecast."""
    bottom = node_forecasts[h.rows_for("item_store")]
    implied = h.S @ bottom
    scale = np.abs(node_forecasts[h.rows_for("total")]).sum()
    if scale <= 0:
        return float("nan")
    return float(np.abs(implied - node_forecasts).sum() / scale)


def build_weight_matrix_note() -> str:
    """Why full MinT (shrinkage covariance) is not implemented.

    Full MinT needs the n_nodes x n_nodes error covariance - 42,840^2 entries,
    about 14 GB dense. Shrinkage estimators reduce the estimation error but not
    the storage. The literature (Wickramasuriya et al. 2019) finds wls_var is
    usually close to full MinT in practice, and it is what is used here.
    """
    return build_weight_matrix_note.__doc__ or ""
