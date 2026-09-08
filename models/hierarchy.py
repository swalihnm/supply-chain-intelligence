"""The M5 aggregation structure, as a sparse summing matrix.

WHY A SUMMING MATRIX
--------------------
Forecasts get consumed at different altitudes. A store orders at SKU level; a
category buyer plans at department level; finance plans at state and total. If
those numbers are produced independently they will not add up, and then two
people hold two different plans for the same business and nobody can say which
is wrong.

The summing matrix S encodes "which bottom series roll into which aggregate".
Given bottom-level values b, the whole hierarchy is S @ b. Reconciliation is the
problem of finding the b that best matches independently-made forecasts at every
level.

THE 12 LEVELS
-------------
M5's published structure. Counts here are the measured ones for this catalogue:

    1  total                      1
    2  state                      3
    3  store                     10
    4  category                   3
    5  department                 7
    6  state x category           9
    7  state x department        21
    8  store x category          30
    9  store x department        70
   10  item                   3,049
   11  item x state           9,147
   12  item x store          30,490   <- bottom

42,840 nodes over 30,490 bottom series.

This is a GROUPED structure, not a strict tree: `item` aggregates across stores
while `store` aggregates across items, and neither is nested in the other. That
rules out the simple recursive tree algorithms and is why reconciliation here
needs a linear solve.

SPARSITY IS THE WHOLE POINT
---------------------------
S is 42,840 x 30,490 = 1.3 billion entries, which cannot be materialised. But
each bottom series belongs to exactly one node per level, so S has exactly
12 x 30,490 = 365,880 non-zeros - a density of 0.03%. Stored sparse it is a few
megabytes, and every operation downstream is a sparse mat-vec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import sparse

from sci.log import get_logger

log = get_logger(__name__)

# (level name, the series columns whose combination defines a node)
LEVEL_SPEC: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("total", ()),
    ("state", ("state_id",)),
    ("store", ("store_id",)),
    ("category", ("cat_id",)),
    ("department", ("dept_id",)),
    ("state_category", ("state_id", "cat_id")),
    ("state_department", ("state_id", "dept_id")),
    ("store_category", ("store_id", "cat_id")),
    ("store_department", ("store_id", "dept_id")),
    ("item", ("item_id",)),
    ("item_state", ("item_id", "state_id")),
    ("item_store", ("item_id", "store_id")),  # bottom
)


@dataclass
class Hierarchy:
    """A sparse summing matrix plus the labels needed to interpret its rows."""

    S: sparse.csr_matrix  # (n_nodes, n_bottom)
    level_of_row: np.ndarray  # (n_nodes,) index into level_names
    level_names: list[str]
    node_labels: list[str]
    level_slices: dict[str, slice]

    @property
    def n_nodes(self) -> int:
        return self.S.shape[0]

    @property
    def n_bottom(self) -> int:
        return self.S.shape[1]

    def aggregate(self, bottom: np.ndarray) -> np.ndarray:
        """Roll bottom-level values up to every node. bottom is (n_bottom, h)."""
        return self.S @ bottom

    def rows_for(self, level: str) -> slice:
        return self.level_slices[level]

    def summary(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "level": self.level_names,
                "n_nodes": [
                    self.level_slices[n].stop - self.level_slices[n].start for n in self.level_names
                ],
            }
        )


def build(series: pl.DataFrame) -> Hierarchy:
    """Construct the summing matrix from the series metadata table.

    `series` must carry item_id, store_id, dept_id, cat_id, state_id in the same
    row order as the bottom-level forecast matrix - row i of S's columns is
    series i everywhere downstream.
    """
    n_bottom = series.height
    key_cols = ("item_id", "store_id", "dept_id", "cat_id", "state_id")
    cols = {c: series[c].to_numpy() for c in key_cols}

    rows_idx: list[int] = []
    cols_idx: list[int] = []
    node_labels: list[str] = []
    level_of_row: list[int] = []
    level_slices: dict[str, slice] = {}
    level_names: list[str] = []

    row_cursor = 0
    for li, (level_name, keys) in enumerate(LEVEL_SPEC):
        level_names.append(level_name)
        start = row_cursor

        if not keys:
            # Total: every bottom series feeds the single node.
            rows_idx.extend([row_cursor] * n_bottom)
            cols_idx.extend(range(n_bottom))
            node_labels.append("total")
            level_of_row.append(li)
            row_cursor += 1
        else:
            # Group key per bottom series, then one node per distinct key.
            key_arr = np.array(["|".join(str(cols[k][i]) for k in keys) for i in range(n_bottom)])
            uniq, inverse = np.unique(key_arr, return_inverse=True)
            rows_idx.extend((inverse + row_cursor).tolist())
            cols_idx.extend(range(n_bottom))
            node_labels.extend(f"{level_name}:{u}" for u in uniq)
            level_of_row.extend([li] * len(uniq))
            row_cursor += len(uniq)

        level_slices[level_name] = slice(start, row_cursor)
        log.info("  level %-18s %6d nodes", level_name, row_cursor - start)

    S = sparse.csr_matrix(
        (np.ones(len(rows_idx), dtype=np.float64), (rows_idx, cols_idx)),
        shape=(row_cursor, n_bottom),
    )

    log.info(
        "hierarchy: %d nodes over %d bottom series, %d non-zeros (%.4f%% dense)",
        row_cursor,
        n_bottom,
        S.nnz,
        100.0 * S.nnz / (row_cursor * n_bottom),
    )

    return Hierarchy(
        S=S,
        level_of_row=np.array(level_of_row, dtype=np.int32),
        level_names=level_names,
        node_labels=node_labels,
        level_slices=level_slices,
    )
