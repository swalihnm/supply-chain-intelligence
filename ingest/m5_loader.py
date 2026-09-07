"""M5 / Walmart transaction spine -> typed parquet in staging.

Source is a local Kaggle download (zip or already-unpacked folder), pointed at
by SCI_M5_SOURCE. The competition file we care about is
`sales_train_evaluation.csv`: 30,490 SKU-store series x 1,941 days held in wide
form, one column per day. We unpivot it to long form, which is the shape every
downstream layer wants.

The unpivot produces ~59M rows, so it runs through the Polars streaming engine
and sinks straight to parquet rather than materialising in memory.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl

from sci.config import settings
from sci.log import get_logger
from sci.manifest import Manifest, file_sha256, text_sha256

log = get_logger(__name__)

# The three files the pipeline actually needs. `sales_train_evaluation.csv`
# carries d_1..d_1941; `sales_train_validation.csv` is the same data truncated
# at d_1913 and is not used.
REQUIRED = ("calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv")

N_DAYS = 1941
N_SERIES = 30_490


class M5SourceMissing(RuntimeError):
    pass


def _raw_dir() -> Path:
    return settings.raw_dir / "m5"


def resolve_source() -> Path:
    """Return a directory containing the M5 CSVs, unpacking the zip if needed."""
    raw = settings.m5_source.strip()
    if not raw:
        raise M5SourceMissing(
            "SCI_M5_SOURCE is not set in .env.\n"
            "Download the M5 data from "
            "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data "
            "(accept the rules, then 'Download All'), and point SCI_M5_SOURCE at "
            "either the downloaded .zip or the folder you unpacked it into."
        )

    src = Path(raw).expanduser()
    if not src.exists():
        raise M5SourceMissing(f"SCI_M5_SOURCE points at {src}, which does not exist.")

    if src.is_dir():
        _assert_complete(src)
        return src

    if src.suffix.lower() != ".zip":
        raise M5SourceMissing(f"SCI_M5_SOURCE must be a .zip or a directory, got {src}")

    dest = _raw_dir()
    dest.mkdir(parents=True, exist_ok=True)
    if all((dest / name).exists() for name in REQUIRED):
        log.info("M5 zip already unpacked at %s", dest)
        return dest

    log.info("Unpacking %s -> %s", src.name, dest)
    with zipfile.ZipFile(src) as zf:
        members = [m for m in zf.namelist() if Path(m).name in REQUIRED]
        missing = set(REQUIRED) - {Path(m).name for m in members}
        if missing:
            raise M5SourceMissing(f"{src.name} is missing {sorted(missing)}")
        for m in members:
            # Flatten any directory prefix the zip carries.
            target = dest / Path(m).name
            with zf.open(m) as fh, open(target, "wb") as out:
                out.write(fh.read())
    _assert_complete(dest)
    return dest


def _assert_complete(d: Path) -> None:
    missing = [n for n in REQUIRED if not (d / n).exists()]
    if missing:
        raise M5SourceMissing(f"{d} is missing required files: {missing}")


# --------------------------------------------------------------------------
# individual table loads
# --------------------------------------------------------------------------


def _load_calendar(src: Path, out: Path) -> int:
    df = (
        pl.scan_csv(src / "calendar.csv")
        .with_columns(
            pl.col("date").str.to_date("%Y-%m-%d"),
            pl.col("d").str.strip_prefix("d_").cast(pl.Int16),
            pl.col("wm_yr_wk").cast(pl.Int32),
            pl.col("wday").cast(pl.Int8),
            pl.col("month").cast(pl.Int8),
            pl.col("year").cast(pl.Int16),
            pl.col("snap_CA").cast(pl.Int8),
            pl.col("snap_TX").cast(pl.Int8),
            pl.col("snap_WI").cast(pl.Int8),
        )
        .select(
            "d",
            "date",
            "wm_yr_wk",
            "weekday",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        )
        .collect()
    )
    df.write_parquet(out, compression="zstd")
    return df.height


def _load_prices(src: Path, out: Path) -> int:
    df = (
        pl.scan_csv(src / "sell_prices.csv")
        .with_columns(
            pl.col("wm_yr_wk").cast(pl.Int32),
            pl.col("sell_price").cast(pl.Float32),
        )
        .collect()
    )
    df.write_parquet(out, compression="zstd")
    return df.height


def _load_sales_long(src: Path, out: Path) -> int:
    """Unpivot d_1..d_1941 into (item_id, store_id, d, units)."""
    path = src / "sales_train_evaluation.csv"
    day_cols = [c for c in pl.read_csv(path, n_rows=0).columns if c.startswith("d_")]
    if not day_cols:
        raise ValueError(f"{path} has no d_* columns")
    log.info("Unpivoting %d day columns", len(day_cols))

    lf = (
        pl.scan_csv(path, schema_overrides=dict.fromkeys(day_cols, pl.Int16))
        .unpivot(
            index=["item_id", "store_id"],
            on=day_cols,
            variable_name="d",
            value_name="units",
        )
        .with_columns(pl.col("d").str.strip_prefix("d_").cast(pl.Int16))
        .select("item_id", "store_id", "d", "units")
    )
    lf.sink_parquet(out, compression="zstd")
    return pl.scan_parquet(out).select(pl.len()).collect().item()


def _load_dims(src: Path, item_out: Path, store_out: Path) -> tuple[int, int]:
    """Series-level attributes, deduplicated out of the wide sales file."""
    head = pl.read_csv(
        src / "sales_train_evaluation.csv",
        columns=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
    )
    items = head.select("item_id", "dept_id", "cat_id").unique().sort("item_id")
    stores = head.select("store_id", "state_id").unique().sort("store_id")
    items.write_parquet(item_out, compression="zstd")
    stores.write_parquet(store_out, compression="zstd")
    return items.height, stores.height


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------


def ingest_m5(force: bool = False) -> dict[str, int]:
    """Load all M5 tables to staging parquet. Returns {table: row_count}."""
    settings.ensure_dirs()
    src = resolve_source()
    stage = settings.staging_dir / "m5"
    stage.mkdir(parents=True, exist_ok=True)
    mf = Manifest(settings.staging_dir / "_manifest.json")

    counts: dict[str, int] = {}

    plan = [
        ("m5_calendar", "calendar.csv", stage / "calendar.parquet", _load_calendar),
        ("m5_sell_prices", "sell_prices.csv", stage / "sell_prices.parquet", _load_prices),
        (
            "m5_sales_long",
            "sales_train_evaluation.csv",
            stage / "sales_long.parquet",
            _load_sales_long,
        ),
    ]

    for key, source_file, out, fn in plan:
        input_sha = file_sha256(src / source_file)
        if not force and mf.is_current(key, input_sha, out):
            rows = (mf.get(key) or {}).get("rows", 0)
            log.info("%-16s up to date (%s rows) - skipping", key, f"{rows:,}")
            counts[key] = rows
            continue
        log.info("%-16s building -> %s", key, out.name)
        rows = fn(src, out)
        mf.record(key, input_sha, out, rows, source=source_file)
        log.info("%-16s wrote %s rows", key, f"{rows:,}")
        counts[key] = rows

    # Dims are cheap, and rebuild whenever the wide sales file changes.
    dim_key = "m5_dims"
    dim_sha = text_sha256(file_sha256(src / "sales_train_evaluation.csv"), "dims-v1")
    item_out = stage / "dim_item.parquet"
    store_out = stage / "dim_store.parquet"
    if force or not mf.is_current(dim_key, dim_sha, item_out) or not store_out.exists():
        n_items, n_stores = _load_dims(src, item_out, store_out)
        mf.record(dim_key, dim_sha, item_out, n_items, n_stores=n_stores)
        log.info("%-16s wrote %d items, %d stores", dim_key, n_items, n_stores)
        counts["m5_items"] = n_items
        counts["m5_stores"] = n_stores
    else:
        entry = mf.get(dim_key) or {}
        counts["m5_items"] = entry.get("rows", 0)
        counts["m5_stores"] = entry.get("n_stores", 0)
        log.info("%-16s up to date - skipping", dim_key)

    return counts
