"""Command-line entrypoint: `uv run sci ...`"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from sci.config import settings
from sci.log import get_logger

app = typer.Typer(help="Supply chain intelligence pipeline.", no_args_is_help=True)
ingest_app = typer.Typer(help="Pull sources into staging parquet.", no_args_is_help=True)
wh_app = typer.Typer(help="Build and inspect the DuckDB warehouse.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(wh_app, name="warehouse")

console = Console()
log = get_logger(__name__)


@app.command()
def config() -> None:
    """Show resolved paths and which credentials are present."""
    t = Table(title="sci configuration", show_header=False)
    t.add_column("key", style="cyan")
    t.add_column("value")
    t.add_row("data_dir", str(settings.data_dir))
    t.add_row("  raw", str(settings.raw_dir))
    t.add_row("  staging", str(settings.staging_dir))
    t.add_row("  warehouse", str(settings.duckdb_path))
    t.add_row("  features", str(settings.features_dir))
    t.add_row("  artifacts", str(settings.artifacts_dir))
    t.add_row("M5 source", settings.m5_source or "[yellow]not set[/yellow]")
    t.add_row(
        "FRED key", "[green]set[/green]" if settings.fred_api_key else "[yellow]not set[/yellow]"
    )
    console.print(t)


@ingest_app.command("m5")
def ingest_m5_cmd(
    force: bool = typer.Option(False, "--force", help="Rebuild even if inputs are unchanged."),
) -> None:
    """Load the M5 transaction spine from the local Kaggle download."""
    from ingest.m5_loader import M5SourceMissing, ingest_m5

    try:
        counts = ingest_m5(force=force)
    except M5SourceMissing as exc:
        console.print(f"[red]M5 source not available.[/red]\n{exc}")
        raise typer.Exit(code=1) from exc
    _report(counts)


@ingest_app.command("weather")
def ingest_weather_cmd(
    force: bool = typer.Option(False, "--force", help="Re-fetch even if cached."),
) -> None:
    """Pull daily historical weather per state from Open-Meteo (no key needed)."""
    from ingest.weather_client import ingest_weather

    _report({"weather_daily": ingest_weather(force=force)})


@ingest_app.command("fred")
def ingest_fred_cmd(
    force: bool = typer.Option(False, "--force", help="Re-fetch even if cached."),
) -> None:
    """Pull macro covariates from FRED. Skipped when FRED_API_KEY is unset."""
    from ingest.fred_client import FredKeyMissing, ingest_fred

    try:
        _report({"macro_series": ingest_fred(force=force)})
    except FredKeyMissing as exc:
        console.print(f"[yellow]Skipping FRED.[/yellow] {exc}")
        raise typer.Exit(code=0) from exc


@ingest_app.command("all")
def ingest_all_cmd(
    force: bool = typer.Option(False, "--force", help="Rebuild everything."),
) -> None:
    """Run every ingest step. Optional sources that are unavailable are skipped."""
    from ingest.fred_client import FredKeyMissing, ingest_fred
    from ingest.m5_loader import M5SourceMissing, ingest_m5
    from ingest.weather_client import ingest_weather

    counts: dict[str, int] = {}
    try:
        counts.update(ingest_m5(force=force))
    except M5SourceMissing as exc:
        console.print(f"[yellow]Skipping M5.[/yellow] {exc}")
    counts["weather_daily"] = ingest_weather(force=force)
    try:
        counts["macro_series"] = ingest_fred(force=force)
    except FredKeyMissing as exc:
        console.print(f"[yellow]Skipping FRED.[/yellow] {exc}")
    _report(counts)


@wh_app.command("build")
def warehouse_build_cmd() -> None:
    """Run the DDL and load every staged table into DuckDB."""
    from sci.warehouse import build

    _report(build())


@wh_app.command("tables")
def warehouse_tables_cmd() -> None:
    """List warehouse tables with their row counts."""
    from sci.warehouse import connect

    with connect(read_only=True) as con:
        rows = con.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
            """
        ).fetchall()
        t = Table(title="warehouse")
        t.add_column("table", style="cyan")
        t.add_column("rows", justify="right")
        for schema, name in rows:
            n = con.execute(f'SELECT count(*) FROM "{schema}"."{name}"').fetchone()[0]
            t.add_row(f"{schema}.{name}", f"{n:,}")
        console.print(t)


@wh_app.command("query")
def warehouse_query_cmd(
    sql: str = typer.Argument(..., help="SQL to run, or a path to a .sql file."),
    limit: int = typer.Option(25, help="Max rows to print."),
) -> None:
    """Run ad-hoc SQL against the warehouse."""
    from pathlib import Path

    from sci.warehouse import connect

    p = Path(sql)
    text = p.read_text(encoding="utf-8") if p.suffix == ".sql" and p.exists() else sql
    with connect(read_only=True) as con:
        rel = con.sql(text)
        if rel is None:
            console.print("[green]ok[/green] (no result set)")
            return
        df = rel.limit(limit).df()
        t = Table()
        for c in df.columns:
            t.add_column(str(c))
        for row in df.itertuples(index=False):
            t.add_row(*[str(v) for v in row])
        console.print(t)


@app.command("analyze")
def analyze_cmd(
    name: str = typer.Argument(
        "", help="Analytics script stem, e.g. abc_xyz_segmentation. Omit to list them."
    ),
    limit: int = typer.Option(60, help="Max rows to print."),
    csv: bool = typer.Option(False, "--csv", help="Write the full result to artifacts/ as CSV."),
) -> None:
    """Run a file from sql/analytics/ against the warehouse."""
    from sci.warehouse import ANALYTICS_DIR, connect

    scripts = {p.stem: p for p in sorted(ANALYTICS_DIR.glob("*.sql"))}
    if not name:
        t = Table(title="sql/analytics")
        t.add_column("name", style="cyan")
        for stem in scripts:
            t.add_row(stem)
        console.print(t)
        return
    if name not in scripts:
        console.print(f"[red]No such analysis '{name}'.[/red] Available: {', '.join(scripts)}")
        raise typer.Exit(code=1)

    with connect(read_only=True) as con:
        df = con.sql(scripts[name].read_text(encoding="utf-8")).df()

    if csv:
        settings.ensure_dirs()
        out = settings.artifacts_dir / f"{name}.csv"
        df.to_csv(out, index=False)
        console.print(f"[green]wrote[/green] {out} ({len(df):,} rows)")

    t = Table(title=f"{name}  ({len(df):,} rows)")
    for c in df.columns:
        t.add_column(str(c), overflow="fold")
    for row in df.head(limit).itertuples(index=False):
        t.add_row(*["" if v is None else str(v) for v in row])
    console.print(t)
    if len(df) > limit:
        console.print(f"[dim]... {len(df) - limit:,} more rows (use --limit or --csv)[/dim]")


def _report(counts: dict[str, int]) -> None:
    t = Table(title="staged")
    t.add_column("table", style="cyan")
    t.add_column("rows", justify="right")
    for k, v in counts.items():
        t.add_row(k, f"{v:,}")
    console.print(t)


if __name__ == "__main__":
    app()
