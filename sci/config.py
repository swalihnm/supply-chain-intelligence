"""Central configuration and filesystem layout.

Every module resolves paths through `settings` rather than hardcoding them, so
the bulk data directory can sit outside the synced repo folder.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage -----------------------------------------------------------
    data_dir: Path = Field(default=REPO_ROOT / "data", alias="SCI_DATA_DIR")

    # --- sources -----------------------------------------------------------
    m5_source: str = Field(default="", alias="SCI_M5_SOURCE")
    fred_api_key: str = Field(default="", alias="FRED_API_KEY")

    # --- engine ------------------------------------------------------------
    # DuckDB defaults to ~80% of system RAM and then FAILS rather than spilling
    # when that estimate is wrong. An explicit, modest limit plus a temp
    # directory makes the heavy queries spill to disk instead. Raise it on a
    # bigger machine; 2 GB is chosen to leave room for everything else.
    duckdb_memory_limit: str = Field(default="2GB", alias="SCI_DUCKDB_MEMORY_LIMIT")

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand(cls, v: str | Path) -> Path:
        return Path(str(v)).expanduser()

    # --- derived layout ----------------------------------------------------
    # raw/      bytes exactly as the source gave them (audit trail)
    # staging/  typed parquet, one directory per source table
    # warehouse/ the DuckDB star schema
    # features/ point-in-time-correct feature parquet
    # artifacts/ models, backtest results, figures
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def staging_dir(self) -> Path:
        return self.data_dir / "staging"

    @property
    def warehouse_dir(self) -> Path:
        return self.data_dir / "warehouse"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def duckdb_path(self) -> Path:
        return self.warehouse_dir / "sci.duckdb"

    @property
    def duckdb_temp_dir(self) -> Path:
        """Where DuckDB spills when a query exceeds the memory limit."""
        return self.data_dir / "tmp"

    def ensure_dirs(self) -> None:
        for p in (
            self.raw_dir,
            self.staging_dir,
            self.warehouse_dir,
            self.features_dir,
            self.artifacts_dir,
            self.duckdb_temp_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
