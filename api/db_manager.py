from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional, Dict, List, Any
from pydantic import BaseModel
import math
import os
import uuid
import json

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from sqlalchemy import (
    create_engine, MetaData, Table, Column, String, Integer, DateTime, Float,
    ForeignKey, JSON, text, UniqueConstraint, Index
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
import requests

# Series properties (no defaults; enforced)
from dataclasses import dataclass

DB_PATH     = os.getenv("DB_PATH", os.path.abspath("./local_db/forecasts.sqlite"))
DEFAULT_TZ  = os.getenv("OUTPUT_TZ", "Europe/Helsinki")

@dataclass(frozen=True)
class SeriesProps:
    series_key: str        # E.g. "consumption_emissions"
    name: str              # E.g. "Consumption emissions"
    unit: str              # E.g. "gCO2eq"
    region: str            # E.g. "FI"
    source: str            # E.g. "Forecast model"
    description: str       # Brief data description that could be displayed in the front-end
    frequency: str         # Pandas offset alias, e.g. "1h"

class TVPoint(BaseModel):
    timestamp: datetime
    value: float

class BatchUpsertPayload(BaseModel):
    series: SeriesProps
    properties: Dict[str, Any] = {}
    history: List[TVPoint]
    forecast: List[TVPoint]
    metrics: Dict[str, Any] = {}

class ForecastDB:
    """
    - schema creation (WAL, indexes)
    - strict validation (uniform freq, no NaN/inf)
    - UTC storage, round to 2 decimals
    - idempotent upserts (latest wins)
    - per-run metrics JSON
    """
    def __init__(self, db_path: str, props: SeriesProps):
        if not all(getattr(props, f) for f in vars(props)):
            raise ValueError("All SeriesProps fields must be provided (no defaults).")

        self.props = props

        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.engine: Engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}", future=True, echo=False)
        self.metadata = MetaData()

        self.series = Table(
            "series", self.metadata,
            Column("series_key", String, primary_key=True),
            Column("name", String, nullable=False),
            Column("unit", String, nullable=False),
            Column("region", String, nullable=False),
            Column("source", String, nullable=False),
            Column("description", String, nullable=False),
            Column("frequency", String, nullable=False),
        )

        # TODO: Lexicographical datetime comparisons are really unreliable!
        #       It would be better to use Unix epoch time (in seconds) for the timestamps
        #       As a bonus BETWEEN queries should be faster as integer comparisons
        self.runs = Table(
            "runs", self.metadata,
            Column("run_id", String, primary_key=True),
            Column("series_key", String, ForeignKey("series.series_key"), nullable=False),
            Column("forecast_horizon", Integer, nullable=True),
            Column("forecast_start", DateTime(timezone=True), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),  # UTC
        )
        Index("ix_runs_series_created", self.runs.c.series_key, self.runs.c.created_at.desc())

        self.metrics = Table(
            "metrics", self.metadata,
            Column("run_id", String, ForeignKey("runs.run_id"), primary_key=True),
            Column("metrics_json", JSON().with_variant(String, "sqlite"), nullable=False),
        )

        self.observations = Table(
            "observations", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("series_key", String, ForeignKey("series.series_key"), nullable=False),
            Column("kind", String, nullable=False),  # 'history' | 'forecast'
            Column("timestamp", DateTime(timezone=True), nullable=False),  # UTC
            Column("value", Float, nullable=False),
            Column("run_id", String, ForeignKey("runs.run_id"), nullable=True),
            UniqueConstraint("series_key", "kind", "timestamp", name="uq_series_kind_ts"),
            Index("ix_obs_series_ts", "series_key", "timestamp"),
        )

        # Init database schema
        self._init_db()

        # Upsert series defined series properties
        with self.engine.begin() as conn:
            stmt = sqlite_insert(self.series).values(
                series_key=props.series_key,
                name=props.name,
                unit=props.unit,
                region=props.region,
                source=props.source,
                description=props.description,
                frequency=props.frequency,
            ).on_conflict_do_update(
                index_elements=[self.series.c.series_key],
                set_={
                    "name": props.name,
                    "unit": props.unit,
                    "region": props.region,
                    "source": props.source,
                    "description": props.description,
                    "frequency": props.frequency,
                }
            )
            conn.execute(stmt)

    # Internal methods
    def _init_db(self):
        # Create the database schema (no-op if tables already exist)
        self.metadata.create_all(self.engine)

        with self.engine.begin() as conn:
            # Enable WAL mode:
            #   - Improves write concurrency
            #   - Allows readers to proceed without being blocked by writers
            #   - Better durability than the default rollback journal
            conn.execute(text("PRAGMA journal_mode=WAL;"))

            # Set WAL-appropriate sync mode:
            #   - NORMAL still syncs the WAL but skips the extra F_FULLFSYNC
            #   - Significantly faster while retaining crash-safe behavior for WAL
            conn.execute(text("PRAGMA synchronous=NORMAL;"))

            # Turn on actual FK enforcement:
            #   - SQLite defines foreign keys in the schema but does not enforce them
            #     unless this pragma is enabled
            #   - Ensures no orphaned (runs which have a non-existent series_key) runs/observations can be inserted
            conn.execute(text("PRAGMA foreign_keys=ON;"))


    def _bulk_upsert(self, df: pd.DataFrame, kind: str, run_id: Optional[str]) -> int:
        if kind not in ("history", "forecast"):
            raise ValueError("kind must be 'history' or 'forecast'")
        
        if any(ts.tzinfo is None for ts in df["timestamp"]):
            raise ValueError("All timestamps must be timezone-aware for UTC conversion")
        
        if df.empty:
            print("[NOTE] No data to write to database")
            return 0
        
        payload = [
            {
                "series_key" : self.props.series_key,
                "kind"       : kind,
                "timestamp"  : ts.astimezone(timezone.utc),
                "value"      : float(val),
                "run_id"     : run_id
            }
            for ts, val in zip(df["timestamp"], df["value"])
        ]

        with self.engine.begin() as conn:
            # Transaction is committed, and Connection is released to the connection pool automatically by the begin() context manager
            ins = sqlite_insert(self.observations).values(payload)
            upd = ins.on_conflict_do_update(
                index_elements=["series_key", "kind", "timestamp"],
                set_={"value": ins.excluded.value, "run_id": ins.excluded.run_id},
            )
            conn.execute(upd)
        return len(payload)
    
    def _parse_simple_delta(self, s : str):
        YEARS  = 365
        MONTHS = 30

        s = s.strip().upper()

        if s.endswith("Y"):
            return pd.Timedelta(days=int(s[:-1]) * YEARS)

        if s.endswith("M") and not s.endswith("MS"):
            return pd.Timedelta(days=int(s[:-1]) * MONTHS)

        return pd.Timedelta(s)

    # Public methods
    def write_history(self, df: pd.Series | pd.DataFrame, input_tz: str = "UTC") -> int:
        return self._bulk_upsert(df, kind="history", run_id=None)
    
    def prune_history(self, max_age: str = "52W") -> int:
            """
            Delete historical observations older than the specified age.
            Uses Pandas Timedelta parsing with Year and month extensions for flexible durations.

            Applies to ALL series.

            Parameters
            ----------
            max_age : str
                Examples:
                "1Y", "6M", "2W", "48H", etc.

            Returns
            -------
            int : number of deleted rows
            """

            # Use Timedelta parser for max age, not calendar-aware
            try:
                offset = self._parse_simple_delta(max_age)
            except Exception as e:
                raise ValueError(f"Cannot parse max_age string '{max_age}' to Timedelta: {e}")

            # Compute cutoff time in UTC
            now_utc   = pd.Timestamp.utcnow()
            cutoff_ts = now_utc - offset

            # Conver from Pandas Timestamp to datetime and then ISO date string
            cutoff = cutoff_ts.to_pydatetime().replace(tzinfo=timezone.utc).isoformat()

            # Delete historical observations before the cutoff
            delete_stmt = text("""
                DELETE FROM observations
                WHERE kind = 'history'
                AND timestamp < :cutoff
                """)

            # Open connection to DB with engine and execute query
            with self.engine.begin() as conn:
                result  = conn.execute(delete_stmt, {"cutoff": cutoff})
                deleted = result.rowcount or 0

            return deleted

    def write_forecast(
        self,
        obj: pd.Series | pd.DataFrame,
        run_id: str | None = None,
        metrics: Mapping[str, float | int | None] | None = None,
        forecast_horizon: int | None = None,
        created_at_utc: datetime | None = None
        # input_tz: str = "UTC" # NOTE: Not used ATM, as everything is assumed as UTC inside the DB
    ) -> str:
        """Upsert forecast rows (latest wins) and optional metrics in one call; returns run_id."""

        # Get forecast creation and time series start dates as UTC
        created_at_utc = created_at_utc or datetime.now(timezone.utc)
        if isinstance(obj, pd.Series):
            forecast_start_utc = obj.index.min().to_pydatetime().astimezone(timezone.utc)
        elif isinstance(obj, pd.DataFrame):
            forecast_start_utc = obj["timestamp"].min().to_pydatetime().astimezone(timezone.utc)
        else:
            raise RuntimeError("Data object is not a Pandas series or DataFrame")

        # Ensure run row has a unique identifier
        if run_id is None:
            run_id = str(uuid.uuid4())

        try:
            # Write forecast payload from forecast model
            with self.engine.begin() as conn:
                stmt = sqlite_insert(self.runs).values(
                    run_id=run_id,
                    series_key=self.props.series_key,
                    forecast_horizon=forecast_horizon,
                    forecast_start=forecast_start_utc,
                    created_at=created_at_utc,
                ).on_conflict_do_update(
                    index_elements=[self.runs.c.run_id],
                    set_={
                        "series_key": self.props.series_key,
                        "forecast_horizon": forecast_horizon,
                        "forecast_start": forecast_start_utc,
                        "created_at": created_at_utc,
                    }
                )
                conn.execute(stmt)

            self._bulk_upsert(obj, kind="forecast", run_id=run_id)
        except Exception as e:
            print(f"[ERROR] Local database update failed: {e}")
            return None

        if metrics is not None:
            self.write_metrics(run_id, metrics)

        return run_id

    def write_metrics(self, run_id: str, metrics: Mapping[str, float | int | None]) -> None:
        def clean(v):
            if v is None:
                return None
            if isinstance(v, (float, int)):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    raise ValueError("Metrics contain NaN/inf.")
                return float(v)
            return v

        # Sanitize metrics
        metrics_clean = {str(k): clean(v) for k, v in dict(metrics).items()}
        
        # Serialize to a json formatted string
        metrics_json_str = json.dumps(metrics_clean)

        with self.engine.begin() as conn:
            stmt = sqlite_insert(self.metrics).values(
                run_id=run_id,
                metrics_json=metrics_json_str
            ).on_conflict_do_update(
                index_elements=[self.metrics.c.run_id],
                set_={"metrics_json": metrics_json_str}
            )
            conn.execute(stmt)