from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional, Dict, Any
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

DB_PATH    = os.getenv("DB_PATH", os.path.abspath("./local_db/forecasts.sqlite"))
DEFAULT_TZ = os.getenv("OUTPUT_TZ", "Europe/Helsinki")

# For remote update through the API
REMOTE_API_URL = os.getenv("REMOTE_API_URL", "http://localhost:8000/") # e.g. "https://my-tunnel.loca.lt/api/v1"
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY", "test1234")               # A secret key to authenticate API POST request
SYNC_REMOTE    = os.getenv("SYNC_REMOTE", "1") == "1"                  # Boolean

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

class SeriesMeta(BaseModel):
    series_key: str
    name: str
    unit: str
    region: str
    source: str
    description: str
    frequency: str

class BatchUpsertPayload(BaseModel):
    series: SeriesMeta
    history: List[TVPoint]
    forecast: List[TVPoint]
    metrics: Dict[str, Any] = {}
    history_prune_max_age: str = "1Y"
    forecast_horizon: Optional[int] = None

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

        self._init_db()

        # Upsert series row
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

    # Public methods
    def write_history(self, obj: pd.Series | pd.DataFrame, *, input_tz: str = "UTC") -> int:
        df = self._prepare(obj, input_tz=input_tz)
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
        created_at_utc: datetime | None = None,
        input_tz: str = "UTC",
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

        # Ensure run row (create if missing; update if exists)
        if run_id is None:
            run_id = str(uuid.uuid4())

        try:
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

            df = self._prepare(obj, input_tz=input_tz)
            self._bulk_upsert(df, kind="forecast", run_id=run_id)
        except Exception as e:
            print("[ERROR] Local database update failed: {e}")
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
        
        # Serialize to json
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

    # Internal methods
    def _init_db(self):
        self.metadata.create_all(self.engine)
        with self.engine.begin() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL;"))
            conn.execute(text("PRAGMA synchronous=NORMAL;"))
            conn.execute(text("PRAGMA foreign_keys=ON;"))

    def _prepare(self, obj: pd.Series | pd.DataFrame, *, input_tz: str) -> pd.DataFrame:
        # Normalize to DataFrame(timestamp,value)
        if isinstance(obj, pd.Series):
            if not isinstance(obj.index, pd.DatetimeIndex):
                raise ValueError("Series must have a DatetimeIndex.")
            df = obj.to_frame(name="value").reset_index(names="timestamp")
        elif isinstance(obj, pd.DataFrame):
            if {"timestamp", "value"}.issubset(obj.columns):
                df = obj[["timestamp", "value"]].copy()
            else:
                if obj.shape[1] < 2:
                    raise ValueError("DataFrame must have 'timestamp' and 'value' columns.")
                df = obj.iloc[:, :2].copy()
                df.columns = ["timestamp", "value"]
        else:
            raise TypeError("obj must be a pandas Series or DataFrame.")

        # Timestamps: localize to input_tz if naive, then convert to UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(input_tz)
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        # Sort & dedupe
        df.sort_values("timestamp", inplace=True)
        df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True, ignore_index=True)

        # Frequency validation
        expected = to_offset(self.props.frequency)
        if not df.empty:
            deltas = df["timestamp"].diff().dropna().dt.total_seconds().to_numpy()
            if deltas.size and not np.allclose(deltas, pd.to_timedelta(expected).total_seconds()):
                raise ValueError(f"Frequency validation failed: expected uniform '{self.props.frequency}'.")

        # Values validation + rounding
        s = pd.to_numeric(df["value"], errors="raise").astype(float)
        if np.isinf(s).any():
            raise ValueError("Values contain +/-inf.")
        if np.isnan(s).any():
            raise ValueError("Values contain NaNs.")
        df["value"] = s.round(2)

        return df

    def _bulk_upsert(self, df: pd.DataFrame, *, kind: str, run_id: Optional[str]) -> int:
        if kind not in ("history", "forecast"):
            raise ValueError("kind must be 'history' or 'forecast'.")
        if df.empty:
            return 0
        payload = [
            {
                "series_key": self.props.series_key,
                "kind": kind,
                "timestamp": ts.to_pydatetime(),
                "value": float(val),
                "run_id": run_id,
            }
            for ts, val in zip(df["timestamp"], df["value"])
        ]
        with self.engine.begin() as conn:
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
    
    def sync_remote_batch(
        self,
        meta: Dict[str, Any],
        series_key: str,
        run_id : str,
        # History and forecast as ["timestamp", "value"] DFs
        h_tv: pd.DataFrame,
        f_tv: pd.DataFrame,
        metrics: Mapping[str, float | int | None],
        freq: str,
        prune_age: str = "1Y",
    ) -> Optional[str]:
        """
        Push the same batch to a remote FastAPI/SQLite service, if configured.
        Returns remote run_id (if any) or None.
        """
        if not REMOTE_API_URL:
            print("[WARNING] No remote API URL defined")
            return None

        # Build payload compatible with FastAPI's BatchUpsertPayload
        series_meta = {
            "series_key": series_key,
            "name": meta["name"],
            "unit": meta["unit"],
            "region": meta["region"],
            "source": meta["source"],
            "description": meta["description"],
            "frequency": freq,
        }

        def to_list(tv_df : pd.DataFrame):
            return [{"timestamp" : ts.isoformat(), "value" : val} 
                    for ts, val in zip(tv_df["timestamp"], tv_df["value"])]

        # Conver dataframes to list of dictionary values for payload
        h_list = to_list(h_tv)
        f_list = to_list(f_tv)

        payload = {
            "series"            : series_meta,
            "history"           : h_list,
            "forecast "         : f_list,
            "metrics"           : dict(metrics or {}),
            "history_prune_age" : prune_age,
            "forecast_horizon"  : len(f_tv),
            "run_id"            : run_id
        }

        url = REMOTE_API_URL.rstrip("/") + "/db/batch_upsert"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": REMOTE_API_KEY or "",
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # If successful, return run_id from remote
            return data.get("run_id")
        except Exception as e:
            # Don't break local writes if remote is down; just log a warning.
            print(f"[WARNING] Remote sync failed for '{series_key}': {e}")
            return None

# Upsert history, forecasts and metrics to DB
def save_forecasts_to_db(
    forecasts: Dict[str, Dict[str, Any]],
    data_description: Dict[str, Dict[str, Any]],
    input_tz: str = "Europe/Helsinki",
    ) -> Dict[str, str]:
    """
    Persist a whole batch of forecast data and variables to SQLite.

    Args:
        forecasts (dict):
            Mapping of variable names to forecast data:

            ```
            {
                "<variable_name>": {
                    "y_train": pd.DataFrame,
                    "y_pred": pd.DataFrame,
                    "metrics": dict
                }
            }
            ```

        data_description (dict):
            Mapping of variable names to metadata:

            ```
            {
                "<variable_name>": {
                    "name": "<display name>",      # REQUIRED
                    "unit": "<unit>",              # REQUIRED
                    "region": "FI",                # REQUIRED (FI only)
                    "source": "<source>",          # REQUIRED
                    "description": "<text>",       # REQUIRED
                    # optional: "frequency": "1h"  # if omitted, inferred from indices
                }
            }
            ```

    Returns:
        run_IDs (dict): 
            Mapping of {series_key -> run_id} for the forecast writes.
    """
    out_run_ids: Dict[str, str] = {}

    for series_key, meta in data_description.items():
        data = forecasts.get(series_key, {})
        if not data:
            raise KeyError(f"Variable '{series_key}' missing in forecasts.")

        # Get and forecast data and possible forecast metricss
        history_df  : pd.DataFrame = data["y_train"]
        forecast_df : pd.DataFrame = data["y_pred"]
        metrics     : Mapping[str, float] = data.get("metrics", {})

        # Normalize to timestamp/value two-column frames
        def to_tv(df: pd.DataFrame) -> pd.DataFrame:
            # Expect single value column; if multiple, take first
            # Use index as timestamp, first column as value
            val = df.columns[0]
            tv = pd.DataFrame({"timestamp": df.index, "value": df[val].values})
            return tv

        h_tv = to_tv(history_df)
        f_tv = to_tv(forecast_df)

        # Infer frequency if not provided
        freq = meta.get("frequency")
        if not freq:
            # Try forecast first; fall back to history
            for idx in (forecast_df.index, history_df.index):
                if isinstance(idx, pd.DatetimeIndex):
                    freq = pd.infer_freq(idx)
                    if freq:
                        break
            if not freq:
                raise ValueError(f"Cannot infer frequency for '{series_key}'. Provide meta['frequency'].")
            
        freq = str(freq).lower()

        # Build series properties for the DB write
        props = SeriesProps(
            series_key=series_key,
            name=meta["name"],
            unit=meta["unit"],
            region=meta["region"],
            source=meta["source"],
            description=meta["description"],
            frequency=freq,  # e.g. "1h"
        )

        db = ForecastDB(db_path=DB_PATH, props=props)

        # History then forecast (forecast returns run_id)
        prune_age = "1Y"
        db.write_history(h_tv, input_tz=input_tz)   
        db.prune_history(prune_age)
        run_id = db.write_forecast(
            f_tv,
            input_tz=input_tz,
            metrics=metrics,
            forecast_horizon=len(forecast_df),
        )
        out_run_ids[series_key] = run_id

        if SYNC_REMOTE and run_id:
            db.sync_remote_batch(meta, 
                                 series_key, 
                                 run_id,
                                 h_tv,
                                 f_tv,
                                 metrics,
                                 freq,
                                 prune_age)

        print(f"[DONE] Persisted \"{meta['name']}\" forecast run results to: {DB_PATH}")

    return out_run_ids