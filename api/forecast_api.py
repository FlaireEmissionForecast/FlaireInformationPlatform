from __future__ import annotations

import os
import json
from datetime import datetime, time, timedelta, timezone
from typing import Optional, Dict
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text, Connection

DB_PATH    = os.getenv("DB_PATH", os.path.abspath("./db/forecasts.sqlite"))
DEFAULT_TZ = os.getenv("OUTPUT_TZ", "Europe/Helsinki")

api = FastAPI(title="Flip Forecast API", version="0.1.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://flaireemissionforecast.github.io",
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:5173",
    ],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

engine = create_engine(f"sqlite:///{DB_PATH}", future=True, echo=False)

def _check_db_exists():
    """Check whether the database file exists before any query."""
    if not os.path.exists(DB_PATH):
        return JSONResponse(
            status_code=503,
            content={
                "error": "Forecast database not found.",
                "message": (
                    "The forecast database does not exist yet. "
                    "It is normally created by the model when saving results. "
                    "Please run the forecast model or data writer first."
                ),
            },
        )
    return None

def _parse_range(tz_str: str, start: Optional[str], end: Optional[str]) -> tuple[datetime, datetime, ZoneInfo]:
    tz = ZoneInfo(tz_str)
    if start and end:
        start = pd.to_datetime(start)
        end   = pd.to_datetime(end)
        if start.tzinfo is None: start = start.tz_localize(tz)
        if end.tzinfo is None:   end   = end.tz_localize(tz)
    else:
        today = datetime.now(tz).date()
        start = datetime.combine(today, time.min, tzinfo=tz)
        end   = start + timedelta(days=2) - timedelta(seconds=1)
    
    start = start.astimezone(timezone.utc).isoformat()
    end   = end.astimezone(timezone.utc).isoformat()
    return start, end, tz

def _iso_map(rows, tz: ZoneInfo) -> Dict[str, float]:
    out = {}
    for ts, val in rows:
        ts = datetime.fromisoformat(ts).astimezone(tz)
        out[ts.isoformat()] = round(float(val), 2)
    return out

def _get_metrics(run_id : str, conn : Connection):
    m = conn.execute(text("SELECT metrics_json FROM metrics WHERE run_id=:r"), {"r": run_id}).mappings().first()
    metrics = {}
    if m:
        val = m["metrics_json"]
        # JSON data is stored as str in SQLite
        if isinstance(val, str):
            try:
                metrics = json.loads(val)
            except json.JSONDecodeError:
                metrics = {}
        # Other DBs such as PostgreSQL and MySQL support JSON natively
        elif isinstance(val, dict):
            metrics = val

    return metrics

def _parse_simple_delta(s : str):
    YEARS  = 365
    MONTHS = 30

    s = s.strip().upper()

    if s.endswith("Y"):
        return pd.Timedelta(days=int(s[:-1]) * YEARS)

    if s.endswith("M") and not s.endswith("MS"):
        return pd.Timedelta(days=int(s[:-1]) * MONTHS)

    return pd.Timedelta(s)

############################
### API Endpoint Methods ###
############################
@api.get("/forecast")
def forecast(
    series_key: str = Query(..., description="Always required, e.g. 'consumption_emissions'"),
    start     : str | None = Query(default=None, description="Start date of query in an ISO compatible format (e.g. YYYY-MM-DDTHH:MM)"),
    end       : str | None = Query(default=None, description="End date of query in an ISO compatible format (e.g. YYYY-MM-DDTHH:MM)"),
    tz        : str = Query(default=DEFAULT_TZ,  description=f"Desired timezone of the query output, defaults to {DEFAULT_TZ}"),
    run_id    : str | None = Query(default=None, description="Forecast run UUID for requesting a specific run results")
    ):
    # Check that the database exists, inform user if it does not
    warn = _check_db_exists()
    if warn:
        return warn
    
    start_utc, end_utc, tzinfo = _parse_range(tz, start, end)

    with engine.begin() as conn:
        series = conn.execute(text("SELECT * FROM series WHERE series_key=:k"), {"k": series_key}).mappings().first()
        if not series:
            return {"error": f"series_key '{series_key}' not found"}

        if not run_id:
            run = conn.execute(
                text("""SELECT * FROM runs WHERE series_key=:k ORDER BY created_at DESC LIMIT 1"""),
                {"k": series_key},
            ).mappings().first()
            run_id = run["run_id"] if run else None
        else:
            run = conn.execute(text("SELECT * FROM runs WHERE run_id=:r"), {"r": run_id}).mappings().first()

        hist_rows = conn.execute(
            text("""
                SELECT timestamp, value FROM observations
                WHERE series_key=:k AND kind='history'
                AND timestamp BETWEEN :a AND :b
                ORDER BY timestamp
                """),
            {"k": series_key, "a": start_utc, "b": end_utc},
        ).all()

        if run_id:
            fc_rows = conn.execute(
                text("""
                    SELECT timestamp, value FROM observations
                    WHERE series_key=:k AND kind='forecast' AND run_id=:r
                    AND timestamp BETWEEN :a AND :b
                    ORDER BY timestamp
                    """),
                {"k": series_key, "r": run_id, "a": start_utc, "b": end_utc},
            ).all()
        else:
            fc_rows = conn.execute(
                text("""
                    SELECT timestamp, value FROM observations
                    WHERE series_key=:k AND kind='forecast'
                    AND timestamp BETWEEN :a AND :b
                    ORDER BY timestamp
                    """),
                {"k": series_key, "a": start_utc, "b": end_utc},
            ).all()

        metrics = {}
        if run_id:
            metrics = _get_metrics(run_id, conn)

    payload = {
        "metadata": {
            "name": series["name"],
            "unit": series["unit"],
            "region": series["region"],
            "source": series["source"],
            "description": series["description"],
            "frequency": series["frequency"],
            "run_id": run_id,
            "forecast_start": run["forecast_start"],
            "generated_at": run["created_at"],
        },
        "data": {
            "history" : _iso_map(hist_rows, tzinfo),
            "forecast": _iso_map(fc_rows,   tzinfo),
        },
        "metrics": metrics,
    }

    return payload

@api.get("/forecast/latest")
def latest_forecast(
    series_key : str = Query(..., description="Always required, e.g. 'consumption_emissions'"),
    history_len: str | None = Query(default="1W", description="Desired amount of history to query with the forecast in Pandas offset alias format (e.g. '1W' for one week)"),
    tz         : str = Query(default=DEFAULT_TZ,  description=f"Desired timezone of the query output, defaults to {DEFAULT_TZ}"),
    run_id     : str | None = Query(default=None, description="Forecast run UUID for requesting a specific run results")
):
    """
    Returns latest available forecast with defined amount of historical data.
    """

    # Check that the database exists, inform user if it does not
    warn = _check_db_exists()
    if warn:
        return warn

    # Find start date for the query i.e. the start date of the latest forecast run series
    with engine.begin() as conn:
        series = conn.execute(text("SELECT * FROM series WHERE series_key=:k"), {"k": series_key}).mappings().first()
        if not series:
            return {"error": f"series_key '{series_key}' not found"}
        
        # Fetch the latest run to determine forecast start time
        if run_id:
            run = conn.execute(
                text("""
                    SELECT *
                    FROM runs
                    WHERE run_id = :r AND series_key = :k
                    """),
                {"r": run_id, "k": series_key},
            ).mappings().first()
        else:
            run = conn.execute(
                text("""
                    SELECT *
                    FROM runs
                    WHERE series_key = :k
                    ORDER BY forecast_start DESC
                    LIMIT 1
                    """),
                {"k": series_key},
            ).mappings().first()

        # Fetch forecast data for the identified run
        fc_rows = conn.execute(
            text("""
                SELECT timestamp, value
                FROM observations
                WHERE series_key = :k
                AND kind = 'forecast'
                AND run_id = :r
                ORDER BY timestamp
                """),
            {"k": series_key, "r": run["run_id"]},
        ).all()

        # Determine history start time based on requested history duration
        try:
            # Interprets many offset strings as Timedeltas, not calendar aware:
            # - "1Y", "6M", "90D", "48H", "30min", etc. as Timedelta
            history_offset = _parse_simple_delta(history_len)
        except Exception as e:
            raise ValueError(f"Cannot parse history string '{history_len}' to Timedelta: {e}")

        forecast_start_utc = datetime.fromisoformat(run["forecast_start"])
        history_start_utc = forecast_start_utc - history_offset

        hist_rows = conn.execute(
            text("""
                SELECT timestamp, value
                FROM observations
                WHERE series_key = :k
                  AND kind = 'history'
                  AND timestamp >= :hs
                  AND timestamp <  :fs
                ORDER BY timestamp
                """
                ),
                {"k": series_key, "hs": history_start_utc, "fs": forecast_start_utc}
        ).all()

        metrics = {}
        if run["run_id"]:
            metrics = _get_metrics(run["run_id"], conn)


    payload = {
        "metadata": {
            "name": series["name"],
            "unit": series["unit"],
            "region": series["region"],
            "source": series["source"],
            "description": series["description"],
            "frequency": series["frequency"],
            "run_id": run["run_id"],
            "forecast_start": run["forecast_start"],
            "generated_at": run["created_at"],
        },
        "data": {
            "history" : _iso_map(hist_rows, ZoneInfo(tz)),
            "forecast": _iso_map(fc_rows,   ZoneInfo(tz)),
        },
        "metrics": metrics,
    }

    return payload

@api.get("/history")
def history():
    """
    Returns historical observational data if available in the requested period.
    """
    return {"message" : "Endpoint for historical data not yet available"}

@api.get("/info")
def info():
    """
    Returns metadata about the available forecast series and database structure.
    """
    warn = _check_db_exists()
    if warn:
        return warn
    with engine.begin() as conn:
        # Get all available series names
        series_list = conn.execute(text("SELECT * FROM series")).mappings().all()

        # Get number of runs per series
        runs_list = []
        hist_list = []
        for s in series_list:
            run_count = conn.execute(
                text("SELECT COUNT(*) AS run_count FROM runs WHERE series_key=:k"),
                {"k": s["series_key"]},
                ).mappings().first()
            runs_list.append({
                "series_key": s["series_key"],
                "run_count" : run_count["run_count"] if run_count else 0,
            })

            # Get historical data availability info per series
            hist_info = conn.execute(
                text("""
                    SELECT 
                        MIN(timestamp) AS start_time,
                        MAX(timestamp) AS end_time,
                        COUNT(*) AS observation_count
                    FROM observations
                    WHERE series_key=:k AND kind='history'
                    """),
                {"k": s["series_key"]},
            ).mappings().first()
            hist_list.append({
                "series_key"    : s["series_key"],
                "history_start" : hist_info["start_time"],
                "history_end"   : hist_info["end_time"],
                "history_count" : hist_info["observation_count"],
            })

    payload = {
        "message": "Forecast database information",
        "series" : series_list,
        "runs"   : runs_list,
        "history": hist_list,
        }

    return payload

@api.get("/")
def root():
    """
    Check that the database exists, inform user if it does not
    """
    warn = _check_db_exists()
    if warn:
        return warn
    
    return {"message" : f"Database OK, found from: {DB_PATH}"}

# For browsers
@api.get("/favicon.ico")
async def favicon():
    return FileResponse("static/icons8-database-view-cute-color-96.png")

if __name__ == "__main__":
    # Use test client for offline debugging, required httpx package
    from fastapi.testclient import TestClient

    client = TestClient(api)

    def test_latest_forecast():
        response = client.get("/forecast/latest", params={"series_key": "consumption_emissions"})
        assert response.status_code == 200

    test_latest_forecast()