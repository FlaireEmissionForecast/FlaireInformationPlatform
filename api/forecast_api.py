from __future__ import annotations

import os
import json
from datetime import datetime, time, timedelta, timezone
from typing import Optional, Dict, List
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, Query, Depends, Header, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text, Connection

# Import class for managing local DB
from db_manager import (ForecastDB, 
                        SeriesProps, 
                        TVPoint, 
                        BatchUpsertPayload)

from dotenv import load_dotenv

# Load variables found in .env to current environment
load_dotenv()

DB_PATH    = os.getenv("DB_PATH", os.path.abspath("./db/forecasts.sqlite"))
DEFAULT_TZ = os.getenv("OUTPUT_TZ", "Europe/Helsinki")
API_KEY    = os.environ["FORECAST_API_KEY"] # A secret key to authenticate API POST request. Throws an error if not found

# Define path to test website
index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../index.html"))

api = FastAPI(title="Flip Forecast API", version="0.2.0")

# CORS only affects browsers, not direct requests from backend, scripts, etc.
api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://flaireemissionforecast.github.io",
        "http://localhost"
    ],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

engine = create_engine(f"sqlite:///{DB_PATH}", future=True, echo=False)

def _verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized request")

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

def _iso_map(rows, tz: ZoneInfo) -> Dict[str, float]:
    out = {}
    for ts, val in rows:
        # Parse UTC Naive DB timestamps as UTC aware
        ts = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        # Convert to requested timezone which shifts the times accordingly
        ts = ts.astimezone(tz)
        out[ts.isoformat()] = round(float(val), 2)
    return out

def _convert_local_ts_to_utc(ts_str : str, tzinfo : ZoneInfo) -> datetime:
    # Parse UTC Naive DB timestamps as UTC aware
    ts = datetime.fromisoformat(ts_str)
    if ts.tzinfo is None:
        # There should pretty much never be accompanying tzinfos
        ts = ts.replace(tzinfo=tzinfo)

    # Convert to database UTC timezone which shifts the times accordingly
    return ts.astimezone(timezone.utc)

def _convert_utc_ts_to_local(ts_str : str, tzinfo : ZoneInfo) -> datetime:
    # Parse UTC Naive DB timestamps as UTC aware
    ts = datetime.fromisoformat(ts_str)
    if ts.tzinfo is None:
        # There should pretty much never be accompanying tzinfos
        ts = ts.replace(tzinfo=timezone.utc)

    # Convert to database UTC timezone which shifts the times accordingly
    return ts.astimezone(tzinfo)

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
    """
    Interprets many offset strings as Timedeltas. Not calendar aware and year and month are approximate:
    - in (str): "1Y", "6M", "90D", "48H", "30min", etc.
    - out (Timedelta): 365 days, 180 days, 90 days, 48 hours, 30 min, etc.

    It is also possible to use ISO 8601 duration strings (e.g. \"P2W2D3H\").

    https://pandas.pydata.org/docs/user_guide/timedeltas.html#parsing
    """
    YEARS  = 365
    MONTHS = 30

    s = s.strip().upper()

    if s.startswith("P"):
        # Parse as ISO8601 duration string
        return pd.Timedelta(s)

    if s.endswith("Y"):
        return pd.Timedelta(days=int(s[:-1]) * YEARS)

    if s.endswith("M") and not s.endswith("MS"):
        return pd.Timedelta(days=int(s[:-1]) * MONTHS)

    return pd.Timedelta(s)

def _get_history_data_by_datetime_range(conn, series_key, history_start, forecast_start):
    # Fetch historical observations for the requested history window
    return conn.execute(
            text("""
                SELECT timestamp, value FROM observations
                WHERE series_key=:k AND kind='history'
                AND timestamp >= :a AND timestamp < :b
                ORDER BY timestamp
                """),
            {"k": series_key, "a": history_start, "b": forecast_start},
        ).all()

def _get_forecast_data_by_run_id(conn, series_key, run_id):
    # Fetch forecast observations for the selected run
    return conn.execute(
            text("""
                SELECT timestamp, value FROM observations
                WHERE series_key=:k AND kind='forecast' AND run_id=:r
                ORDER BY timestamp
                """),
            {"k": series_key, "r": run_id},
        ).all()

def _get_forecast_payload(series, run, hist_rows, fc_rows, metrics, tzinfo, run_id=None):
    """
    Build the standard forecast payload dict used by the endpoints.
    """

    # Convert timestamps from UTC to requested timezone
    forecast_start_tz = _convert_utc_ts_to_local(run["forecast_start"], tzinfo)
    generated_at_tz   = _convert_utc_ts_to_local(run["created_at"],     tzinfo)

    return {
        "metadata": {
            "name"           : series["name"],
            "unit"           : series["unit"],
            "region"         : series["region"],
            "source"         : series["source"],
            "description"    : series["description"],
            "frequency"      : series["frequency"],
            "run_id"         : run_id or run.get("run_id"),
            "forecast_start" : forecast_start_tz.replace(microsecond=0),
            "generated_at"   : generated_at_tz.replace(microsecond=0),
        },
        "data": {
            "history"  : _iso_map(hist_rows, tzinfo),
            "forecast" : _iso_map(fc_rows,   tzinfo),
        },
        "metrics" : metrics or {},
    }

############################
### API Endpoint Methods ###
############################
@api.get("/forecast")
def forecast(
    series_key  : str = Query(..., description="Always required, e.g. 'consumption_emissions'"),
    date        : str = Query(..., description="The date for which latest forecast is queried by creation date (e.g. '2025-06-15')"),
    history_len : str | None = Query(default="1W", description="Desired amount of history to query with the forecast in Pandas offset alias format (e.g. '1W' for one week)"),
    tz          : str = Query(default=DEFAULT_TZ,  description=f"Desired timezone of the query output, defaults to {DEFAULT_TZ}"),
    run_id      : str | None = Query(default=None, description="Forecast run UUID for requesting a specific run results")
    ):
    """
    Returns latest available forecast for a certain date with defined amount of historical data.
    """
    # Check that the database exists, inform user if it does not
    warn = _check_db_exists()
    if warn:
        return warn
    
    # Prepare timezone info
    tzinfo = ZoneInfo(tz)

    with engine.begin() as conn:
        series = conn.execute(text("SELECT * FROM series WHERE series_key=:k"), {"k": series_key}).mappings().first()
        if not series:
            return {"error": f"series_key '{series_key}' not found"}

        # Determine which run to use: explicit run_id > date cutoff
        if run_id:
            run = conn.execute(
                text("SELECT * FROM runs WHERE run_id=:r AND series_key=:k"),
                {"r": run_id, "k": series_key},
            ).mappings().first()
        elif date:
            # Conver timestamp ot database compatible format
            ts = _convert_local_ts_to_utc(date, tzinfo)

            # Use the whole day (00:00:00 .. < next day 00:00:00) in requested tz
            day = ts.date()
            day_start = datetime.combine(day, time.min, tzinfo=ZoneInfo(tz))
            day_end = day_start + timedelta(days=1)

            # Convert to UTC for DB comparison (DB stores timestamps in UTC ISO format)
            day_start_utc = day_start.astimezone(timezone.utc).isoformat()
            day_end_utc   = day_end.astimezone(timezone.utc).isoformat()

            run = conn.execute(
                text("""
                    SELECT *
                    FROM runs
                    WHERE series_key = :k
                      AND forecast_start >= :a
                      AND forecast_start <  :b
                    ORDER BY created_at DESC
                    LIMIT 1
                    """),
                {"k": series_key, "a": day_start_utc, "b": day_end_utc},
            ).mappings().first()

            if not run:
                return {"error": f"No forecast run found for date '{date}'"}

            run_id = run["run_id"]
        else:
            return {"error": "Either run_id or date parameter must be provided"}

        # Parse requested history length
        try:
            history_offset = _parse_simple_delta(history_len)
        except Exception as e:
            raise ValueError(f"Cannot parse history string '{history_len}' to Timedelta") from e

        # Calculate start datetimes for historical and forecast data
        forecast_start_utc = datetime.fromisoformat(run["forecast_start"])
        history_start_utc = forecast_start_utc - history_offset

        # Get historical and forecast data from the DB with date range and run ID
        hist_rows = _get_history_data_by_datetime_range(conn, series_key, history_start_utc, forecast_start_utc)        
        fc_rows   = _get_forecast_data_by_run_id(conn, series_key, run_id)

        metrics = {}
        if run_id:
            metrics = _get_metrics(run_id, conn)

    # Build forecast JSON payload
    payload = _get_forecast_payload(series, run, hist_rows, fc_rows, 
                                    metrics, tzinfo, run_id)

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
    
    # Prepare timezone info
    tzinfo = ZoneInfo(tz)

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

            # Set run ID
            run_id = run["run_id"]

        # Determine history start time based on requested history duration
        try:
            history_offset = _parse_simple_delta(history_len)
        except Exception as e:
            raise ValueError(f"Cannot parse history string '{history_len}' to Timedelta: {e}")

        # Calculate start datetimes for historical and forecast data
        forecast_start_utc = datetime.fromisoformat(run["forecast_start"])
        history_start_utc = forecast_start_utc - history_offset

        # Get historical and forecast data from the DB with date range and run ID
        hist_rows = _get_history_data_by_datetime_range(conn, series_key, history_start_utc, forecast_start_utc)        
        fc_rows   = _get_forecast_data_by_run_id(conn, series_key, run_id)

        metrics = {}
        if run_id:
            metrics = _get_metrics(run_id, conn)

    # Build forecast JSON payload
    payload = _get_forecast_payload(series, run, hist_rows, fc_rows, 
                                    metrics, tzinfo, run_id)

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

@api.get("/status")
def status():
    """
    Check that the database exists, inform user if it does not
    """
    warn = _check_db_exists()
    if warn:
        return warn
    
    return {"message" : f"Database OK, found from: {DB_PATH}"}

################################################
### Upsert incoming POST request to local DB ###
################################################
@api.post("/db/batch_upsert", 
          dependencies=[Depends(_verify_api_key)], 
          include_in_schema=False)
# Utilize pydantic data classes for data and type validation
def batch_upsert(payload: BatchUpsertPayload):
    # Recreate SeriesProps
    props = SeriesProps(
        series_key=payload.series.series_key,
        name=payload.series.name,
        unit=payload.series.unit,
        region=payload.series.region,
        source=payload.series.source,
        description=payload.series.description,
        frequency=payload.series.frequency,
    )

    db = ForecastDB(db_path=DB_PATH, props=props)

    # Recreate DataFrames
    def to_df(points: List[TVPoint]) -> pd.DataFrame:
        if not points:
            return pd.DataFrame(columns=["timestamp", "value"])
        return pd.DataFrame(
            {"timestamp" : [p.timestamp for p in points],
             "value"     : [p.value for p in points]}
        )

    history_df  = to_df(payload.history)
    forecast_df = to_df(payload.forecast)

    # Upsert history + prune + forecast + metrics
    if not history_df.empty:
        db.write_history(history_df, input_tz="UTC")
        db.prune_history(payload.properties.get("history_prune_age", "1Y"))

    if not forecast_df.empty:
        run_id = db.write_forecast(
            forecast_df,
            payload.properties.get("run_id", None),
            forecast_horizon=payload.properties.get("forecast_horizon", len(forecast_df)),
            metrics=payload.metrics or {}
        )

    return {"status": "ok", "run_id": run_id}

# Serve website from root
@api.get("/")
def serve_website():
    return FileResponse(index_path)

# For browsers
@api.get("/favicon.ico")
async def favicon():
    return FileResponse("static/icons8-database-view-cute-color-96.png")

if __name__ == "__main__":
    # Use test client for offline debugging, requires httpx package
    from fastapi.testclient import TestClient

    # Set path to test database
    DB_PATH = os.getenv("DB_PATH", os.path.abspath("./test_db/forecasts.sqlite"))

    # Set test API key
    API_KEY = "test1234"

    client = TestClient(api)

    def test_batch_upsert_direct(strip_run_id = False):
        with open('test_data/test_payload.json', "r") as data:
            json_payload = dict(json.load(data))

        # Remove run ID from payload to se if batch upsert still functions as intended i.e. creates a run ID
        if strip_run_id:
            json_payload['properties'].update({"run_id" : ""})

        # Convert to pydantic data model manually. FastAPI does this automatically if POST endpoint is used
        payload = BatchUpsertPayload.model_validate(json_payload)

        # Call upsert method directly
        result = batch_upsert(payload)

        # Check if result OK
        assert result["status"] == "ok" and result["run_id"] is not None

    def test_batch_upsert_post():
        with open('test_data/test_payload.json', "r") as data:
            json_payload = json.load(data)

        # Call POST method for upsert, send JSON body and include API key header expected by the endpoint
        response = client.post("/db/batch_upsert", json=json_payload, headers={"Content-Type": "application/json",
                                                                               "x-api-key": API_KEY})
        assert response.status_code == 200

    def test_latest_forecast():
        response = client.get("/forecast/latest", params={"series_key": "consumption_emissions"})
        assert response.status_code == 200

    def test_forecast():
        response = client.get("/forecast/", params={"series_key" : "consumption_emissions",
                                                    "date"       : "2025-11-04"})
        assert response.status_code == 200

    try:
        test_batch_upsert_direct(False)
        test_batch_upsert_direct(True)
        test_batch_upsert_post()
        test_forecast()
        test_latest_forecast()
    except Exception as e:
        raise RuntimeError(f"One or more database API tests failed") from e
    finally:
        # Remove created test database
        import glob

        for p in glob.glob(f'{DB_PATH}*'):
            if os.path.isfile(p):
                os.remove(p)

        # Remove test DB dir
        print(DB_PATH.split("/")[-1])
        os.removedirs(DB_PATH.removesuffix(DB_PATH.split("/")[-1]))

    print("[DONE] All database and API tests passed successfully!")