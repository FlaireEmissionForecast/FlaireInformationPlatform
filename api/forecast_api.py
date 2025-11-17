from __future__ import annotations

import os
import json
from datetime import datetime, time, timedelta, timezone
from typing import Optional, Dict
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text

DB_PATH = os.getenv("DB_PATH", os.path.abspath("./db/forecasts.sqlite"))
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
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        if s.tzinfo is None: s = s.tz_localize(tz)
        if e.tzinfo is None: e = e.tz_localize(tz)
    else:
        today = datetime.now(tz).date()
        s = datetime.combine(today, time.min, tzinfo=tz)
        e = s + timedelta(days=2) - timedelta(seconds=1)
    return s.astimezone(timezone.utc), e.astimezone(timezone.utc), tz

def _iso_map(rows, tz: ZoneInfo) -> Dict[str, float]:
    out = {}
    for ts, val in rows:
        ts = datetime.fromisoformat(ts).astimezone(tz)
        out[ts.isoformat()] = round(float(val), 2)
    return out

@api.get("/forecast")
def forecast(
    series_key: str = Query(..., description="e.g. 'consumption_emissions'"),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    tz: str = Query(default=DEFAULT_TZ),
    run_id: str | None = Query(default=None),
):
    # Check that the database exists, inform user if it does not
    warn = _check_db_exists()
    if warn:
        return warn
    
    start_utc, end_utc, tzinfo = _parse_range(tz, start, end)

    with engine.begin() as conn:
        s = conn.execute(text("SELECT * FROM series WHERE series_key=:k"), {"k": series_key}).mappings().first()
        if not s:
            return {"error": f"series_key '{series_key}' not found"}

        if not run_id:
            rr = conn.execute(
                text("""SELECT run_id, created_at FROM runs WHERE series_key=:k ORDER BY created_at DESC LIMIT 1"""),
                {"k": series_key},
            ).mappings().first()
            run_id = rr["run_id"] if rr else None
            run_created_at = datetime.fromisoformat(rr["created_at"]) if rr else None
        else:
            rr = conn.execute(text("SELECT created_at FROM runs WHERE run_id=:r"), {"r": run_id}).mappings().first()
            run_created_at = datetime.fromisoformat(rr["created_at"]) if rr else None

        hist_rows = conn.execute(
            text("""
                SELECT timestamp, value FROM observations
                WHERE series_key=:k AND kind='history'
                AND timestamp BETWEEN :a AND :b
                ORDER BY timestamp
                """),
            {"k": series_key, "a": start_utc.isoformat(), "b": end_utc.isoformat()},
        ).all()

        if run_id:
            fc_rows = conn.execute(
                text("""
                    SELECT timestamp, value FROM observations
                    WHERE series_key=:k AND kind='forecast' AND run_id=:r
                    AND timestamp BETWEEN :a AND :b
                    ORDER BY timestamp
                    """),
                {"k": series_key, "r": run_id, "a": start_utc.isoformat(), "b": end_utc.isoformat()},
            ).all()
        else:
            fc_rows = conn.execute(
                text("""
                    SELECT timestamp, value FROM observations
                    WHERE series_key=:k AND kind='forecast'
                    AND timestamp BETWEEN :a AND :b
                    ORDER BY timestamp
                    """),
                {"k": series_key, "a": start_utc.isoformat(), "b": end_utc.isoformat()},
            ).all()

        metrics = {}
        if run_id:
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

    return {
        "metadata": {
            "name": s["name"],
            "unit": s["unit"],
            "region": s["region"],
            "source": s["source"],
            "description": s["description"],
            "frequency": s["frequency"],
            "generated_at": (run_created_at.astimezone(ZoneInfo(tz)).isoformat() if run_id and run_created_at else None),
        },
        "data": {
            "history" : _iso_map(hist_rows, tzinfo),
            "forecast": _iso_map(fc_rows,   tzinfo),
        },
        "metrics": metrics,
    }


@api.get("/")
def root():
    # Check that the database exists, inform user if it does not
    warn = _check_db_exists()
    if warn:
        return warn
    
    return {"message": "OK. Use /forecast?series_key=..."}

@api.get("/info")
def info():
    """
    Returns metadata about the available forecast series and database structure.
    """
    import sqlite3
    
    # Check that the database exists, inform user if it does not
    warn = _check_db_exists()
    if warn:
        return warn

    info_data = {"tables": [], "series": []}

    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        # List all tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        info_data["tables"] = [row["name"] for row in cur.fetchall()]

        # List all available series (if table exists)
        if "series" in info_data["tables"]:
            cur.execute("SELECT * FROM series ORDER BY series_key;")
            info_data["series"] = [dict(row) for row in cur.fetchall()]
            
        # 1️⃣ Get all user tables
        cur.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
        """)
        tables = [r[0] for r in cur.fetchall()]
        print(f"Tables found: {tables}\n")

        # 2️⃣ Show the contents of each table
        for table in tables:
            print(f"--- {table.upper()} ---")
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 10;", con)
            print(df)
            print()
            
    except sqlite3.Error as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"SQLite error: {str(e)}"},
        )
    finally:
        con.close()

    return info_data

