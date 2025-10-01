from fastapi import FastAPI
from fastapi.responses import FileResponse
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import os

# Instantiate API
api = FastAPI()

# Absolute path to the database file and index page
db_path    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../db_data/forecast.sqlite"))
index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../index.html"))

# Create the SQLite database engine and define path to the DB
engine = create_engine(f"sqlite:///{db_path}", echo=False)

# Example: generate 7 days of dummy forecasts
df = pd.DataFrame({
    "timestamp": pd.date_range(datetime.now().date(), periods=7*24, freq="1h"),
    "value": pd.Series(range(7*24)).apply(lambda x: (x % 24)).tolist()
})

# Write records stored in a DataFrame to a SQL database (which are supported by SQLAlchemy)
df.to_sql("forecast", engine, if_exists="replace", index=False)

# API endpoint: return forecasts
@api.get("/forecast")
def get_forecast(start: str = "", end: str = ""):
    """
    Return forecast data as JSON.
    - Default: today 00:00 → tomorrow 23:59 (local time).
    - Optional query params: start=YYYY-MM-DD, end=YYYY-MM-DD
    """
    # Default: current day + next day
    if not start or not end:
        today = datetime.now().date()
        start_dt = datetime.combine(today, datetime.min.time())
        end_dt = start_dt + timedelta(days=2) - timedelta(seconds=1)  # up to tomorrow 23:59
    else:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

    query = f"""
        SELECT * FROM forecast
        WHERE timestamp BETWEEN '{start_dt}' AND '{end_dt}'
    """
    df = pd.read_sql(query, engine)

    return df.to_dict(orient="records")

# Serve frontend page
@api.get("/")
def serve_frontend():
    return FileResponse(index_path)
