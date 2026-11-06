from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

class SimpleCache:
    def __init__(
        self,
        filepath   : str,
        value_col  : str,
        history    : str = '2W',
        timezone   : str = "Europe/Helsinki",
        query_freq : str = '1h'
    ):
        self.filepath      = Path(filepath, f"entsoe_{value_col}.csv")
        self.value_col     = value_col
        self.history_hours = int(pd.Timedelta(history).total_seconds() / 3600)
        self.timezone      = ZoneInfo(timezone)
        self.query_freq    = query_freq

        if self.history_hours <= 0:
            raise ValueError("[CACHE] History_hours must be positive")

    def _to_utc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate dataframe and normalize its index to UTC."""

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Dataframe must have a DatetimeIndex")

        if df.index.tz is None:
            raise ValueError("DatetimeIndex must be timezone-aware")

        if self.value_col not in df.columns:
            raise ValueError(f"Missing column: {self.value_col}")

        df = df[[self.value_col]].copy()
        df.index = df.index.tz_convert("UTC")
        df.index.name = "datetime"

        return df.sort_index()

    def _trim(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove entries older than the configured history."""

        if df.empty:
            return df

        cutoff = (df.index.max() - pd.Timedelta(hours=self.history_hours))

        return df[df.index >= cutoff]

    def _read(self) -> pd.DataFrame:
        """Read CSV and return a UTC-indexed dataframe."""

        # Return empty DF if cache file does not exist
        if not self.filepath.exists():
            return pd.DataFrame(
                columns=[self.value_col],
                index=pd.DatetimeIndex([], tz="UTC", name="datetime")
            )

        df = pd.read_csv(self.filepath)

        # Set datetime column as the index
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime")

        return df[[self.value_col]].sort_index()

    def _write(self, df: pd.DataFrame):
        """Write UTC-indexed dataframe to CSV."""

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            self.filepath,
            index_label="datetime",
            date_format="%Y-%m-%dT%H:%M:%S%z"
        )

    def load(self) -> pd.DataFrame:
        """Load cache and return the queried amount of data with local timestamps."""

        # Read the cached data
        df = self._read()

        return df.tz_convert(self.timezone)

    def save(self, df: pd.DataFrame):
        """Save new data, merging with existing cache if present."""

        new = self._to_utc(df)

        if self.filepath.exists():
            old = self._read()
            df  = pd.concat([old, new])
        else:
            df = new

        # New values take precedence over existing values.
        df = df[~df.index.duplicated(keep="last")]

        # Enforce retention and chronological ordering.
        df = self._trim(df.sort_index())

        self._write(df)

    def has_data(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """Check whether cache contains all timestamps between [start, end)."""

        if not self.filepath.exists():
            return False

        df = self._read()

        if df.empty:
            return False

        # Normalize to UTC to match cache
        start = pd.Timestamp(start).tz_convert("UTC")
        end   = pd.Timestamp(end).tz_convert("UTC")

        # Create expected index for the requested range
        expected = pd.date_range(
            start     = start,
            end       = end,
            freq      = self.query_freq,
            inclusive = "left", # Right exclusive to match ENTSO-E API semantics
        )

        # Drop any NaN values
        cached = df[self.value_col].dropna().index

        # Check that all expected datetimes are present in cache index
        return expected.isin(cached).all()