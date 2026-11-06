### Common utility methods for all APIs ###
import os
import pytz
import pandas as pd

def get_env_var(key : str):
    """
    Retrieves the value of the specified environment variable.
    Args:
        key (str): The name of the environment variable to retrieve.
    Returns:
        str: The value of the specified environment variable.
    Raises:
        SystemExit: If the environment variable is not set, prints an error message and exits the program with status code 100.
    """

    try:
        var = os.environ[key]
    except KeyError:
        print(f"Environment variable \"{key}\" not set!")
        exit(100)
    else:
        return var

def get_time_gaps(index: pd.DatetimeIndex) -> pd.Series:
    deltas = index.to_series().diff().dropna()
    return deltas[deltas != deltas.mode()[0]]

def clean_and_infer_freq(df):
    # Remove potential index duplicates
    df = df[~df.index.duplicated()]

    # DF freq information lost with concat, check and restore it for resampling
    freq = pd.infer_freq(df.index)
    if freq:
        df.index.freq = freq
        # print(f"Inferred data frequency: {freq}")
    else:
        print(get_time_gaps(df.index))
        raise RuntimeWarning("Could not infer frequency — index is irregular")

    return df

def set_datetime_index(value_col_nm : str, df_in : pd.DataFrame, tz_in = None, 
                       date_col : int = 0, value_col : int = 1):
    """Parse and rebuild the dataframe to ensure correct shape, index format and UTC offset"""
    # Datetime index name is hardcoded
    date_col_nm = "date_time"

    # Ensure the datetime column is parsed and UTC-localized immediately
    dt_series = pd.to_datetime(df_in.iloc[:, date_col], utc=True, errors='coerce')

    # Convert to desired timezone, if specified and not UTC
    if tz_in and tz_in != "UTC":
        try:
            timezone = pytz.timezone(tz_in)
            dt_series = dt_series.dt.tz_convert(timezone)
        except pytz.UnknownTimeZoneError:
            print(f"Warning: Unknown timezone '{tz_in}'. Keeping timestamps in UTC.")

    # Build output DataFrame
    df_out = pd.DataFrame({date_col_nm  : dt_series,
                           value_col_nm : df_in.iloc[:, value_col]})

    # Set converted datetime as index
    df_out = df_out.set_index(date_col_nm)

    # Set index frequency
    # df_out = clean_and_infer_freq(df_out)

    return df_out.sort_index()

def resample_data(df_in        : pd.DataFrame,
                  time_freq    : str = '1h',
                  method       : str = "interpolate"):
    """ Resample based on the datetime index. Alias for 15 minutes is "15T" and "H" for hour"""
    match method:
        case "interpolate":
            return df_in.resample(time_freq).mean().interpolate(method="time")
        case "ffill":
            return df_in.resample(time_freq).ffill()
        case _:
            raise ValueError(f"Unknown resampling method: {method}")

def clean_filename(filename : str) -> str:
    """
    Cleans the filename by replacing spaces and removing special characters.

    Args:
        filename (str): The original filename.

    Returns:
        str: The cleaned filename.
    """
    return filename.translate(str.maketrans(' ', '_', r'.()[]/\?%*:|"<>'))

def td_to_min(td_str : str) -> int:
    """Converts Pandas Timedelta string formats to integer minutes"""
    td = pd.to_timedelta(td_str)
    return int(td.total_seconds() / 60)

def check_empty_date_range(start, end):
    # Check for empty date range
    if not start < end:
        raise ValueError(f"No time between start ({start}) and end dates ({end})")