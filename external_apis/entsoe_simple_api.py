from entsoe import EntsoePandasClient
import datetime as dt
import pandas as pd
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from typing import Callable

# Import utlis and simple cache
from external_apis.entsoe_api_utils import *
from external_apis.simple_cache import SimpleCache

# Load .env for environment variables
load_dotenv()

# Replace with your own ENTSO-E API security token
ENTSOE_API_KEY = get_env_var("ENTSOE_SECURITY_TOKEN")

class entsoeSimpleAPI:
    def __init__(self, timezone="UTC"):
        self.client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
        self.tz     = timezone

    def _get_entsoe_data(self,
                         val_nm          : str, 
                         start           : pd.Timestamp, 
                         end             : pd.Timestamp,
                         query_func      : Callable[[str, pd.Timestamp, pd.Timestamp], pd.Series],
                         rs_freq         : str = '1h',
                         rs_method       : str = "interpolate",
                         **country_codes : str, 
                         ) -> pd.DataFrame:
        # Init and try to get data from cache
        Cache = SimpleCache(filepath   = "./data_cache", 
                            value_col  = val_nm, 
                            history    = '2W', 
                            timezone   = self.tz, 
                            query_freq = rs_freq
                            )

        # Check if the cache already has data for the queried range, if not, query API and update cache
        if Cache.has_data(start, end):
           print(f"[CACHE] Queried ENTSO-E data \"{val_nm}\" from {start} to {end} loaded from cache")
           return self._clip_to_range(Cache.load(), start, end)
        
        print(f"[API] Fetching ENTSO-E \"{val_nm}\" data from {start} to {end}...")

        # Updated entsoe-py decorators require start and end defined explicitly, otherwise assumed as kwargs
        ds = query_func(**country_codes, start=start, end=end)

        if isinstance(ds, pd.Series):
            df = ds.to_frame().reset_index()
        else:
            df = ds.reset_index()

        # Parse and rebuild the dataframe to ensure correct shape, index format and UTC offset
        df_dt = set_datetime_index(val_nm, df, self.tz)

        # Resample to 1h, as 15 min market went live on 30.9.2025
        df_rs = resample_data(df_dt, rs_freq, rs_method)

        # Clip resampled data to requested range to avoid issues with inclusive and exclusive endpoints
        df_clipped = self._clip_to_range(df_rs, start, end)

        # Create or update local cache
        Cache.save(df_clipped)

        # Return requested data
        return df_clipped

    def _clip_to_range(self,
                       df    : pd.DataFrame,
                       start : pd.Timestamp,
                       end   : pd.Timestamp,
                       ) -> pd.DataFrame:
        """Clip dataframe to the requested time range with inclusive start and exlusive end"""
        return df.loc[(df.index >= start) & (df.index < end)]

    def _get_forecast_time_range(self, start_date : dt.datetime, query_hours : int = 24):
        """Calculates correct start and end dates, localizes to set timezone, 
           converts datetimes to Pandas Timestamp for UTC aware arithmetics and
           removes redundant sub-minute precision."""
        # Convert to timezone-aware Pandas timestamps for ENSTSO-E API client
        start = pd.Timestamp(start_date).tz_localize(ZoneInfo(self.tz))

        # Remove minutes to always get current hour DA spot price
        start = start.replace(minute=0, second=0, microsecond=0)

        # Pandas calculates deltas with DST-aware arithmetic by default, so this works even during DST transitions
        if query_hours == 24:
            # Next tz aware midnight, accounting for DST
            end = (start.normalize() + pd.DateOffset(days=1))
        else:
            # Explicit elapsed-hour query
            end = start + pd.Timedelta(hours=query_hours)
        
        return start, end
    
    def _get_observation_time_range(self, start_date : dt.datetime, end_date : dt.datetime):
        """Localizes to set timezone, converts datetimes to Pandas Timestamp 
           for UTC aware arithmetics and removes redundant sub-minute precision."""
        # Localize timestamps and remove redundant precision
        start = pd.Timestamp(start_date).tz_localize(ZoneInfo(self.tz)).replace(second=0, microsecond=0)
        end   = pd.Timestamp(end_date).tz_localize(ZoneInfo(self.tz)).replace(second=0, microsecond=0)
        
        return start, end

    def get_day_ahead_spot_prices(self,
                                  now_date   : dt.datetime,
                                  query_hours  : int = 24,
                                  country_code : str = 'FI',
                                  cache_dir    : str = "./data_cache/",
                                  use_cache    : bool = True
                                  ) -> pd.DataFrame:
        """
        Fetches day-ahead electricity prices for a specified country and time range.
        This method retrieves day-ahead electricity prices from the ENTSO-E API for a given country
        and time range. It supports caching to avoid redundant API calls and improve performance.
        Args:
            start_date (datetime.datetime): The starting date and time for the query.
            query_hours (int, optional): The number of hours to query from the start_date. Defaults to 24.
            country_code (str, optional): The country code for which to fetch the prices (e.g., 'FI' for Finland). Defaults to 'FI'.
            cache_dir (str, optional): The directory where cached data will be stored. Defaults to "./data/".
            use_cache (bool, optional): Whether to use cached data if available. Defaults to True.
        Returns:
            pandas.DataFrame: A DataFrame containing the day-ahead electricity prices with datetime indices.
        Raises:
            FileNotFoundError: If the cache file does not exist and caching is enabled.
            ValueError: If the API query fails or returns invalid data.
        Notes:
            - The method automatically adjusts the start_date to remove minutes, seconds, and microseconds.
            - If caching is enabled and the cache file exists, the data is loaded from the cache.
            - If caching is enabled and the cache file does not exist, the fetched data is saved to the cache.
        """
        # Add one day to the current date for a day-ahead query (date() sets the time at day boundary)
        start_date = now_date.date() + dt.timedelta(days=1)

        # Format and localize time range for forecast query
        start, end = self._get_forecast_time_range(start_date, query_hours)

        # Create value name
        val_type   = "spot_da"
        val_nm     = f"{country_code}_{val_type}"

        # Get the requested data with the correct client query
        res_df = self._get_entsoe_data(val_nm, 
                                       start, 
                                       end, 
                                       self.client.query_day_ahead_prices, 
                                       '1h',
                                       country_code = country_code)
        
        # Get data from cache for the requested range
        return res_df

    def get_range_spot_prices(self,
                              start_date   : dt.datetime,
                              end_date     : dt.datetime,
                              country_code : str = 'FI',
                              cache_dir    : str = "./data/",
                              use_cache    : bool = True
                              ) -> pd.DataFrame:
        """
        Fetches spot prices for a specified country within a given date range.
        This method retrieves historical spot price data from the ENTSO-E API for a specified country
        and time range. The data can be cached locally to avoid redundant API calls.
        Args:
            start_date (datetime.datetime): The start date of the time range for which to fetch spot prices.
            end_date (datetime.datetime): The end date of the time range for which to fetch spot prices.
            country_code (str, optional): The country code for which to fetch spot prices. Defaults to 'FI' (Finland).
            cache_dir (str, optional): Directory path to store cached data. Defaults to "./data/".
            use_cache (bool, optional): Whether to use cached data if available. Defaults to True.
        Returns:
            pandas.DataFrame: A DataFrame containing the spot price data with a datetime index.
        Raises:
            FileNotFoundError: If the cache file does not exist and `use_cache` is set to True.
            Exception: If there is an issue with the ENTSO-E API query or data processing.
        Notes:
            - The timestamps are localized to the timezone specified in the class (`self.tz`).
            - If caching is enabled and a cache file exists, the data will be loaded from the cache.
            - If caching is enabled and no cache file exists, the data will be saved to the cache after fetching.
        """
        # Format and localize time range for forecast query
        start, end = self._get_observation_time_range(start_date, end_date)
        
        # Create value and cache names
        val_type   = "spot_obs"
        val_nm     = f"{country_code}_{val_type}"

        # Get the requested data with the correct client query
        res_df = self._get_entsoe_data(val_nm, 
                                       start, 
                                       end, 
                                       self.client.query_day_ahead_prices, 
                                       '1h',
                                       country_code = country_code)
        
        # Get data from cache for the requested range
        return res_df

    def get_intraday_imbalance_prices(self,
                                      start_date   : dt.datetime,
                                      end_date     : dt.datetime,
                                      country_code : str = 'FI',
                                      cache_dir    : str = "./data/",
                                      use_cache    : bool = True
                                      ) -> pd.DataFrame:
        """
        Fetches intraday imbalance prices for a specified country within a given date range.
        This method retrieves historical intraday imbalance price data from the ENTSO-E API for a specified country
        and time range. The data can be cached locally to avoid redundant API calls.
        Args:
            start_date (datetime.datetime): The start date of the time range for which to fetch imbalance prices.
            end_date (datetime.datetime): The end date of the time range for which to fetch imbalance prices.
            country_code (str, optional): The country code for which to fetch imbalance prices. Defaults to 'FI' (Finland).
            cache_dir (str, optional): Directory path to store cached data. Defaults to "./data/".
            use_cache (bool, optional): Whether to use cached data if available. Defaults to True.
        Returns:
            pandas.DataFrame: A DataFrame containing the intraday imbalance price data with a datetime index.
        """
        # Format and localize time range for forecast query
        start, end = self._get_observation_time_range(start_date, end_date)

        # Create value and cache names
        val_type = "intraday_imbal_prices"
        val_nm   = f"{country_code}_{val_type}"

        # Get the requested data with the correct client query
        res_df = self._get_entsoe_data(val_nm, 
                                        start, 
                                        end, 
                                        self.client.query_imbalance_prices, 
                                        '1h')
        
        # Get data from cache for the requested range
        return res_df

if __name__ == "__main__":
    # Test queries and cache
    API = entsoeSimpleAPI('Europe/Helsinki')

    # Get latest day-ahead prices
    # now   = dt.datetime.now()
    now   = dt.datetime(2026, 11, 5, 14, 15)

    # Check if day-ahead spot prices are available i.e. is the current time past 14:15
    check = dt.datetime(now.year, now.month, now.day, 14, 15)

    if now >= check:
        # Day-ahead prices are available, query for them
        spot_df = API.get_day_ahead_spot_prices(now, 24, "FI")
    else:
        # Day-ahead prices are not yet available
        raise RuntimeError(f"[ENTSOE-API] Day-ahead prices are not yet available (query time: {now.hour}:{now.minute}:{now.second})")