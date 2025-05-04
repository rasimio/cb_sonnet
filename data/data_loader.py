"""
Data Loading Module for TensorTrade

This module handles loading data from various sources including:
- CSV files
- API endpoints (cryptocurrencies, stocks)
- Databases
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta, date
import requests
from pathlib import Path

# Setup logger
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader class to fetch and prepare data for the trading models
    """

    def __init__(self, symbol: str, timeframe: str,
                start_date: Optional[Union[str, datetime, date]] = None,
                end_date: Optional[Union[str, datetime, date]] = None,
                data_source: str = 'csv', csv_path: Optional[str] = None,
                api_key: Optional[str] = None):
        """
        Initialize the data loader

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '1d')
            start_date: Start date for data (format: 'YYYY-MM-DD' or datetime object)
            end_date: End date for data (format: 'YYYY-MM-DD' or datetime object)
            data_source: Source of data ('csv', 'binance', 'alpha_vantage', 'yahoo')
            csv_path: Path to CSV file if data_source is 'csv'
            api_key: API key for external data sources
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_source = data_source.lower()
        self.csv_path = csv_path
        self.api_key = api_key

        # Parse dates
        self.start_date = self._parse_date(start_date)
        self.end_date = self._parse_date(end_date) if end_date else datetime.now()

        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

    def _parse_date(self, date_input: Optional[Union[str, datetime, date]]) -> Optional[datetime]:
        """
        Parse date input to datetime object

        Args:
            date_input: Date input as string or datetime

        Returns:
            Datetime object or None
        """
        if date_input is None:
            return None
        elif isinstance(date_input, datetime):
            return date_input
        elif isinstance(date_input, date):
            return datetime.combine(date_input, datetime.min.time())
        elif isinstance(date_input, str):
            try:
                return datetime.strptime(date_input, "%Y-%m-%d")
            except ValueError:
                # Try alternate formats
                formats = [
                    "%Y/%m/%d",
                    "%d-%m-%Y",
                    "%d/%m/%Y",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S"
                ]

                for fmt in formats:
                    try:
                        return datetime.strptime(date_input, fmt)
                    except ValueError:
                        continue

                # If we get here, none of the formats worked
                raise ValueError(f"Could not parse date: {date_input}")
        else:
            raise TypeError(f"Unsupported date type: {type(date_input)}")

    def load_data(self) -> pd.DataFrame:
        """
        Load data from the specified source

        Returns:
            DataFrame with OHLCV data
        """
        if self.data_source == 'csv':
            return self._load_from_csv()
        elif self.data_source == 'binance':
            return self._load_from_binance()
        elif self.data_source == 'alpha_vantage':
            return self._load_from_alpha_vantage()
        elif self.data_source == 'yahoo':
            return self._load_from_yahoo()
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")

    def _load_from_csv(self) -> pd.DataFrame:
        """
        Load data from a CSV file

        Returns:
            DataFrame with OHLCV data
        """
        # Use provided path or default
        if self.csv_path:
            file_path = self.csv_path
        else:
            file_path = f"data/{self.symbol}_{self.timeframe}.csv"

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # Load data
        logger.info(f"Loading data from CSV: {file_path}")
        df = pd.read_csv(file_path)

        # Make sure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            # Try to infer columns based on common formats
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'timestamp'})

            if 'Open' in df.columns:
                df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                                      'Close': 'close', 'Volume': 'volume'})

            # Check again after renaming
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")

        # Convert timestamp to datetime if needed
        if df['timestamp'].dtype == object:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                logger.warning("Could not convert timestamp column to datetime")

        # Set timestamp as index
        df = df.set_index('timestamp')

        # Filter by date range if specified
        if self.start_date:
            df = df[df.index >= self.start_date]
        if self.end_date:
            df = df[df.index <= self.end_date]

        return df

    def _load_from_binance(self) -> pd.DataFrame:
        """
        Load data from Binance API

        Returns:
            DataFrame with OHLCV data
        """
        # Map timeframe to Binance interval format
        timeframe_map = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
        }

        interval = timeframe_map.get(self.timeframe)
        if not interval:
            raise ValueError(f"Unsupported timeframe for Binance: {self.timeframe}")

        # Format dates to milliseconds
        start_ms = int(self.start_date.timestamp() * 1000) if self.start_date else None
        end_ms = int(self.end_date.timestamp() * 1000)

        # Build URL
        base_url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': self.symbol.upper(),
            'interval': interval,
            'endTime': end_ms,
            'limit': 1000  # Max limit
        }

        if start_ms:
            params['startTime'] = start_ms

        # Fetch data in chunks if needed
        all_candles = []
        while True:
            logger.info(f"Fetching data from Binance: {params}")
            response = requests.get(base_url, params=params)

            if response.status_code != 200:
                raise Exception(f"Error fetching data from Binance: {response.text}")

            candles = response.json()
            if not candles:
                break

            all_candles.extend(candles)

            # Update parameters for next request
            params['endTime'] = candles[0][0] - 1  # End time is the start of first candle - 1

            # Break if we've reached the start date or have enough data
            if start_ms and params['endTime'] <= start_ms:
                break

            # Avoid excessive requests
            if len(all_candles) >= 10000:
                logger.warning("Reached maximum data points (10,000)")
                break

        # Create DataFrame
        df = pd.DataFrame(all_candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)

        # Set timestamp as index and sort
        df = df.set_index('timestamp')
        df = df.sort_index()

        # Select only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]

        return df

    def _load_from_alpha_vantage(self) -> pd.DataFrame:
        """
        Load data from Alpha Vantage API

        Returns:
            DataFrame with OHLCV data
        """
        if not self.api_key:
            raise ValueError("API key is required for Alpha Vantage")

        # Map timeframe to Alpha Vantage interval format
        timeframe_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '60min', '1d': 'daily', '1w': 'weekly', '1M': 'monthly'
        }

        interval = timeframe_map.get(self.timeframe)
        if not interval:
            raise ValueError(f"Unsupported timeframe for Alpha Vantage: {self.timeframe}")

        # Build URL
        base_url = "https://www.alphavantage.co/query"

        if interval in ['daily', 'weekly', 'monthly']:
            function = f"TIME_SERIES_{interval.upper()}"
            params = {
                'function': function,
                'symbol': self.symbol,
                'outputsize': 'full',
                'apikey': self.api_key
            }
        else:
            function = "TIME_SERIES_INTRADAY"
            params = {
                'function': function,
                'symbol': self.symbol,
                'interval': interval,
                'outputsize': 'full',
                'apikey': self.api_key
            }

        # Fetch data
        logger.info(f"Fetching data from Alpha Vantage: {params}")
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            raise Exception(f"Error fetching data from Alpha Vantage: {response.text}")

        data = response.json()

        # Extract time series data
        if function == "TIME_SERIES_INTRADAY":
            time_series_key = f"Time Series ({interval})"
        elif function == "TIME_SERIES_DAILY":
            time_series_key = "Time Series (Daily)"
        elif function == "TIME_SERIES_WEEKLY":
            time_series_key = "Weekly Time Series"
        elif function == "TIME_SERIES_MONTHLY":
            time_series_key = "Monthly Time Series"
        else:
            raise ValueError(f"Unsupported function: {function}")

        if time_series_key not in data:
            raise Exception(f"Error in Alpha Vantage response: {data}")

        time_series = data[time_series_key]

        # Create DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')

        # Rename columns
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })

        # Convert types
        df.index = pd.to_datetime(df.index)
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)

        # Sort by date
        df = df.sort_index()

        # Filter by date range if specified
        if self.start_date:
            df = df[df.index >= self.start_date]
        if self.end_date:
            df = df[df.index <= self.end_date]

        return df

    def _load_from_yahoo(self) -> pd.DataFrame:
        """
        Load data from Yahoo Finance

        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance package is required for Yahoo Finance data. Install it using: pip install yfinance")

        # Map timeframe to Yahoo period parameter
        timeframe_map = {
            '1m': '1m', '2m': '2m', '5m': '5m', '15m': '15m', '30m': '30m', '60m': '60m', '1h': '60m',
            '90m': '90m', '1d': '1d', '5d': '5d', '1wk': '1wk', '1w': '1wk', '1mo': '1mo', '1M': '1mo',
            '3mo': '3mo'
        }

        interval = timeframe_map.get(self.timeframe)
        if not interval:
            raise ValueError(f"Unsupported timeframe for Yahoo Finance: {self.timeframe}")

        # Format dates
        start_str = self.start_date.strftime("%Y-%m-%d") if self.start_date else None
        end_str = self.end_date.strftime("%Y-%m-%d")

        # Fetch data
        logger.info(f"Fetching data from Yahoo Finance: {self.symbol}, {interval}")
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(
            period="max" if not start_str else None,
            start=start_str,
            end=end_str,
            interval=interval
        )

        # Rename columns
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Select only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]

        return df

    def save_to_csv(self, df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        Save data to CSV file

        Args:
            df: DataFrame to save
            filename: Filename to save to (optional)

        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"data/{self.symbol}_{self.timeframe}.csv"

        # Reset index to save timestamp as column
        df_to_save = df.reset_index()

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save to CSV
        df_to_save.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")

        return filename

    @staticmethod
    def fetch_symbols(exchange: str = 'binance') -> List[str]:
        """
        Fetch available symbols from an exchange

        Args:
            exchange: Exchange name

        Returns:
            List of available symbols
        """
        if exchange.lower() == 'binance':
            response = requests.get("https://api.binance.com/api/v3/exchangeInfo")
            if response.status_code != 200:
                raise Exception(f"Error fetching Binance symbols: {response.text}")

            data = response.json()
            symbols = [symbol['symbol'] for symbol in data['symbols']]
            return symbols
        elif exchange.lower() == 'yahoo':
            logger.warning("Yahoo Finance doesn't provide a comprehensive symbol list API")
            return []
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")

    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """
        Validate that dataframe has required columns and format

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, raises exception otherwise
        """
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check for data types
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must be numeric")

        # Check for NaN values
        if df[required_columns].isna().any().any():
            logger.warning("Data contains NaN values")

        # Check for ascending index
        if not df.index.is_monotonic_increasing:
            logger.warning("Index is not monotonically increasing")

        return True