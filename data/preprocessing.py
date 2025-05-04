"""
Data Preprocessing Utilities for TensorTrade

This module provides functions for data normalization, feature engineering,
and sequence creation for time series data.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalize_data(data: pd.DataFrame, method: str = 'minmax') -> Tuple[np.ndarray, Any]:
    """
    Normalize data using specified method

    Args:
        data: DataFrame containing raw data
        method: Normalization method ('minmax' or 'standard')

    Returns:
        Tuple of (normalized_data, scaler)
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    # Create a copy of the data for normalization
    normalized_data = scaler.fit_transform(data.values)

    return normalized_data, scaler


def create_sequences(data: np.ndarray, sequence_length: int, target_column_idx: int = -1) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction

    Args:
        data: Normalized data as numpy array
        sequence_length: Length of each sequence
        target_column_idx: Index of target column (-1 for last column)

    Returns:
        Tuple of (X, y) where X is input sequences and y is target values
    """
    X, y = [], []

    for i in range(len(data) - sequence_length):
        # Extract sequence
        seq = data[i:i + sequence_length]

        # Extract target (next value after sequence)
        target = data[i + sequence_length, target_column_idx]

        X.append(seq)
        y.append(target)

    return np.array(X), np.array(y)


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to price data

    Args:
        data: DataFrame with OHLCV data

    Returns:
        DataFrame with added technical indicators
    """
    df = data.copy()

    # Simple moving averages
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()

    # Exponential moving averages
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()

    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # Price momentum
    df['momentum_14'] = df['close'] / df['close'].shift(14) - 1

    # Rate of Change
    df['roc_10'] = (df['close'] / df['close'].shift(10) - 1) * 100

    # Stochastic Oscillator
    n = 14
    df['stoch_k'] = 100 * ((df['close'] - df['low'].rolling(n).min()) /
                           (df['high'].rolling(n).max() - df['low'].rolling(n).min()))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # Williams %R
    df['williams_r'] = -100 * ((df['high'].rolling(n).max() - df['close']) /
                               (df['high'].rolling(n).max() - df['low'].rolling(n).min()))

    # Commodity Channel Index
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    mean_dev = pd.Series(np.zeros(len(df)))
    for i in range(n - 1, len(df)):
        mean_dev.iloc[i] = np.mean(np.abs(typical_price.iloc[i - n + 1:i + 1] -
                                          typical_price.iloc[i - n + 1:i + 1].mean()))
    df['cci'] = (typical_price - typical_price.rolling(n).mean()) / (0.015 * mean_dev)

    return df


def process_data_for_model(data: pd.DataFrame, model_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Process data for model training or prediction

    Args:
        data: DataFrame with OHLCV data
        model_config: Model configuration dictionary

    Returns:
        Tuple of (X, y, scaler)
    """
    # Add technical indicators if requested
    if model_config.get('use_technical_indicators', True):
        data = add_technical_indicators(data)

    # Drop NaN values
    data = data.dropna()

    # Get feature columns
    feature_columns = model_config.get('feature_columns', ['open', 'high', 'low', 'close', 'volume'])

    # Create target variable
    target_column = model_config.get('target_column', 'close')
    prediction_horizon = model_config.get('prediction_horizon', 1)

    if model_config.get('target_type', 'price') == 'price':
        # Predict actual price
        data['target'] = data[target_column].shift(-prediction_horizon)
    elif model_config.get('target_type', 'price') == 'return':
        # Predict percentage return
        data['target'] = data[target_column].pct_change(prediction_horizon).shift(-prediction_horizon) * 100
    else:
        # Predict price direction (1 for up, 0 for down)
        data['target'] = (data[target_column].shift(-prediction_horizon) >
                          data[target_column]).astype(int)

    # Drop rows with NaN targets
    data = data.dropna()

    # Normalize data
    normalized_data, scaler = normalize_data(data[feature_columns + ['target']])

    # Create sequences
    sequence_length = model_config.get('sequence_length', 60)
    X, y = create_sequences(
        normalized_data,
        sequence_length=sequence_length,
        target_column_idx=-1  # Target is the last column
    )

    return X, y, scaler