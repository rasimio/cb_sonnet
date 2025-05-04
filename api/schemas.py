"""
API Schema Definitions for TensorTrade

This module defines the Pydantic models for API request and response schemas.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# Set protected_namespaces to avoid warnings
class BaseModelConfig(BaseModel):
    model_config = {"protected_namespaces": ()}


# Request models
class TrainingRequest(BaseModelConfig):
    model_type: str = Field(..., description="Type of model to train (lstm, rl, ensemble)")
    data_source: str = Field(..., description="Source of training data (binance, csv, alpha_vantage, yahoo)")
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(..., description="Data timeframe (e.g., 1h, 1d)")
    start_date: str = Field(..., description="Start date for training data (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for training data (YYYY-MM-DD)")
    hyperparameters: Dict[str, Any] = Field({}, description="Model hyperparameters")
    api_key: Optional[str] = Field(None, description="API key for data source")
    csv_path: Optional[str] = Field(None, description="Path to CSV file if data_source is 'csv'")


class BacktestRequest(BaseModelConfig):
    model_id: str = Field(..., description="ID of the trained model")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Data timeframe")
    start_date: str = Field(..., description="Start date for backtest (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for backtest (YYYY-MM-DD)")
    initial_capital: float = Field(10000.0, description="Initial capital for backtest")
    commission: float = Field(0.001, description="Trading commission (e.g., 0.001 = 0.1%)")
    risk_per_trade: float = Field(0.02, description="Percentage of capital to risk per trade")
    data_source: str = Field("binance", description="Source of backtest data")
    api_key: Optional[str] = Field(None, description="API key for data source")


class PredictionRequest(BaseModelConfig):
    model_id: str = Field(..., description="ID of the trained model")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Data timeframe")
    bars: int = Field(100, description="Number of historical bars to use")
    data_source: str = Field("binance", description="Source of data")
    api_key: Optional[str] = Field(None, description="API key for data source")


class LiveTradingRequest(BaseModelConfig):
    model_id: str = Field(..., description="ID of the trained model")
    exchange: str = Field(..., description="Exchange name (e.g., binance)")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Data timeframe")
    capital_allocation: float = Field(..., description="Amount of capital to allocate")
    risk_percentage: float = Field(1.0, description="Maximum risk per trade in percentage")
    api_key: str = Field(..., description="Exchange API key")
    api_secret: str = Field(..., description="Exchange API secret")


class ParameterSweepRequest(BaseModelConfig):
    model_id: str = Field(..., description="ID of the trained model")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Data timeframe")
    start_date: str = Field(..., description="Start date for backtest (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for backtest (YYYY-MM-DD)")
    parameters: Dict[str, List[Any]] = Field(..., description="Dictionary of parameter ranges to test")
    data_source: str = Field("binance", description="Source of backtest data")


# Response models
class ModelSummary(BaseModelConfig):
    id: str
    type: str
    symbol: str
    timeframe: str
    created_at: str
    has_config: bool


class ModelsList(BaseModelConfig):
    models: List[ModelSummary]


class ModelDetail(BaseModelConfig):
    id: str
    type: str
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    status: str
    config: Optional[Dict[str, Any]] = None


class TrainingResponse(BaseModelConfig):
    status: str
    model_id: str
    message: str


class BacktestResponse(BaseModelConfig):
    status: str
    backtest_id: str
    message: str


class BacktestSummary(BaseModelConfig):
    id: str
    model_id: str
    created_at: str
    metrics: Dict[str, Any]


class BacktestsList(BaseModelConfig):
    backtests: List[BacktestSummary]


class BacktestDetail(BaseModelConfig):
    id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    dashboard_url: Optional[str] = None


class PredictionResponse(BaseModelConfig):
    model_id: str
    current_price: Optional[float] = None
    predicted_price: Optional[float] = None
    prediction_delta: Optional[float] = None
    prediction_delta_pct: Optional[float] = None
    predicted_return_pct: Optional[float] = None
    predicted_direction: Optional[float] = None
    signal: str
    action: Optional[str] = None
    action_code: Optional[int] = None
    confidence: Optional[float] = None


class LiveTradingResponse(BaseModelConfig):
    status: str
    session_id: str
    message: str


class LiveSessionSummary(BaseModelConfig):
    id: str
    model_id: str
    exchange: str
    symbol: str
    status: str
    started_at: str
    stopped_at: Optional[str] = None


class LiveSessionsList(BaseModelConfig):
    sessions: List[LiveSessionSummary]


class LiveSessionDetail(BaseModelConfig):
    id: str
    status: str
    config: Dict[str, Any]
    started_at: str
    stopped_at: Optional[str] = None
    metrics: Dict[str, Any]
    trades: List[Dict[str, Any]]


class ParameterSweepResponse(BaseModelConfig):
    status: str
    sweep_id: str
    message: str


class HealthResponse(BaseModelConfig):
    status: str
    timestamp: str