"""
API Routes for TensorTrade

This module defines the FastAPI routes for the TensorTrade microservice.
"""
import os
import logging
from visualization import perfomance_charts
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Body
from pydantic import BaseModel, Field

from models.model_factory import create_model, load_model
from data.data_loader import DataLoader
from backtest.engine import BacktestEngine
# from visualization.performance_charts import plot_backtest_results, create_performance_dashboard
from live.exchange_connector import ExchangeConnector
from live.order_manager import OrderManager
from live.risk_manager import RiskManager

# Create logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Define request and response models
class TrainingRequest(BaseModel):
    model_type: str = Field(..., description="Type of model to train (lstm, rl, ensemble)")
    data_source: str = Field(..., description="Source of training data (binance, csv, alpha_vantage, yahoo)")
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(..., description="Data timeframe (e.g., 1h, 1d)")
    start_date: str = Field(..., description="Start date for training data (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for training data (YYYY-MM-DD)")
    hyperparameters: Dict[str, Any] = Field({}, description="Model hyperparameters")
    api_key: Optional[str] = Field(None, description="API key for data source")
    csv_path: Optional[str] = Field(None, description="Path to CSV file if data_source is 'csv'")


class BacktestRequest(BaseModel):
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


class PredictionRequest(BaseModel):
    model_id: str = Field(..., description="ID of the trained model")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Data timeframe")
    bars: int = Field(100, description="Number of historical bars to use")
    data_source: str = Field("binance", description="Source of data")
    api_key: Optional[str] = Field(None, description="API key for data source")


class LiveTradingRequest(BaseModel):
    model_id: str = Field(..., description="ID of the trained model")
    exchange: str = Field(..., description="Exchange name (e.g., binance)")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Data timeframe")
    capital_allocation: float = Field(..., description="Amount of capital to allocate")
    risk_percentage: float = Field(1.0, description="Maximum risk per trade in percentage")
    api_key: str = Field(..., description="Exchange API key")
    api_secret: str = Field(..., description="Exchange API secret")


class ParameterSweepRequest(BaseModel):
    model_id: str = Field(..., description="ID of the trained model")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Data timeframe")
    start_date: str = Field(..., description="Start date for backtest (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for backtest (YYYY-MM-DD)")
    parameters: Dict[str, List[Any]] = Field(..., description="Dictionary of parameter ranges to test")
    data_source: str = Field("binance", description="Source of backtest data")


# In-memory storage for active trading sessions
active_trading_sessions = {}


# ROUTES

# Health check endpoint
@router.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Model training endpoint
@router.post("/models/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train a new model with the specified parameters"""
    try:
        # Generate model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{request.model_type}_{request.symbol}_{request.timeframe}_{timestamp}"

        # Log training request
        logger.info(f"Training new model: {model_id}")

        # Function to run in background
        def train_in_background(model_id: str, request: TrainingRequest):
            try:
                # Load data
                data_loader = DataLoader(
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    data_source=request.data_source,
                    csv_path=request.csv_path,
                    api_key=request.api_key
                )
                data = data_loader.load_data()

                # Create model
                model = create_model(request.model_type, request.hyperparameters)

                # Train model
                model.train(data)

                # Save model
                model_path = os.path.join("saved_models", model_id)
                model.save(model_path)

                logger.info(f"Model {model_id} trained successfully")
            except Exception as e:
                logger.error(f"Error training model {model_id}: {str(e)}")

        # Schedule training in background
        background_tasks.add_task(train_in_background, model_id, request)

        return {
            "status": "training_started",
            "model_id": model_id,
            "message": "Model training started in the background"
        }
    except Exception as e:
        logger.error(f"Error starting model training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting model training: {str(e)}")


# List models endpoint
@router.get("/models")
async def list_models():
    """List all trained models"""
    try:
        models_dir = "saved_models"
        if not os.path.exists(models_dir):
            return {"models": []}

        # Get all model directories
        model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]

        models = []
        for model_dir in model_dirs:
            # Extract model information from directory name
            parts = model_dir.split('_')
            if len(parts) >= 4:
                model_type = parts[0]
                symbol = parts[1]
                timeframe = parts[2]
                timestamp = '_'.join(parts[3:])

                # Check if config file exists
                config_path = os.path.join(models_dir, model_dir, 'config.json')
                has_config = os.path.exists(config_path)

                models.append({
                    "id": model_dir,
                    "type": model_type,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "created_at": timestamp,
                    "has_config": has_config
                })

        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


# Model details endpoint
@router.get("/models/{model_id}")
async def get_model_details(model_id: str):
    """Get details of a specific model"""
    try:
        model_path = os.path.join("saved_models", model_id)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        # Check if config file exists
        config_path = os.path.join(model_path, 'config.json')
        if not os.path.exists(config_path):
            return {
                "id": model_id,
                "status": "available",
                "config": None
            }

        # Load model config
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Extract model details
        parts = model_id.split('_')
        model_type = parts[0]
        symbol = parts[1] if len(parts) > 1 else None
        timeframe = parts[2] if len(parts) > 2 else None

        return {
            "id": model_id,
            "type": model_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "available",
            "config": config
        }
    except Exception as e:
        logger.error(f"Error getting model details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model details: {str(e)}")


# Prediction endpoint
@router.post("/models/predict")
async def get_prediction(request: PredictionRequest):
    """Get prediction from a trained model"""
    try:
        # Load model
        model_path = os.path.join("saved_models", request.model_id)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

        model = load_model(model_path)

        # Load data
        data_loader = DataLoader(
            symbol=request.symbol,
            timeframe=request.timeframe,
            data_source=request.data_source,
            api_key=request.api_key
        )
        data = data_loader.load_data().tail(request.bars)

        # Make prediction
        prediction = model.predict(data)

        # Process prediction based on model type
        model_type = request.model_id.split('_')[0]

        if model_type == 'lstm':
            target_type = model.config.get('target_type', 'price')
            current_price = data.iloc[-1]['close']

            if target_type == 'price':
                prediction_delta = prediction - current_price
                prediction_delta_pct = (prediction - current_price) / current_price * 100

                return {
                    "model_id": request.model_id,
                    "current_price": float(current_price),
                    "predicted_price": float(prediction),
                    "prediction_delta": float(prediction_delta),
                    "prediction_delta_pct": float(prediction_delta_pct),
                    "signal": "BUY" if prediction > current_price else "SELL"
                }
            elif target_type == 'return':
                return {
                    "model_id": request.model_id,
                    "current_price": float(current_price),
                    "predicted_return_pct": float(prediction),
                    "signal": "BUY" if prediction > 0 else "SELL"
                }
            else:  # direction
                return {
                    "model_id": request.model_id,
                    "current_price": float(current_price),
                    "predicted_direction": float(prediction),
                    "signal": "BUY" if prediction > 0.5 else "SELL"
                }
        elif model_type == 'rl':
            # RL model returns a dictionary with action and other info
            return {
                "model_id": request.model_id,
                "current_price": float(data.iloc[-1]['close']),
                "action": prediction['action'],
                "action_code": prediction['action_code']
            }
        else:
            return {
                "model_id": request.model_id,
                "prediction": prediction
            }
    except Exception as e:
        logger.error(f"Error getting prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting prediction: {str(e)}")


# Backtest endpoint
@router.post("/backtest/run")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run backtest with a trained model"""
    try:
        # Generate backtest ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backtest_id = f"backtest_{request.model_id}_{timestamp}"

        # Log backtest request
        logger.info(f"Running backtest: {backtest_id}")

        # Function to run in background
        def backtest_in_background(backtest_id: str, request: BacktestRequest):
            try:
                # Create output directory
                output_dir = os.path.join("backtest_results", backtest_id)
                os.makedirs(output_dir, exist_ok=True)

                # Load model
                model_path = os.path.join("saved_models", request.model_id)
                model = load_model(model_path)

                # Load data
                data_loader = DataLoader(
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    data_source=request.data_source,
                    api_key=request.api_key
                )
                data = data_loader.load_data()

                # Run backtest
                engine = BacktestEngine(
                    model=model,
                    initial_capital=request.initial_capital,
                    commission=request.commission,
                    risk_per_trade=request.risk_per_trade
                )

                results = engine.run(data)

                # Create dashboard
                dashboard_path = perfomance_charts.create_performance_dashboard(results, output_dir)

                # Save results to JSON
                import json
                results_json = {
                    "metrics": results['metrics'],
                    "trades": results['trades_history']
                }

                with open(os.path.join(output_dir, "results.json"), 'w') as f:
                    json.dump(results_json, f, default=str)

                logger.info(f"Backtest {backtest_id} completed successfully")
            except Exception as e:
                logger.error(f"Error running backtest {backtest_id}: {str(e)}")

        # Schedule backtest in background
        background_tasks.add_task(backtest_in_background, backtest_id, request)

        return {
            "status": "backtest_started",
            "backtest_id": backtest_id,
            "message": "Backtest started in the background"
        }
    except Exception as e:
        logger.error(f"Error starting backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting backtest: {str(e)}")


# List backtests endpoint
@router.get("/backtest")
async def list_backtests():
    """List all backtest results"""
    try:
        backtests_dir = "backtest_results"
        if not os.path.exists(backtests_dir):
            return {"backtests": []}

        # Get all backtest directories
        backtest_dirs = [d for d in os.listdir(backtests_dir) if os.path.isdir(os.path.join(backtests_dir, d))]

        backtests = []
        for backtest_dir in backtest_dirs:
            # Check if results file exists
            results_path = os.path.join(backtests_dir, backtest_dir, 'results.json')
            if os.path.exists(results_path):
                # Load results JSON
                import json
                with open(results_path, 'r') as f:
                    results = json.load(f)

                # Extract backtest details
                parts = backtest_dir.split('_')
                model_id = parts[1] if len(parts) > 1 else None
                timestamp = '_'.join(parts[2:]) if len(parts) > 2 else None

                backtests.append({
                    "id": backtest_dir,
                    "model_id": model_id,
                    "created_at": timestamp,
                    "metrics": results.get('metrics', {})
                })

        return {"backtests": backtests}
    except Exception as e:
        logger.error(f"Error listing backtests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing backtests: {str(e)}")


# Backtest details endpoint
@router.get("/backtest/{backtest_id}")
async def get_backtest_details(backtest_id: str):
    """Get details of a specific backtest"""
    try:
        backtest_path = os.path.join("backtest_results", backtest_id)
        if not os.path.exists(backtest_path):
            raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

        # Check if results file exists
        results_path = os.path.join(backtest_path, 'results.json')
        if not os.path.exists(results_path):
            return {
                "id": backtest_id,
                "status": "in_progress",
                "results": None
            }

        # Load results JSON
        import json
        with open(results_path, 'r') as f:
            results = json.load(f)

        # Check if dashboard exists
        dashboard_path = os.path.join(backtest_path, 'dashboard.html')
        has_dashboard = os.path.exists(dashboard_path)

        return {
            "id": backtest_id,
            "status": "completed",
            "results": results,
            "dashboard_url": f"/backtest/{backtest_id}/dashboard" if has_dashboard else None
        }
    except Exception as e:
        logger.error(f"Error getting backtest details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting backtest details: {str(e)}")


# Start live trading endpoint
@router.post("/live/start")
async def start_live_trading(request: LiveTradingRequest):
    """Start live trading with a trained model"""
    try:
        # Generate session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"live_{request.model_id}_{timestamp}"

        # Check if model exists
        model_path = os.path.join("saved_models", request.model_id)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

        # Create output directory
        output_dir = os.path.join("live_results", session_id)
        os.makedirs(output_dir, exist_ok=True)

        # Load model
        model = load_model(model_path)

        # Setup exchange connection
        exchange = ExchangeConnector(
            exchange_id=request.exchange,
            api_key=request.api_key,
            api_secret=request.api_secret
        )

        # Setup risk manager
        risk_manager = RiskManager(
            initial_capital=request.capital_allocation,
            max_risk_per_trade=request.risk_percentage / 100,
            max_open_positions=1
        )

        # Setup order manager
        order_manager = OrderManager(
            exchange=exchange,
            risk_manager=risk_manager,
            symbol=request.symbol
        )

        # Store objects in active sessions
        active_trading_sessions[session_id] = {
            "model": model,
            "exchange": exchange,
            "order_manager": order_manager,
            "risk_manager": risk_manager,
            "config": request.dict(),
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "trades": []
        }

        logger.info(f"Live trading session {session_id} started")

        return {
            "status": "started",
            "session_id": session_id,
            "message": "Live trading session started"
        }
    except Exception as e:
        logger.error(f"Error starting live trading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting live trading: {str(e)}")


# Stop live trading endpoint
@router.post("/live/stop/{session_id}")
async def stop_live_trading(session_id: str):
    """Stop a live trading session"""
    try:
        if session_id not in active_trading_sessions:
            raise HTTPException(status_code=404, detail=f"Live trading session {session_id} not found")

        session = active_trading_sessions[session_id]

        # Close exchange connection
        session["exchange"].close()

        # Update session status
        session["status"] = "stopped"
        session["stopped_at"] = datetime.now().isoformat()

        logger.info(f"Live trading session {session_id} stopped")

        return {
            "status": "stopped",
            "session_id": session_id,
            "message": "Live trading session stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping live trading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping live trading: {str(e)}")


# List live trading sessions endpoint
@router.get("/live")
async def list_live_sessions():
    """List all live trading sessions"""
    try:
        sessions = []
        for session_id, session in active_trading_sessions.items():
            sessions.append({
                "id": session_id,
                "model_id": session["config"]["model_id"],
                "exchange": session["config"]["exchange"],
                "symbol": session["config"]["symbol"],
                "status": session["status"],
                "started_at": session["started_at"],
                "stopped_at": session.get("stopped_at")
            })

        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error listing live trading sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing live trading sessions: {str(e)}")


# Live trading status endpoint
@router.get("/live/{session_id}")
async def get_live_status(session_id: str):
    """Get status of a live trading session"""
    try:
        if session_id not in active_trading_sessions:
            raise HTTPException(status_code=404, detail=f"Live trading session {session_id} not found")

        session = active_trading_sessions[session_id]

        # Get latest performance metrics
        order_manager = session["order_manager"]

        metrics = {
            "total_trades": len(order_manager.trades_history),
            "open_positions": len(order_manager.open_positions),
            "initial_capital": order_manager.risk_manager.initial_capital,
            "current_equity": order_manager.calculate_equity(),
            "profit": order_manager.calculate_equity() - order_manager.risk_manager.initial_capital,
            "profit_pct": (order_manager.calculate_equity() / order_manager.risk_manager.initial_capital - 1) * 100
        }

        return {
            "id": session_id,
            "status": session["status"],
            "config": session["config"],
            "started_at": session["started_at"],
            "stopped_at": session.get("stopped_at"),
            "metrics": metrics,
            "trades": order_manager.trades_history
        }
    except Exception as e:
        logger.error(f"Error getting live trading status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting live trading status: {str(e)}")


# Parameter sweep endpoint
@router.post("/backtest/parameter-sweep")
async def run_parameter_sweep(request: ParameterSweepRequest, background_tasks: BackgroundTasks):
    """Run backtest with different parameter combinations"""
    try:
        # Generate sweep ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_id = f"sweep_{request.model_id}_{timestamp}"

        # Log parameter sweep request
        logger.info(f"Running parameter sweep: {sweep_id}")

        # Function to run in background
        def sweep_in_background(sweep_id: str, request: ParameterSweepRequest):
            try:
                from backtest.engine import run_parameter_sweep

                # Create output directory
                output_dir = os.path.join("backtest_results", sweep_id)
                os.makedirs(output_dir, exist_ok=True)

                # Load model
                model_path = os.path.join("saved_models", request.model_id)
                model = load_model(model_path)

                # Load data
                data_loader = DataLoader(
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    data_source=request.data_source
                )
                data = data_loader.load_data()

                # Run parameter sweep
                results = run_parameter_sweep(model, data, request.parameters)

                # Extract main metrics for comparison
                metrics_to_track = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct']

                comparison = []
                for i, result in enumerate(results):
                    params_str = ', '.join([f"{k}={v}" for k, v in result['parameters'].items()])

                    comparison.append({
                        "id": i,
                        "parameters": result['parameters'],
                        "params_str": params_str,
                        "metrics": {metric: result['metrics'][metric] for metric in metrics_to_track}
                    })

                # Sort by descending Sharpe ratio
                comparison = sorted(comparison, key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)

                # Save results to JSON
                import json
                with open(os.path.join(output_dir, "results.json"), 'w') as f:
                    json.dump(comparison, f, default=str)

                logger.info(f"Parameter sweep {sweep_id} completed successfully")
            except Exception as e:
                logger.error(f"Error running parameter sweep {sweep_id}: {str(e)}")

        # Schedule parameter sweep in background
        background_tasks.add_task(sweep_in_background, sweep_id, request)

        return {
            "status": "sweep_started",
            "sweep_id": sweep_id,
            "message": "Parameter sweep started in the background"
        }
    except Exception as e:
        logger.error(f"Error starting parameter sweep: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting parameter sweep: {str(e)}")