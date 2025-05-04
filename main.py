"""
TensorTrade - LSTM & RL Trading Bot
Main entry point for the application
"""
import os
import argparse
import yaml
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import router
from utils.config import load_config
from utils.logging_utils import setup_logging

# Setup argument parser
parser = argparse.ArgumentParser(description='TensorTrade - LSTM & RL Trading Bot')
parser.add_argument('--config', type=str, default='config/default_config.yaml',
                   help='Path to configuration file')
parser.add_argument('--mode', type=str, choices=['api', 'train', 'backtest', 'live'],
                   default='api', help='Application mode')
parser.add_argument('--model', type=str, choices=['lstm', 'rl', 'ensemble'],
                   default='ensemble', help='Model type to use')
parser.add_argument('--symbol', type=str, default='BTCUSDT',
                   help='Trading symbol')
parser.add_argument('--timeframe', type=str, default='1h',
                   help='Trading timeframe')

# Create directories if they don't exist
for directory in ['logs', 'saved_models', 'data']:
    os.makedirs(directory, exist_ok=True)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="TensorTrade API",
        description="API for LSTM & RL Trading Bot",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router)

    return app

def run_training(config, args):
    """Run model training with given configuration"""
    from models.model_factory import create_model
    from data.data_loader import DataLoader

    logger.info(f"Starting training for {args.model} model on {args.symbol} {args.timeframe}")

    # Load data
    data_loader = DataLoader(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=config['training'].get('start_date'),
        end_date=config['training'].get('end_date'),
        data_source=config['data'].get('source')
    )
    data = data_loader.load_data()

    # Create model
    model = create_model(args.model, config['models'][args.model])

    # Train model based on model type
    if args.model == 'lstm':
        # LSTM model uses epochs, batch_size, validation_split
        model.train(
            data,
            epochs=config['training'].get('epochs', 100),
            batch_size=config['training'].get('batch_size', 64),
            validation_split=config['training'].get('validation_split', 0.2)
        )
    elif args.model == 'rl':
        # RL model uses total_timesteps
        model.train(
            data,
            total_timesteps=config['training'].get('total_timesteps', 500000),
            eval_freq=config['training'].get('eval_freq', 10000),
            eval_episodes=config['training'].get('eval_episodes', 5)
        )
    elif args.model == 'ensemble':
        # Ensemble model might need specific parameters
        model.train(data)
    else:
        # Generic fallback
        model.train(data)

    # Save model
    model_path = os.path.join('saved_models', f"{args.model}_{args.symbol}_{args.timeframe}")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

def run_backtest(config, args):
    """Run backtest with given configuration"""
    from models.model_factory import load_model
    from data.data_loader import DataLoader
    from backtest.engine import BacktestEngine
    from visualization import perfomance_charts

    logger.info(f"Starting backtest for {args.model} model on {args.symbol} {args.timeframe}")

    # Load data
    data_loader = DataLoader(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=config['backtest'].get('start_date'),
        end_date=config['backtest'].get('end_date'),
        data_source=config['data'].get('source')
    )
    data = data_loader.load_data()

    # Load model
    model_path = os.path.join('saved_models', f"{args.model}_{args.symbol}_{args.timeframe}")
    model = load_model(model_path)

    # Run backtest
    backtest_engine = BacktestEngine(
        model=model,
        initial_capital=config['backtest'].get('initial_capital', 10000.0),
        commission=config['backtest'].get('commission', 0.001),
        risk_per_trade=config['backtest'].get('risk_per_trade', 0.02)
    )
    results = backtest_engine.run(data)

    # Generate visualization
    perfomance_charts.plot_backtest_results(results,
                         output_path=f"backtest_results_{args.model}_{args.symbol}_{args.timeframe}.html")

    logger.info(f"Backtest completed. Results saved.")

def run_live_trading(config, args):
    """Run live trading with given configuration"""
    from models.model_factory import load_model
    from live.exchange_connector import ExchangeConnector
    from live.order_manager import OrderManager
    from live.risk_manager import RiskManager

    logger.info(f"Starting live trading for {args.model} model on {args.symbol} {args.timeframe}")

    # Load model
    model_path = os.path.join('saved_models', f"{args.model}_{args.symbol}_{args.timeframe}")
    model = load_model(model_path)

    # Setup exchange connection
    exchange = ExchangeConnector(
        exchange_id=config['live'].get('exchange'),
        api_key=config['live'].get('api_key'),
        api_secret=config['live'].get('api_secret')
    )

    # Setup risk manager
    risk_manager = RiskManager(
        initial_capital=config['live'].get('initial_capital'),
        max_risk_per_trade=config['live'].get('max_risk_per_trade'),
        max_open_positions=config['live'].get('max_open_positions')
    )

    # Setup order manager
    order_manager = OrderManager(
        exchange=exchange,
        risk_manager=risk_manager,
        symbol=args.symbol
    )

    # Start trading loop
    from time import sleep
    try:
        while True:
            # Get latest data
            data = exchange.get_latest_data(args.symbol, args.timeframe, 100)

            # Generate predictions
            prediction = model.predict(data)

            # Execute trading logic
            order_manager.process_prediction(prediction)

            # Wait for next interval
            interval_seconds = config['live'].get('update_interval_seconds', 60)
            logger.info(f"Waiting {interval_seconds} seconds for next update")
            sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Live trading stopped by user")
    finally:
        exchange.close()

def main():
    """Main application entry point"""
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Execute requested mode
    if args.mode == 'api':
        logger.info("Starting API server")
        app = create_app()
        uvicorn.run(app, host=config['api'].get('host', '0.0.0.0'), port=config['api'].get('port', 8000))
    elif args.mode == 'train':
        run_training(config, args)
    elif args.mode == 'backtest':
        run_backtest(config, args)
    elif args.mode == 'live':
        run_live_trading(config, args)
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()