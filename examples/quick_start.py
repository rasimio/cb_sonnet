"""
TensorTrade Quick Start Example

This script demonstrates the basic functionality of TensorTrade:
1. Download historical data
2. Train an LSTM model
3. Run a backtest
4. Visualize the results
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from models.lstm_model import LSTMModel
from backtest.engine import BacktestEngine
from visualization.performance_charts import plot_backtest_results, create_performance_dashboard

# Configuration
SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"
START_DATE = "2022-01-01"
END_DATE = "2023-01-01"
TEST_START_DATE = "2023-01-01"
TEST_END_DATE = "2023-06-01"
DATA_SOURCE = "binance"  # or "yahoo", "csv", etc.


def main():
    print("TensorTrade Quick Start Example")
    print("-------------------------------")

    # Step 1: Download historical data
    print("\n1. Downloading historical data...")

    train_loader = DataLoader(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE,
        data_source=DATA_SOURCE
    )

    train_data = train_loader.load_data()
    print(f"Downloaded {len(train_data)} training data points")

    # Save data to CSV
    train_data_path = train_loader.save_to_csv(train_data)
    print(f"Training data saved to {train_data_path}")

    # Download test data
    test_loader = DataLoader(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        data_source=DATA_SOURCE
    )

    test_data = test_loader.load_data()
    print(f"Downloaded {len(test_data)} test data points")

    # Step 2: Create and train the model
    print("\n2. Training LSTM model...")

    # Model configuration
    model_config = {
        'units': 128,
        'layers': 2,
        'dropout': 0.2,
        'sequence_length': 60,
        'prediction_horizon': 1,
        'target_type': 'price',
        'learning_rate': 0.001,
        'use_technical_indicators': True
    }

    # Create model
    model = LSTMModel(model_config)

    # Train model
    history = model.train(
        train_data,
        epochs=50,
        batch_size=32,
        validation_split=0.2
    )

    # Save model
    model_path = os.path.join('saved_models', f"lstm_{SYMBOL}_{TIMEFRAME}_quickstart")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Step 3: Run backtest
    print("\n3. Running backtest...")

    # Backtest configuration
    backtest_config = {
        'initial_capital': 10000.0,
        'commission': 0.001,  # 0.1%
        'risk_per_trade': 0.02,  # 2%
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04
    }

    # Create backtest engine
    engine = BacktestEngine(
        model=model,
        **backtest_config
    )

    # Run backtest
    results = engine.run(test_data)

    # Step 4: Visualize results
    print("\n4. Visualizing results...")

    # Create output directory
    output_dir = os.path.join("backtest_results", f"quickstart_{SYMBOL}_{TIMEFRAME}")
    os.makedirs(output_dir, exist_ok=True)

    # Create dashboard
    dashboard_path = create_performance_dashboard(results, output_dir)
    print(f"Dashboard created at {dashboard_path}")

    # Display summary metrics
    print("\nBacktest Summary:")
    print(f"Total Return: {results['metrics']['total_return_pct']:.2f}%")
    print(f"Annualized Return: {results['metrics']['annualized_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['metrics']['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {results['metrics']['win_rate_pct']:.2f}%")
    print(f"Total Trades: {results['metrics']['total_trades']}")
    print(f"Final Equity: ${results['metrics']['final_equity']:.2f}")

    print("\nDone! Open the dashboard HTML file to view detailed results.")


if __name__ == "__main__":
    main()