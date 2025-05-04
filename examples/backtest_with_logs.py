"""
Example of running a backtest with detailed trade history logs
and performance dashboard generation.
"""
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from models.model_factory import load_model
from backtest.engine import BacktestEngine


def run_backtest_with_logs(symbol, timeframe, model_type, start_date, end_date,
                           data_source='yahoo', initial_capital=10000.0):
    """
    Run a backtest and generate detailed trade logs and performance dashboard

    Args:
        symbol: Trading symbol
        timeframe: Timeframe ('1h', '1d', etc.)
        model_type: Model type ('lstm', 'rl', 'ensemble')
        start_date: Start date for backtest data
        end_date: End date for backtest data
        data_source: Data source ('yahoo', 'binance', 'csv', etc.)
        initial_capital: Initial capital for backtest
    """
    print(f"Running backtest for {model_type} model on {symbol} {timeframe}...")

    # Load data
    data_loader = DataLoader(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        data_source=data_source
    )
    data = data_loader.load_data()
    print(f"Loaded {len(data)} data points from {data.index[0]} to {data.index[-1]}")

    # Load model
    model_path = os.path.join('saved_models', f"{model_type}_{symbol}_{timeframe}")
    try:
        model = load_model(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("You need to train the model first. Run main.py with --mode train")
        return

    # Run backtest
    backtest_engine = BacktestEngine(
        model=model,
        initial_capital=initial_capital,
        commission=0.001,  # 0.1%
        risk_per_trade=0.02,  # 2%
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )

    print("Running backtest...")
    results = backtest_engine.run(data)

    # Generate trade history CSV
    csv_path = backtest_engine.generate_trade_history_csv(symbol)
    print(f"Trade history CSV generated: {csv_path}")

    # Generate performance dashboard
    dashboard_path = backtest_engine.generate_performance_dashboard(symbol)
    print(f"Performance dashboard generated: {dashboard_path}")

    # Print summary
    metrics = results['metrics']
    print("\nBacktest Summary:")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['win_count']}")
    print(f"Losing Trades: {metrics['loss_count']}")

    print(f"\nDetailed logs saved to: {csv_path}")
    print(f"Performance dashboard: {dashboard_path}")
    print("Open the dashboard in a web browser to view detailed performance metrics and visualizations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run backtest with detailed logs')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe')
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['lstm', 'rl', 'ensemble'], help='Model type')
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-06-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--data_source', type=str, default='yahoo', help='Data source')
    parser.add_argument('--initial_capital', type=float, default=10000.0, help='Initial capital')

    args = parser.parse_args()

    run_backtest_with_logs(
        symbol=args.symbol,
        timeframe=args.timeframe,
        model_type=args.model,
        start_date=args.start_date,
        end_date=args.end_date,
        data_source=args.data_source,
        initial_capital=args.initial_capital
    )