"""
Backtest Logging Utilities for TensorTrade

This module provides functions for generating detailed trade logs and
performance reports from backtest results.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger(__name__)


def generate_trade_history_csv(backtest_results: Dict[str, Any], symbol: str,
                               output_path: Optional[str] = None) -> str:
    """
    Generate a detailed CSV of trade history from backtest results

    Args:
        backtest_results: Dictionary with backtest results
        symbol: Trading symbol used for the backtest
        output_path: Path to save the CSV file (None for auto-generated path)

    Returns:
        Path to the generated CSV file
    """
    logger.info("Generating trade history CSV...")

    # Extract trades from backtest results
    trades = backtest_results['trades_history']
    equity_curve = backtest_results['equity_curve']

    # Get initial capital from metrics or use a default value if not available
    if 'metrics' in backtest_results and 'initial_equity' in backtest_results['metrics']:
        initial_capital = backtest_results['metrics']['initial_equity']
    else:
        # Fall back to the first value in equity curve or a default value
        initial_capital = equity_curve.iloc[0] if len(equity_curve) > 0 else 10000.0

    # Create a list to store detailed trade records
    detailed_trades = []

    # Running variables for portfolio tracking
    running_profit = 0
    portfolio_value = initial_capital

    # Process each trade
    for i, trade in enumerate(trades):
        # Extract trade details
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        position_type = trade['type']
        size = trade['size']
        pnl = trade['pnl']
        pnl_pct = trade.get('pnl_pct', (pnl / size) * 100 if size > 0 else 0)

        # Calculate quantity (in base currency units)
        quantity = size / entry_price

        # Update running profit
        running_profit += pnl

        # Calculate portfolio value after trade
        portfolio_value = initial_capital + running_profit

        # Record the entry
        detailed_trades.append({
            'datetime': entry_date,
            'action': 'BUY' if position_type == 'long' else 'SELL',
            'symbol': symbol,
            'price': entry_price,
            'quantity': quantity,
            'trade_value': size,
            'portfolio_value': portfolio_value - pnl,  # Portfolio value before this trade's PnL
            'running_profit': running_profit - pnl,
            'trade_id': i + 1,
            'trade_phase': 'ENTRY'
        })

        # Record the exit
        detailed_trades.append({
            'datetime': exit_date,
            'action': 'SELL' if position_type == 'long' else 'BUY',
            'symbol': symbol,
            'price': exit_price,
            'quantity': quantity,
            'trade_value': size + pnl,
            'portfolio_value': portfolio_value,
            'running_profit': running_profit,
            'trade_id': i + 1,
            'trade_phase': 'EXIT',
            'trade_pnl': pnl,
            'trade_pnl_pct': pnl_pct
        })

    # Convert to DataFrame and sort by datetime
    trades_df = pd.DataFrame(detailed_trades)

    # Skip further processing if no trades were found
    if len(trades_df) == 0:
        # Create an empty CSV
        empty_columns = ['datetime', 'action', 'symbol', 'price', 'quantity',
                         'trade_value', 'portfolio_value', 'running_profit', 'trade_id', 'trade_phase']
        trades_df = pd.DataFrame(columns=empty_columns)

        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "backtest_results"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"trade_history_{symbol}_{timestamp}.csv")

        # Save empty CSV
        trades_df.to_csv(output_path, index=False)
        print(f"\n=== CSV TRADE HISTORY GENERATED ===")
        print(f"File saved to: {os.path.abspath(output_path)}")
        print(f"Contains 0 trades - no trades were executed in this backtest")
        print(f"===============================\n")

        return output_path

    # Sort by datetime
    trades_df = trades_df.sort_values('datetime')

    # Format numeric columns
    trades_df['price'] = trades_df['price'].round(2)
    trades_df['quantity'] = trades_df['quantity'].round(8)
    trades_df['trade_value'] = trades_df['trade_value'].round(2)
    trades_df['portfolio_value'] = trades_df['portfolio_value'].round(2)
    trades_df['running_profit'] = trades_df['running_profit'].round(2)
    if 'trade_pnl' in trades_df.columns:
        trades_df['trade_pnl'] = trades_df['trade_pnl'].round(2)
    if 'trade_pnl_pct' in trades_df.columns:
        trades_df['trade_pnl_pct'] = trades_df['trade_pnl_pct'].round(2)

    # Generate the output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "backtest_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"trade_history_{symbol}_{timestamp}.csv")

    # Ensure we have an absolute path for clearer messaging
    output_path = os.path.abspath(output_path)

    # Save to CSV
    trades_df.to_csv(output_path, index=False)

    # Print absolute path to make it easier to find
    print(f"\n=== CSV TRADE HISTORY GENERATED ===")
    print(f"File saved to: {output_path}")
    print(f"Contains {len(trades_df)} trade entries for {len(trades)} complete trades")
    print(f"===============================\n")

    logger.info(f"Trade history saved to {output_path}")

    return output_path


def generate_performance_dashboard(backtest_results: Dict[str, Any], symbol: str,
                                   csv_path: Optional[str] = None,
                                   output_dir: Optional[str] = None) -> str:
    """
    Generate performance dashboard from backtest results

    Args:
        backtest_results: Dictionary with backtest results
        symbol: Trading symbol
        csv_path: Path to trade history CSV (if already generated)
        output_dir: Directory to save dashboard files

    Returns:
        Path to the generated dashboard HTML file
    """
    logger.info("Generating performance dashboard...")

    # Generate CSV if not provided
    if csv_path is None:
        csv_path = generate_trade_history_csv(backtest_results, symbol)

    # Create output directory if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("backtest_results", f"dashboard_{symbol}_{timestamp}")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data from CSV
    df = pd.read_csv(csv_path)

    # Convert datetime to proper type
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Filter to just the exit rows for trade-level metrics
    exit_trades = df[df['trade_phase'] == 'EXIT'].copy()

    # Calculate performance metrics
    initial_capital = backtest_results['metrics']['initial_capital']
    final_capital = backtest_results['metrics']['final_equity']
    total_return = backtest_results['metrics']['total_return_pct']
    total_trades = backtest_results['metrics']['total_trades']
    win_rate = backtest_results['metrics']['win_rate_pct']
    max_drawdown = backtest_results['metrics']['max_drawdown_pct']
    sharpe_ratio = backtest_results['metrics']['sharpe_ratio']
    profit_factor = backtest_results['metrics']['profit_factor']

    # Extract additional metrics
    wins = backtest_results['metrics']['win_count']
    losses = backtest_results['metrics']['loss_count']
    avg_win = backtest_results['metrics']['avg_win']
    avg_loss = backtest_results['metrics']['avg_loss']
    avg_profit = backtest_results['metrics']['avg_profit']

    # Calculate holding period metrics if available
    if 'entry_date' in exit_trades.columns:
        exit_trades['entry_time'] = exit_trades.apply(
            lambda row: df[(df['trade_id'] == row['trade_id']) & (df['trade_phase'] == 'ENTRY')]['datetime'].iloc[0],
            axis=1
        )
        exit_trades['holding_period'] = (exit_trades['datetime'] - exit_trades['entry_time']).dt.total_seconds() / 3600  # in hours

        avg_holding_period = exit_trades['holding_period'].mean()
        max_holding_period = exit_trades['holding_period'].max()
        min_holding_period = exit_trades['holding_period'].min()
    else:
        avg_holding_period = 0
        max_holding_period = 0
        min_holding_period = 0

    # Generate visualizations

    # 1. Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_results['equity_curve'].index, backtest_results['equity_curve'].values, label='Portfolio Value')
    plt.title(f'Equity Curve - {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
    plt.close()

    # 2. Drawdown Chart
    equity_series = backtest_results['equity_curve']
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100

    plt.figure(figsize=(12, 6))
    plt.plot(drawdown.index, drawdown.values, color='red', label='Drawdown %')
    plt.fill_between(drawdown.index, 0, drawdown.values, alpha=0.3, color='red')
    plt.title(f'Drawdown Chart - {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Drawdown %')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdown_chart.png'))
    plt.close()

    # 3. Trade PnL Distribution if we have trade-level data
    if 'trade_pnl' in exit_trades.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(exit_trades['trade_pnl'], bins=20, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title(f'Trade PnL Distribution - {symbol}')
        plt.xlabel('Profit/Loss')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pnl_distribution.png'))
        plt.close()

    # 4. Win/Loss Pie Chart
    plt.figure(figsize=(10, 8))
    plt.pie([wins, losses],
            labels=['Profitable', 'Losing'],
            autopct='%1.1f%%',
            colors=['green', 'red'],
            startangle=90)
    plt.title(f'Win/Loss Distribution - {symbol}')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_loss_pie.png'))
    plt.close()

    # 5. Holding Period Distribution if we have the data
    if 'holding_period' in exit_trades.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(exit_trades['holding_period'], bins=20, kde=True)
        plt.title(f'Trade Holding Period Distribution - {symbol} (in hours)')
        plt.xlabel('Holding Period (hours)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'holding_period.png'))
        plt.close()

    # 6. Monthly Returns if we have the data
    if isinstance(backtest_results['equity_curve'].index[0], pd.Timestamp):
        # Calculate monthly returns
        monthly_returns = backtest_results['equity_curve'].resample('M').last().pct_change().dropna() * 100

        if len(monthly_returns) > 0:
            plt.figure(figsize=(12, 6))
            monthly_returns.plot(kind='bar', color=np.where(monthly_returns >= 0, 'green', 'red'))
            plt.title(f'Monthly Returns - {symbol}')
            plt.xlabel('Month')
            plt.ylabel('Return (%)')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'monthly_returns.png'))
            plt.close()

    # 7. Create summary text file
    with open(os.path.join(output_dir, 'performance_summary.txt'), 'w') as f:
        f.write(f"Performance Summary for {symbol}\n")
        f.write(f"=============================================\n\n")

        start_date = backtest_results['equity_curve'].index[0] if isinstance(backtest_results['equity_curve'].index[0], pd.Timestamp) else "N/A"
        end_date = backtest_results['equity_curve'].index[-1] if isinstance(backtest_results['equity_curve'].index[-1], pd.Timestamp) else "N/A"

        f.write(f"Trading Period: {start_date} to {end_date}\n\n")
        f.write(f"Initial Capital: ${initial_capital:.2f}\n")
        f.write(f"Final Capital: ${final_capital:.2f}\n")
        f.write(f"Total Return: {total_return:.2f}%\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
        f.write(f"Max Drawdown: {max_drawdown:.2f}%\n\n")
        f.write(f"Total Trades: {total_trades}\n")
        f.write(f"Profitable Trades: {wins} ({win_rate:.2f}%)\n")
        f.write(f"Losing Trades: {losses} ({100 - win_rate:.2f}%)\n")
        f.write(f"Profit Factor: {profit_factor:.2f}\n\n")
        f.write(f"Average Trade P&L: ${avg_profit:.2f}\n")
        f.write(f"Average Winning Trade: ${avg_win:.2f}\n")
        f.write(f"Average Losing Trade: ${avg_loss:.2f}\n")

        if 'holding_period' in exit_trades.columns:
            f.write(f"Average Holding Period: {avg_holding_period:.2f} hours\n")
            f.write(f"Min Holding Period: {min_holding_period:.2f} hours\n")
            f.write(f"Max Holding Period: {max_holding_period:.2f} hours\n")

    # 8. Create HTML dashboard
    html_path = os.path.join(output_dir, 'dashboard.html')

    # HTML content (abbreviated for brevity)
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Trading Performance Dashboard - {symbol}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .metrics-panel {{
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }}
        .metric-box {{
            width: 22%;
            padding: 15px;
            text-align: center;
            background-color: #f9f9f9;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
            color: #2c3e50;
        }}
        .positive {{
            color: green;
        }}
        .negative {{
            color: red;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        .chart-container {{
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }}
        .chart-title {{
            font-size: 18px;
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        .chart-row {{
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
        }}
        .chart-box {{
            width: 48%;
            margin-bottom: 20px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading Performance Dashboard</h1>
        <p>{symbol} | {start_date} to {end_date}</p>
    </div>

    <div class="container">
        <div class="metrics-panel">
            <div class="metric-box">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {('positive' if total_return >= 0 else 'negative')}">{total_return:.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Final Capital</div>
                <div class="metric-value">${final_capital:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{sharpe_ratio:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{max_drawdown:.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{win_rate:.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{total_trades}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{profit_factor:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Avg Trade P&L</div>
                <div class="metric-value {('positive' if avg_profit >= 0 else 'negative')}">${avg_profit:.2f}</div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Equity Curve</div>
            <img src="equity_curve.png" alt="Equity Curve" />
        </div>

        <div class="chart-row">
            <div class="chart-box">
                <div class="chart-container">
                    <div class="chart-title">Drawdown</div>
                    <img src="drawdown_chart.png" alt="Drawdown Chart" />
                </div>
            </div>
            <div class="chart-box">
                <div class="chart-container">
                    <div class="chart-title">Win/Loss Distribution</div>
                    <img src="win_loss_pie.png" alt="Win/Loss Distribution" />
                </div>
            </div>
        </div>

        <div class="chart-row">
            <div class="chart-box">
                <div class="chart-container">
                    <div class="chart-title">PnL Distribution</div>
                    <img src="pnl_distribution.png" alt="P&L Distribution" />
                </div>
            </div>
            <div class="chart-box">
                <div class="chart-container">
                    <div class="chart-title">Monthly Returns</div>
                    <img src="monthly_returns.png" alt="Monthly Returns" />
                </div>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="chart-container">
            <div class="chart-title">Trade History</div>
            <p>A detailed CSV file with all trades has been saved at: <code>{csv_path}</code></p>
            <p>You can open this file in Excel or any spreadsheet software for further analysis.</p>
        </div>
    </div>
</body>
</html>
"""

    with open(html_path, 'w') as f:
        f.write(html_content)

    logger.info(f"Performance dashboard created at: {html_path}")

    return html_path