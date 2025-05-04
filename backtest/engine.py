"""
Backtesting Engine for TensorTrade

This module provides a framework for backtesting trading strategies
using historical data with the trained models.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Engine for backtesting trading strategies with trained models
    """

    def __init__(self, model: Any, initial_capital: float = 10000.0,
                 commission: float = 0.001, risk_per_trade: float = 0.02,
                 stop_loss_pct: float = 0.02, take_profit_pct: float = 0.04,
                 max_open_positions: int = 1):
        """
        Initialize the backtest engine

        Args:
            model: Trained model (LSTM or RL)
            initial_capital: Starting capital
            commission: Trading commission as a decimal (e.g., 0.001 = 0.1%)
            risk_per_trade: Percentage of capital to risk per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_open_positions: Maximum number of open positions
        """
        self.model = model
        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_open_positions = max_open_positions

        # State variables
        self.capital = initial_capital
        self.equity_curve = []
        self.positions = []
        self.trades = []
        self.trades_history = []
        self.data = None  # Store the data reference

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the backtest on historical data

        Args:
            data: DataFrame with price data

        Returns:
            Dictionary with backtest results
        """
        # Store the data reference
        self.data = data

        # Log the full data range before any filtering
        logger.info(f"Full data range: {data.index[0]} to {data.index[-1]}, total rows: {len(data)}")

        # Reset state
        self.capital = self.initial_capital
        self.equity_curve = []
        self.positions = []
        self.trades = []
        self.trades_history = []

        # Minimum required data for the model
        min_data_length = getattr(self.model, 'window_size', 60)
        logger.info(f"Model requires minimum {min_data_length} data points")

        # Verify if we have enough data
        if len(data) < min_data_length:
            logger.error(f"Not enough data points: {len(data)} available, {min_data_length} required")
            return {'metrics': {}, 'equity_curve': pd.Series(), 'position_history': pd.Series(),
                    'trades_history': [], 'data': data}

        # Initialize equity and position tracking
        equity_history = []
        position_history = []

        # For each date in the data, generate an equity value
        for i in tqdm(range(len(data)), desc="Backtesting"):
            current_date = data.index[i]
            current_price = data.iloc[i]['close']

            # If we don't have enough data yet for prediction, just record initial capital
            if i < min_data_length:
                equity_history.append(self.initial_capital)
                position_history.append(0)  # No position
                continue

            # Prepare data for prediction - use data up to current point
            prediction_data = data.iloc[:i + 1]

            # Get prediction from model
            model_type = self.model.__class__.__name__
            if model_type == 'LSTMModel':
                # Get price prediction or direction
                prediction = self.model.predict(prediction_data)

                # Generate trading signal based on prediction
                if self.model.config.get('target_type', 'price') == 'price':
                    signal = 1 if prediction > current_price else -1
                elif self.model.config.get('target_type', 'price') == 'return':
                    signal = 1 if prediction > 0 else -1
                else:
                    signal = 1 if prediction > 0.5 else -1

            elif model_type == 'RLModel':
                # Get trading decision directly
                decision = self.model.predict(prediction_data)

                if decision['action'] == 'BUY':
                    signal = 1
                elif decision['action'] == 'SELL':
                    signal = -1
                elif decision['action'] == 'CLOSE':
                    signal = 0
                else:  # HOLD
                    signal = None

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Update open positions with current market price
            self._update_positions(current_price, current_date)

            # Process signal
            if signal is not None:
                self._process_signal(signal, current_price, current_date)

            # Update equity and position for this date
            current_equity = self._calculate_equity(current_price)
            equity_history.append(current_equity)

            current_position = 1 if any(p['type'] == 'long' for p in self.positions) else -1 if any(
                p['type'] == 'short' for p in self.positions) else 0
            position_history.append(current_position)

        # Close all positions at the end of the backtest
        final_price = data.iloc[-1]['close']
        final_date = data.index[-1]
        self._close_all_positions(final_price, final_date)

        # Create proper Series with date index for the full data range
        self.equity_curve = pd.Series(equity_history, index=data.index)
        self.position_history = pd.Series(position_history, index=data.index)

        logger.info(f"Equity curve generated from {self.equity_curve.index[0]} to {self.equity_curve.index[-1]}")

        # Calculate backtest metrics
        metrics = self._calculate_metrics(data, equity_history, position_history)

        # Create result object
        results = {
            'metrics': metrics,
            'equity_curve': self.equity_curve,
            'position_history': self.position_history,
            'trades_history': self.trades_history,
            'data': data
        }

        logger.info(f"Backtest completed. Final equity: {metrics['final_equity']:.2f}")

        return results

    def _process_signal(self, signal: int, current_price: float, current_date: datetime) -> None:
        """
        Process a trading signal - modified for spot trading only (no shorts)

        Args:
            signal: Trading signal (1 = buy, -1 = sell, 0 = close)
            current_price: Current asset price
            current_date: Current date
        """
        # Check if we have reached max positions
        current_positions_count = len(self.positions)

        if signal == 1:  # Buy signal (go long)
            # Open a new long position if we have capacity and no existing positions
            if current_positions_count == 0:
                self._open_position('long', current_price, current_date)

        elif signal == -1 or signal == 0:  # Sell signal or Close signal
            # Close any existing long positions
            for pos in list(self.positions):
                if pos['type'] == 'long':
                    self._close_position(pos, current_price, current_date)

    def _open_position(self, position_type: str, price: float, date: datetime) -> None:
        """
        Open a new trading position

        Args:
            position_type: Type of position ('long' or 'short')
            price: Entry price
            date: Entry date
        """
        # Calculate position size based on risk
        position_size = self.capital * self.risk_per_trade / self.stop_loss_pct

        # Ensure position size doesn't exceed capital
        if position_size > self.capital:
            position_size = self.capital * 0.95  # Use 95% of capital at most

        # Calculate entry price with commission
        if position_type == 'long':
            entry_price = price * (1 + self.commission)
        else:  # short
            entry_price = price * (1 - self.commission)

        # Calculate stop loss and take profit levels
        if position_type == 'long':
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # short
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)

        # Create position object
        position = {
            'type': position_type,
            'entry_price': entry_price,
            'entry_date': date,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

        # Update capital
        self.capital -= position_size

        # Add to positions list
        self.positions.append(position)

        logger.debug(f"Opened {position_type} position: {position}")

    def _close_position(self, position: Dict[str, Any], price: float, date: datetime) -> None:
        """
        Close a trading position

        Args:
            position: Position object
            price: Exit price
            date: Exit date
        """
        # Calculate exit price with commission
        if position['type'] == 'long':
            exit_price = price * (1 - self.commission)
        else:  # short
            exit_price = price * (1 + self.commission)

        # Calculate profit/loss
        if position['type'] == 'long':
            pnl = position['size'] * (exit_price - position['entry_price']) / position['entry_price']
        else:  # short
            pnl = position['size'] * (position['entry_price'] - exit_price) / position['entry_price']

        # Create trade record
        trade = {
            'type': position['type'],
            'entry_date': position['entry_date'],
            'exit_date': date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'pnl': pnl,
            'pnl_pct': pnl / position['size'] * 100
        }

        # Update capital
        self.capital += position['size'] + pnl

        # Add to trades history
        self.trades_history.append(trade)

        # Remove from positions list
        self.positions.remove(position)

        logger.debug(f"Closed {position['type']} position: {trade}")

    def _update_positions(self, current_price: float, current_date: datetime) -> None:
        """
        Update open positions with current market price

        Args:
            current_price: Current market price
            current_date: Current date
        """
        for position in list(self.positions):
            # Check if stop loss or take profit hit
            if position['type'] == 'long':
                if current_price <= position['stop_loss']:
                    # Stop loss hit
                    logger.debug(f"Stop loss hit for long position")
                    self._close_position(position, position['stop_loss'], current_date)
                elif current_price >= position['take_profit']:
                    # Take profit hit
                    logger.debug(f"Take profit hit for long position")
                    self._close_position(position, position['take_profit'], current_date)
            else:  # short
                if current_price >= position['stop_loss']:
                    # Stop loss hit
                    logger.debug(f"Stop loss hit for short position")
                    self._close_position(position, position['stop_loss'], current_date)
                elif current_price <= position['take_profit']:
                    # Take profit hit
                    logger.debug(f"Take profit hit for short position")
                    self._close_position(position, position['take_profit'], current_date)

    def _close_all_positions(self, price: float, date: datetime) -> None:
        """
        Close all open positions

        Args:
            price: Exit price
            date: Exit date
        """
        for position in list(self.positions):
            self._close_position(position, price, date)

    def _calculate_equity(self, current_price: float) -> float:
        """
        Calculate current equity

        Args:
            current_price: Current market price

        Returns:
            Current equity value
        """
        equity = self.capital

        for position in self.positions:
            if position['type'] == 'long':
                pnl = position['size'] * (current_price - position['entry_price']) / position['entry_price']
            else:  # short
                pnl = position['size'] * (position['entry_price'] - current_price) / position['entry_price']

            equity += position['size'] + pnl

        return equity

    def _calculate_metrics(self, data: pd.DataFrame, equity_history: List[float],
                           position_history: List[int]) -> Dict[str, Any]:
        """
        Calculate backtest performance metrics

        Args:
            data: Price data
            equity_history: List of equity values
            position_history: List of position values

        Returns:
            Dictionary with metrics
        """
        # Convert to numpy arrays
        equity = np.array(equity_history)

        # Basic metrics
        total_return = (equity[-1] - equity[0]) / equity[0] * 100
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        max_drawdown = np.min(drawdown)

        # Calculate daily returns
        equity_series = pd.Series(equity, index=data.index[len(data) - len(equity):])
        daily_returns = equity_series.pct_change().dropna()

        # Annualize returns and risk
        days = (data.index[-1] - data.index[0]).days
        years = days / 365.25

        annualized_return = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100
        annualized_volatility = daily_returns.std() * np.sqrt(252) * 100

        # Sharpe and Sortino ratios
        risk_free_rate = 0.02  # 2% risk-free rate
        excess_return = annualized_return / 100 - risk_free_rate
        sharpe_ratio = excess_return / (annualized_volatility / 100) if annualized_volatility != 0 else 0

        # Sortino ratio (downside risk only)
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) * 100
        sortino_ratio = excess_return / (downside_deviation / 100) if downside_deviation != 0 else 0

        # Win rate and profit factor
        wins = sum(1 for trade in self.trades_history if trade['pnl'] > 0)
        losses = sum(1 for trade in self.trades_history if trade['pnl'] <= 0)
        total_trades = len(self.trades_history)

        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        gross_profit = sum(trade['pnl'] for trade in self.trades_history if trade['pnl'] > 0)
        gross_loss = sum(trade['pnl'] for trade in self.trades_history if trade['pnl'] <= 0)
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

        # Average trade metrics
        avg_profit = sum(trade['pnl'] for trade in self.trades_history) / total_trades if total_trades > 0 else 0
        avg_win = gross_profit / wins if wins > 0 else 0
        avg_loss = gross_loss / losses if losses > 0 else 0

        # Profit per day
        profit_per_day = (equity[-1] - equity[0]) / days if days > 0 else 0

        # Create metrics dictionary
        metrics = {
            'initial_equity': self.initial_capital,
            'final_equity': equity[-1],
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'annualized_volatility_pct': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'win_count': wins,
            'loss_count': losses,
            'avg_profit': avg_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_per_day': profit_per_day,
            'test_period_days': days
        }

        return metrics

    def generate_trade_history_csv(self, symbol: str, output_path: str = None) -> str:
        """
        Generate a detailed CSV file with trade history

        Args:
            symbol: Trading symbol
            output_path: Path to save the CSV file (optional)

        Returns:
            Path to the generated CSV file
        """
        try:
            # Import the logger module (import here to avoid circular imports)
            from utils.backtest_logger import generate_trade_history_csv

            # Create results dictionary with necessary fields
            results = {
                'trades_history': self.trades_history,
                'equity_curve': self.equity_curve,
                'metrics': self._calculate_metrics(self.data, self.equity_curve, self.position_history)
            }

            # Generate and return the CSV path
            return generate_trade_history_csv(results, symbol, output_path)

        except Exception as e:
            logger.error(f"Error generating trade history CSV: {str(e)}")
            raise

    def generate_performance_dashboard(self, symbol: str, output_dir: str = None) -> str:
        """
        Generate a performance dashboard from backtest results

        Args:
            symbol: Trading symbol
            output_dir: Directory to save dashboard files (optional)

        Returns:
            Path to the generated dashboard HTML
        """
        try:
            # Import the logger module (import here to avoid circular imports)
            from utils.backtest_logger import generate_performance_dashboard

            # Generate trade history CSV first
            csv_path = self.generate_trade_history_csv(symbol)

            # Create results dictionary with necessary fields
            results = {
                'trades_history': self.trades_history,
                'equity_curve': self.equity_curve,
                'metrics': self._calculate_metrics(self.data, self.equity_curve, self.position_history),
                'data': self.data
            }

            # Generate and return the dashboard path
            return generate_performance_dashboard(results, symbol, csv_path, output_dir)

        except Exception as e:
            logger.error(f"Error generating performance dashboard: {str(e)}")
            raise


def run_single_backtest(model, data, initial_capital=10000, **kwargs):
    """
    Run a single backtest and return results

    Args:
        model: Trained model
        data: Price data
        initial_capital: Starting capital
        **kwargs: Additional parameters for BacktestEngine

    Returns:
        Dictionary with backtest results
    """
    engine = BacktestEngine(
        model=model,
        initial_capital=initial_capital,
        **kwargs
    )

    results = engine.run(data)
    return results


def run_parameter_sweep(model, data, param_grid, initial_capital=10000):
    """
    Run backtest with different parameter combinations

    Args:
        model: Trained model
        data: Price data
        param_grid: Dictionary with parameter ranges
        initial_capital: Starting capital

    Returns:
        Dictionary with results for each parameter combination
    """
    import itertools

    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))

    results = []

    for combo in tqdm(param_combinations, desc="Parameter sweep"):
        # Create parameter dictionary
        params = dict(zip(param_names, combo))

        # Run backtest
        engine = BacktestEngine(
            model=model,
            initial_capital=initial_capital,
            **params
        )

        result = engine.run(data)

        # Add parameters to result
        result['parameters'] = params
        results.append(result)

    return results


def compare_models(models, data, initial_capital=10000, **kwargs):
    """
    Compare multiple models on the same data

    Args:
        models: List of (name, model) tuples
        data: Price data
        initial_capital: Starting capital
        **kwargs: Additional parameters for BacktestEngine

    Returns:
        Dictionary with results for each model
    """
    results = {}

    for name, model in models:
        engine = BacktestEngine(
            model=model,
            initial_capital=initial_capital,
            **kwargs
        )

        result = engine.run(data)
        results[name] = result

    return results