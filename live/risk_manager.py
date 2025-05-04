"""
Risk Manager for Live Trading

This module handles risk management for live trading, including position sizing,
maximum drawdown protection, and other risk controls.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import time

# Setup logger
logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk manager for controlling trading risk parameters
    """

    def __init__(self, initial_capital: float, max_risk_per_trade: float = 0.02,
                 max_open_positions: int = 1, max_daily_trades: int = 10,
                 max_drawdown_pct: float = 0.2, position_sizing_method: str = 'fixed_percent',
                 allow_short: bool = False, min_position_size: float = 10.0):
        """
        Initialize the risk manager

        Args:
            initial_capital: Starting capital
            max_risk_per_trade: Maximum risk per trade as percentage of capital (e.g., 0.02 = 2%)
            max_open_positions: Maximum number of concurrent open positions
            max_daily_trades: Maximum number of trades per day
            max_drawdown_pct: Maximum drawdown allowed as percentage (e.g., 0.2 = 20%)
            position_sizing_method: Method for calculating position size
            allow_short: Whether to allow short positions
            min_position_size: Minimum position size in quote currency
        """
        self.initial_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_open_positions = max_open_positions
        self.max_daily_trades = max_daily_trades
        self.max_drawdown_pct = max_drawdown_pct
        self.position_sizing_method = position_sizing_method
        self.allow_short = allow_short
        self.min_position_size = min_position_size

        # State variables
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.open_positions_count = 0
        self.daily_trades_count = 0
        self.last_trade_time = None
        self.last_entry_price = None

        # Trading session data
        self.trades_history = []
        self.equity_history = [{'timestamp': datetime.now(), 'equity': initial_capital}]
        self.session_start_time = datetime.now()

        # Risk controls
        self.is_active = True  # Whether trading is allowed

        logger.info(f"Initialized RiskManager with {initial_capital:.2f} initial capital")

    def update_capital(self, new_capital: float) -> None:
        """
        Update current capital amount

        Args:
            new_capital: New capital amount
        """
        self.current_capital = new_capital

        # Update peak capital if needed
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

        # Add to equity history
        self.equity_history.append({
            'timestamp': datetime.now(),
            'equity': new_capital
        })

        # Check drawdown
        self._check_drawdown()

        logger.debug(f"Capital updated to {new_capital:.2f}")

    def calculate_position_size(self, available_capital: float,
                                current_price: float, stop_loss_pct: Optional[float] = None) -> float:
        """
        Calculate position size based on risk parameters

        Args:
            available_capital: Available capital for trading
            current_price: Current asset price
            stop_loss_pct: Stop loss percentage (optional)

        Returns:
            Position size in quote currency
        """
        if not self.is_active:
            logger.warning("Risk manager is not active, returning zero position size")
            return 0.0

        # Default position sizing based on configured method
        if self.position_sizing_method == 'fixed_percent':
            # Simply allocate a percentage of capital
            position_size = available_capital * self.max_risk_per_trade * 10  # multiply by 10 since risk is usually 10% of allocation

        elif self.position_sizing_method == 'risk_based':
            # If stop loss is provided, calculate based on risk amount and stop distance
            if stop_loss_pct:
                # Formula: risk_amount / stop_distance
                risk_amount = available_capital * self.max_risk_per_trade
                stop_distance = stop_loss_pct
                position_size = risk_amount / stop_distance
            else:
                # Use default stop of 2%
                risk_amount = available_capital * self.max_risk_per_trade
                position_size = risk_amount / 0.02

        elif self.position_sizing_method == 'kelly':
            # Kelly criterion (simplified version)
            # This would need win rate and average win/loss data
            win_rate = 0.55  # Default assumption
            avg_win = 0.03  # Default assumption
            avg_loss = 0.02  # Default assumption

            kelly_fraction = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
            kelly_fraction = max(0, min(kelly_fraction, 0.2))  # Limit to 20% max

            position_size = available_capital * kelly_fraction

        else:
            # Default to fixed amount
            position_size = min(available_capital * 0.1, 1000)  # 10% of capital or $1000, whichever is less

        # Ensure position size doesn't exceed available capital
        position_size = min(position_size, available_capital * 0.95)  # Use 95% of available capital at most

        # Apply minimum position size check
        if position_size < self.min_position_size:
            logger.warning(f"Calculated position size {position_size:.2f} is below minimum {self.min_position_size}")
            return 0.0

        logger.info(f"Calculated position size: {position_size:.2f}")
        return position_size

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """
        Record a completed trade

        Args:
            trade: Trade information dictionary
        """
        self.trades_history.append(trade)

        # Update trade count and time
        self.daily_trades_count += 1
        self.last_trade_time = datetime.now()

        # If trade has PnL, update capital
        if 'pnl' in trade:
            self.update_capital(self.current_capital + trade['pnl'])

        # Reset daily trades count at midnight
        current_date = datetime.now().date()
        last_trade_date = self.last_trade_time.date()
        if current_date > last_trade_date:
            self.daily_trades_count = 0
            logger.info("Reset daily trades count for new day")

        logger.debug(f"Recorded trade: {trade}")

    def update_position_count(self, count: int) -> None:
        """
        Update the count of open positions

        Args:
            count: Number of open positions
        """
        self.open_positions_count = count
        logger.debug(f"Open positions count updated to {count}")

    def can_trade(self) -> bool:
        """
        Check if trading is allowed based on risk parameters

        Returns:
            True if trading is allowed
        """
        # Check if risk manager is active
        if not self.is_active:
            logger.info("Risk manager is not active, trading disallowed")
            return False

        # Check max positions
        if self.open_positions_count >= self.max_open_positions:
            logger.info(f"Max open positions ({self.max_open_positions}) reached, trading disallowed")
            return False

        # Check max daily trades
        if self.daily_trades_count >= self.max_daily_trades:
            logger.info(f"Max daily trades ({self.max_daily_trades}) reached, trading disallowed")
            return False

        return True

    def _check_drawdown(self) -> None:
        """Check for drawdown and apply safety measures if needed"""
        # Calculate current drawdown
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

            # Log significant drawdowns
            if drawdown > 0.05:  # 5%
                logger.warning(f"Current drawdown: {drawdown:.2%}")

            # Check against max drawdown
            if drawdown > self.max_drawdown_pct:
                logger.critical(f"Max drawdown ({self.max_drawdown_pct:.2%}) exceeded: {drawdown:.2%}")
                self.is_active = False

                # Additional safety measures could be triggered here
                # e.g., sending alerts, closing all positions, etc.

    def reset_daily_limits(self) -> None:
        """Reset daily trading limits"""
        self.daily_trades_count = 0
        logger.info("Daily trading limits reset")

    def activate(self) -> None:
        """Activate the risk manager"""
        self.is_active = True
        logger.info("Risk manager activated")

    def deactivate(self) -> None:
        """Deactivate the risk manager to prevent trading"""
        self.is_active = False
        logger.info("Risk manager deactivated - trading halted")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current risk statistics

        Returns:
            Dictionary with risk statistics
        """
        # Calculate current drawdown
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0

        # Calculate win rate
        winning_trades = sum(1 for trade in self.trades_history if trade.get('pnl', 0) > 0)
        total_trades = len(self.trades_history)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate average P&L
        if total_trades > 0:
            avg_pnl = sum(trade.get('pnl', 0) for trade in self.trades_history) / total_trades
            avg_pnl_pct = sum(trade.get('pnl_pct', 0) for trade in self.trades_history) / total_trades
        else:
            avg_pnl = 0
            avg_pnl_pct = 0

        # Calculate session duration
        session_duration = datetime.now() - self.session_start_time

        return {
            'is_active': self.is_active,
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'drawdown': drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'avg_pnl': avg_pnl,
            'avg_pnl_pct': avg_pnl_pct,
            'open_positions': self.open_positions_count,
            'daily_trades': self.daily_trades_count,
            'max_daily_trades': self.max_daily_trades,
            'session_duration': session_duration.total_seconds() / 3600,  # hours
            'session_start': self.session_start_time.isoformat()
        }