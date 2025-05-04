"""
Order Manager for Live Trading

This module manages trading orders, positions, and executes trading strategies
based on model predictions.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import time
import json

from live.exchange_connector import ExchangeConnector
from live.risk_manager import RiskManager

# Setup logger
logger = logging.getLogger(__name__)


class OrderManager:
    """
    Manager for handling orders and positions in live trading
    """

    def __init__(self, exchange: ExchangeConnector, risk_manager: RiskManager,
                 symbol: str, update_interval: int = 60):
        """
        Initialize the order manager

        Args:
            exchange: Exchange connector instance
            risk_manager: Risk manager instance
            symbol: Trading symbol
            update_interval: Interval in seconds for status updates
        """
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.symbol = symbol
        self.update_interval = update_interval

        # Initialize state
        self.open_positions = []
        self.pending_orders = []
        self.trades_history = []
        self.last_update_time = datetime.now()

        # Get initial account state
        self._update_account_state()

        logger.info(f"Initialized OrderManager for {symbol}")

    def _update_account_state(self):
        """Update internal state with latest account information"""
        try:
            # Get open orders
            self.pending_orders = self.exchange.get_open_orders(self.symbol)

            # Get current positions (implementation depends on exchange)
            # For spot trading, we check balances
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['lastPrice']) if 'lastPrice' in ticker else float(ticker['last'])

            # Parse symbol to get base and quote assets
            if '/' in self.symbol:
                base_asset, quote_asset = self.symbol.split('/')
            else:
                # For symbols like BTCUSDT, extract the base and quote
                for quote in ['USDT', 'USD', 'BUSD', 'USDC', 'BTC', 'ETH']:
                    if self.symbol.endswith(quote):
                        base_asset = self.symbol[:-len(quote)]
                        quote_asset = quote
                        break
                else:
                    base_asset = self.symbol[:-4]  # Assume last 4 chars are quote
                    quote_asset = self.symbol[-4:]

            # Get balances
            base_balance = self.exchange.get_balance(base_asset)
            quote_balance = self.exchange.get_balance(quote_asset)

            # Check if we have an open position
            if base_balance > self.risk_manager.min_position_size:
                # We have a long position
                self.open_positions = [{
                    'type': 'long',
                    'symbol': self.symbol,
                    'size': base_balance,
                    'entry_price': self.risk_manager.last_entry_price or current_price,
                    'current_price': current_price,
                    'unrealized_pnl': (current_price - (
                                self.risk_manager.last_entry_price or current_price)) * base_balance
                }]
            else:
                self.open_positions = []

            self.last_update_time = datetime.now()

        except Exception as e:
            logger.error(f"Error updating account state: {str(e)}")

    def process_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a model prediction and execute trading actions

        Args:
            prediction: Prediction dictionary from model

        Returns:
            Dictionary with action taken and result
        """
        # Update account state
        self._update_account_state()

        # Check if we're rate limited
        time_since_last_update = (datetime.now() - self.last_update_time).total_seconds()
        if time_since_last_update < self.update_interval:
            logger.debug(f"Skipping prediction processing, last update was {time_since_last_update}s ago")
            return {
                'action': 'SKIP',
                'reason': 'too_frequent',
                'time_to_next': self.update_interval - time_since_last_update
            }

        # Extract trading signal from prediction
        signal = self._extract_signal(prediction)

        # Get current ticker
        ticker = self.exchange.get_ticker(self.symbol)
        current_price = float(ticker['lastPrice']) if 'lastPrice' in ticker else float(ticker['last'])

        logger.info(f"Processing signal: {signal} at price {current_price}")

        # Check risk parameters
        if not self.risk_manager.can_trade():
            logger.warning("Risk manager prevents trading at this time")
            return {
                'action': 'SKIP',
                'reason': 'risk_limits',
                'signal': signal
            }

        # Check if we have pending orders
        if self.pending_orders:
            logger.info(f"Have {len(self.pending_orders)} pending orders, processing them first")
            self._handle_pending_orders()

        # Execute trading logic
        if signal == 'BUY':
            return self._handle_buy_signal(current_price)
        elif signal == 'SELL':
            return self._handle_sell_signal(current_price)
        elif signal == 'CLOSE':
            return self._handle_close_signal(current_price)
        else:  # HOLD or unknown
            return {
                'action': 'HOLD',
                'reason': 'model_signal',
                'signal': signal
            }

    def _extract_signal(self, prediction: Dict[str, Any]) -> str:
        """
        Extract trading signal from model prediction

        Args:
            prediction: Model prediction dictionary

        Returns:
            Trading signal (BUY, SELL, CLOSE, HOLD)
        """
        # Check if prediction contains an explicit action
        if 'action' in prediction:
            return prediction['action']

        # Check if prediction contains a signal
        if 'signal' in prediction:
            return prediction['signal']

        # For LSTM predictions predicting price
        if 'predicted_price' in prediction and 'current_price' in prediction:
            predicted_price = prediction['predicted_price']
            current_price = prediction['current_price']

            # Simple threshold strategy
            threshold = 0.001  # 0.1% price movement threshold

            if predicted_price > current_price * (1 + threshold):
                return 'BUY'
            elif predicted_price < current_price * (1 - threshold):
                return 'SELL'
            else:
                return 'HOLD'

        # For predictions with predicted return
        if 'predicted_return_pct' in prediction:
            predicted_return = prediction['predicted_return_pct']

            # Simple threshold strategy
            threshold = 0.5  # 0.5% return threshold

            if predicted_return > threshold:
                return 'BUY'
            elif predicted_return < -threshold:
                return 'SELL'
            else:
                return 'HOLD'

        # For predictions with directional probability
        if 'predicted_direction' in prediction:
            direction = prediction['predicted_direction']

            if direction > 0.6:  # Require 60% confidence for buy
                return 'BUY'
            elif direction < 0.4:  # Require 60% confidence for sell
                return 'SELL'
            else:
                return 'HOLD'

        # Default case
        logger.warning(f"Could not extract signal from prediction: {prediction}")
        return 'HOLD'

    def _handle_buy_signal(self, current_price: float) -> Dict[str, Any]:
        """
        Handle a BUY signal from the model

        Args:
            current_price: Current market price

        Returns:
            Action result
        """
        # Check if we already have a long position
        if any(pos['type'] == 'long' for pos in self.open_positions):
            logger.info("Already have a long position, skipping buy")
            return {
                'action': 'SKIP',
                'reason': 'position_exists',
                'signal': 'BUY'
            }

        # Calculate position size
        quote_balance = self.exchange.get_balance(self.symbol[-4:])  # Last 4 chars assumed to be quote asset
        position_size = self.risk_manager.calculate_position_size(quote_balance, current_price)

        if position_size <= self.risk_manager.min_position_size:
            logger.warning(
                f"Position size {position_size} is too small, minimum is {self.risk_manager.min_position_size}")
            return {
                'action': 'SKIP',
                'reason': 'position_too_small',
                'signal': 'BUY'
            }

        # Convert quote amount to base amount
        base_quantity = position_size / current_price

        # Place market buy order
        try:
            order = self.exchange.place_order(
                symbol=self.symbol,
                order_type='MARKET',
                side='BUY',
                quantity=base_quantity
            )

            # Record trade
            trade = {
                'type': 'long',
                'entry_date': datetime.now(),
                'entry_price': current_price,
                'size': position_size,
                'order_id': order['orderId'] if 'orderId' in order else order['order_id'],
                'status': 'open'
            }

            # Save entry price for future reference
            self.risk_manager.last_entry_price = current_price

            # Add to trades history
            self.trades_history.append(trade)

            logger.info(f"Executed BUY order at {current_price}: {order}")

            return {
                'action': 'BUY',
                'price': current_price,
                'size': position_size,
                'order': order
            }

        except Exception as e:
            logger.error(f"Error executing buy order: {str(e)}")
            return {
                'action': 'ERROR',
                'reason': str(e),
                'signal': 'BUY'
            }

    def _handle_sell_signal(self, current_price: float) -> Dict[str, Any]:
        """
        Handle a SELL signal from the model

        Args:
            current_price: Current market price

        Returns:
            Action result
        """
        # For spot trading, sell means closing a long position
        if any(pos['type'] == 'long' for pos in self.open_positions):
            return self._handle_close_signal(current_price)

        # If we don't support short selling, just return
        if not self.risk_manager.allow_short:
            logger.info("Short selling not allowed, ignoring sell signal")
            return {
                'action': 'SKIP',
                'reason': 'no_short_selling',
                'signal': 'SELL'
            }

        # If we reach here, we can open a short position
        # This would be implemented for margin/futures trading
        logger.warning("Short selling not implemented yet")
        return {
            'action': 'SKIP',
            'reason': 'not_implemented',
            'signal': 'SELL'
        }

    def _handle_close_signal(self, current_price: float) -> Dict[str, Any]:
        """
        Handle a CLOSE signal from the model

        Args:
            current_price: Current market price

        Returns:
            Action result
        """
        # Check if we have any positions to close
        if not self.open_positions:
            logger.info("No open positions to close")
            return {
                'action': 'SKIP',
                'reason': 'no_position',
                'signal': 'CLOSE'
            }

        # Close all positions
        results = []

        for position in self.open_positions:
            try:
                if position['type'] == 'long':
                    # Get current balance of base asset
                    symbol_parts = self.symbol.split('/') if '/' in self.symbol else [self.symbol[:-4],
                                                                                      self.symbol[-4:]]
                    base_asset = symbol_parts[0]
                    base_balance = self.exchange.get_balance(base_asset)

                    # Place market sell order
                    order = self.exchange.place_order(
                        symbol=self.symbol,
                        order_type='MARKET',
                        side='SELL',
                        quantity=base_balance
                    )

                    # Calculate profit/loss
                    entry_price = position.get('entry_price', self.risk_manager.last_entry_price or current_price)
                    pnl = (current_price - entry_price) * base_balance
                    pnl_pct = (current_price / entry_price - 1) * 100

                    # Record trade
                    for trade in self.trades_history:
                        if trade.get('status') == 'open' and trade.get('type') == 'long':
                            trade['exit_date'] = datetime.now()
                            trade['exit_price'] = current_price
                            trade['pnl'] = pnl
                            trade['pnl_pct'] = pnl_pct
                            trade['status'] = 'closed'

                    logger.info(f"Executed SELL order to close long position at {current_price}: {order}")

                    results.append({
                        'action': 'CLOSE_LONG',
                        'price': current_price,
                        'size': base_balance,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'order': order
                    })

                elif position['type'] == 'short':
                    # Short closing would be implemented for margin/futures trading
                    logger.warning("Closing short positions not implemented yet")

            except Exception as e:
                logger.error(f"Error closing position: {str(e)}")
                results.append({
                    'action': 'ERROR',
                    'reason': str(e),
                    'signal': 'CLOSE'
                })

        # Clear positions
        self.open_positions = []

        return {
            'action': 'CLOSE',
            'results': results
        }

    def _handle_pending_orders(self) -> None:
        """Process any pending orders"""
        for order in self.pending_orders:
            try:
                # Check order status
                order_id = order['orderId'] if 'orderId' in order else order['order_id']
                status = self.exchange.get_order_status(self.symbol, order_id)

                # If order is filled, update trades history
                if status.get('status') == 'FILLED':
                    for trade in self.trades_history:
                        if trade.get('order_id') == order_id:
                            trade['status'] = 'filled'
                            logger.info(f"Order {order_id} is now filled")

                # If order is hanging for too long, cancel it
                elif self._should_cancel_order(order):
                    self.exchange.cancel_order(self.symbol, order_id)
                    logger.info(f"Cancelled hanging order {order_id}")

            except Exception as e:
                logger.error(f"Error handling pending order: {str(e)}")

    def _should_cancel_order(self, order) -> bool:
        """
        Determine if an order should be cancelled

        Args:
            order: Order object

        Returns:
            True if order should be cancelled
        """
        # Get order age
        create_time = order.get('time', order.get('created_at'))
        if isinstance(create_time, str):
            if create_time.isdigit():
                create_time = datetime.fromtimestamp(int(create_time) / 1000)
            else:
                create_time = datetime.fromisoformat(create_time.replace('Z', '+00:00'))

        if not create_time:
            return False

        # Cancel if order is older than 5 minutes
        order_age = (datetime.now() - create_time).total_seconds()
        return order_age > 300  # 5 minutes

    def calculate_equity(self) -> float:
        """
        Calculate current account equity

        Returns:
            Current equity value
        """
        try:
            # Get quote asset balance
            quote_asset = self.symbol[-4:]
            quote_balance = self.exchange.get_balance(quote_asset)

            # Get base asset balance
            base_asset = self.symbol[:-4]
            base_balance = self.exchange.get_balance(base_asset)

            # Get current price
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = float(ticker['lastPrice']) if 'lastPrice' in ticker else float(ticker['last'])

            # Calculate total equity
            equity = quote_balance + (base_balance * current_price)

            return equity

        except Exception as e:
            logger.error(f"Error calculating equity: {str(e)}")
            return 0.0