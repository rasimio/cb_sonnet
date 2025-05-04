"""
Exchange Connector for Live Trading

This module handles connections to cryptocurrency and stock exchanges
for live trading operations.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import time
import json

# Setup logger
logger = logging.getLogger(__name__)


class ExchangeConnector:
    """
    Connector for interacting with trading exchanges
    """

    def __init__(self, exchange_id: str, api_key: str, api_secret: str,
                 testnet: bool = True, additional_params: Dict[str, Any] = None):
        """
        Initialize the exchange connector

        Args:
            exchange_id: ID of the exchange (e.g., 'binance', 'bybit')
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Whether to use the testnet/sandbox environment
            additional_params: Additional parameters for the exchange
        """
        self.exchange_id = exchange_id.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.additional_params = additional_params or {}

        # Initialize exchange client
        self.client = self._initialize_client()

        # Store open orders and positions
        self.open_orders = []

        logger.info(f"Initialized {exchange_id} connector (testnet: {testnet})")

    def _initialize_client(self) -> Any:
        """
        Initialize the exchange API client

        Returns:
            Exchange client object
        """
        if self.exchange_id == 'binance':
            try:
                from binance.client import Client
                from binance.exceptions import BinanceAPIException

                client = Client(self.api_key, self.api_secret, testnet=self.testnet)

                # Test API connection
                server_time = client.get_server_time()
                logger.info(f"Connected to Binance. Server time: {server_time}")

                return client
            except ImportError:
                logger.error("binance-python package is required for Binance integration")
                raise ImportError("Please install binance-python package: pip install python-binance")
            except Exception as e:
                logger.error(f"Failed to initialize Binance client: {str(e)}")
                raise

        elif self.exchange_id == 'bybit':
            try:
                import pybit

                endpoint = "https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com"
                client = pybit.usdt_perpetual.HTTP(
                    endpoint=endpoint,
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )

                # Test API connection
                server_time = client.server_time()
                logger.info(f"Connected to Bybit. Server time: {server_time}")

                return client
            except ImportError:
                logger.error("pybit package is required for Bybit integration")
                raise ImportError("Please install pybit package: pip install pybit")
            except Exception as e:
                logger.error(f"Failed to initialize Bybit client: {str(e)}")
                raise

        elif self.exchange_id == 'kucoin':
            try:
                from kucoin.client import Client

                client = Client(self.api_key, self.api_secret, self.additional_params.get('api_passphrase', ''))

                # Test API connection
                server_time = client.get_server_timestamp()
                logger.info(f"Connected to KuCoin. Server time: {server_time}")

                return client
            except ImportError:
                logger.error("kucoin-python package is required for KuCoin integration")
                raise ImportError("Please install kucoin-python package: pip install kucoin-python")
            except Exception as e:
                logger.error(f"Failed to initialize KuCoin client: {str(e)}")
                raise

        else:
            raise ValueError(f"Unsupported exchange: {self.exchange_id}")

    def get_latest_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Get latest market data from the exchange

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            limit: Number of candles to retrieve

        Returns:
            DataFrame with OHLCV data
        """
        try:
            if self.exchange_id == 'binance':
                # Map timeframe to Binance interval format
                timeframe_map = {
                    '1m': Client.KLINE_INTERVAL_1MINUTE,
                    '3m': Client.KLINE_INTERVAL_3MINUTE,
                    '5m': Client.KLINE_INTERVAL_5MINUTE,
                    '15m': Client.KLINE_INTERVAL_15MINUTE,
                    '30m': Client.KLINE_INTERVAL_30MINUTE,
                    '1h': Client.KLINE_INTERVAL_1HOUR,
                    '2h': Client.KLINE_INTERVAL_2HOUR,
                    '4h': Client.KLINE_INTERVAL_4HOUR,
                    '6h': Client.KLINE_INTERVAL_6HOUR,
                    '8h': Client.KLINE_INTERVAL_8HOUR,
                    '12h': Client.KLINE_INTERVAL_12HOUR,
                    '1d': Client.KLINE_INTERVAL_1DAY,
                    '3d': Client.KLINE_INTERVAL_3DAY,
                    '1w': Client.KLINE_INTERVAL_1WEEK,
                    '1M': Client.KLINE_INTERVAL_1MONTH
                }

                interval = timeframe_map.get(timeframe)
                if not interval:
                    raise ValueError(f"Unsupported timeframe for Binance: {timeframe}")

                # Fetch klines (candlestick data)
                klines = self.client.get_klines(
                    symbol=symbol.upper(),
                    interval=interval,
                    limit=limit
                )

                # Create DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])

                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_columns] = df[numeric_columns].astype(float)

                # Set timestamp as index and select OHLCV columns
                df.set_index('timestamp', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]

                return df

            elif self.exchange_id == 'bybit':
                # Map timeframe to Bybit interval format
                timeframe_map = {
                    '1m': 1,
                    '3m': 3,
                    '5m': 5,
                    '15m': 15,
                    '30m': 30,
                    '1h': 60,
                    '2h': 120,
                    '4h': 240,
                    '6h': 360,
                    '12h': 720,
                    '1d': 'D',
                    '1w': 'W',
                    '1M': 'M'
                }

                interval = timeframe_map.get(timeframe)
                if not interval:
                    raise ValueError(f"Unsupported timeframe for Bybit: {timeframe}")

                # Fetch klines
                response = self.client.query_kline(
                    symbol=symbol.upper(),
                    interval=interval,
                    limit=limit
                )

                if not response or 'result' not in response:
                    raise Exception(f"Failed to get data from Bybit: {response}")

                klines = response['result']

                # Create DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                ])

                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_columns] = df[numeric_columns].astype(float)

                # Set timestamp as index and select OHLCV columns
                df.set_index('timestamp', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]

                return df

            elif self.exchange_id == 'kucoin':
                # Map timeframe to KuCoin interval format
                timeframe_map = {
                    '1m': '1min',
                    '3m': '3min',
                    '5m': '5min',
                    '15m': '15min',
                    '30m': '30min',
                    '1h': '1hour',
                    '2h': '2hour',
                    '4h': '4hour',
                    '6h': '6hour',
                    '12h': '12hour',
                    '1d': '1day',
                    '1w': '1week'
                }

                interval = timeframe_map.get(timeframe)
                if not interval:
                    raise ValueError(f"Unsupported timeframe for KuCoin: {timeframe}")

                # Fetch klines
                klines = self.client.get_kline_data(
                    symbol=symbol.upper(),
                    kline_type=interval,
                    size=limit
                )

                # Create DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'
                ])

                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_columns] = df[numeric_columns].astype(float)

                # Set timestamp as index and select OHLCV columns
                df.set_index('timestamp', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]

                return df

            else:
                raise ValueError(f"Getting market data not implemented for {self.exchange_id}")

        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            raise

    def place_order(self, symbol: str, order_type: str, side: str,
                    quantity: float, price: Optional[float] = None,
                    time_in_force: str = 'GTC', **kwargs) -> Dict[str, Any]:
        """
        Place an order on the exchange

        Args:
            symbol: Trading symbol
            order_type: Type of order (LIMIT, MARKET)
            side: Order side (BUY, SELL)
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            **kwargs: Additional parameters for the order

        Returns:
            Order information
        """
        try:
            if self.exchange_id == 'binance':
                # Validate order parameters
                if order_type.upper() == 'LIMIT' and price is None:
                    raise ValueError("Price is required for LIMIT orders")

                # Place order
                order_params = {
                    'symbol': symbol.upper(),
                    'side': side.upper(),
                    'type': order_type.upper(),
                    'quantity': quantity,
                    'timeInForce': time_in_force
                }

                if price is not None:
                    order_params['price'] = price

                # Add any additional parameters
                order_params.update(kwargs)

                # Place the order
                if order_type.upper() == 'MARKET':
                    order = self.client.create_order(
                        symbol=symbol.upper(),
                        side=side.upper(),
                        type=order_type.upper(),
                        quantity=quantity
                    )
                else:
                    order = self.client.create_order(**order_params)

                # Store order in open orders list
                self.open_orders.append(order)

                logger.info(f"Placed {side} {order_type} order on {symbol}: {order}")
                return order

            elif self.exchange_id == 'bybit':
                # Convert order parameters to Bybit format
                bybit_side = 'Buy' if side.upper() == 'BUY' else 'Sell'
                bybit_type = 'Market' if order_type.upper() == 'MARKET' else 'Limit'

                # Place order
                order_params = {
                    'symbol': symbol.upper(),
                    'side': bybit_side,
                    'order_type': bybit_type,
                    'qty': quantity,
                    'time_in_force': time_in_force
                }

                if price is not None:
                    order_params['price'] = price

                # Add any additional parameters
                order_params.update(kwargs)

                # Place the order
                response = self.client.place_active_order(**order_params)

                if not response or 'result' not in response:
                    raise Exception(f"Failed to place order on Bybit: {response}")

                order = response['result']

                # Store order in open orders list
                self.open_orders.append(order)

                logger.info(f"Placed {side} {order_type} order on {symbol}: {order}")
                return order

            elif self.exchange_id == 'kucoin':
                # Convert order parameters to KuCoin format
                kucoin_side = side.upper()
                kucoin_type = order_type.upper()

                # Place order
                order_params = {
                    'symbol': symbol.upper(),
                    'side': kucoin_side,
                    'type': kucoin_type,
                    'size': quantity
                }

                if price is not None:
                    order_params['price'] = price

                # Add any additional parameters
                order_params.update(kwargs)

                # Place the order
                order = self.client.create_limit_order(
                    **order_params) if kucoin_type == 'LIMIT' else self.client.create_market_order(**order_params)

                # Store order in open orders list
                self.open_orders.append(order)

                logger.info(f"Placed {side} {order_type} order on {symbol}: {order}")
                return order

            else:
                raise ValueError(f"Placing orders not implemented for {self.exchange_id}")

        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise

    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order

        Args:
            symbol: Trading symbol
            order_id: ID of the order to cancel

        Returns:
            Cancellation result
        """
        try:
            if self.exchange_id == 'binance':
                result = self.client.cancel_order(
                    symbol=symbol.upper(),
                    orderId=order_id
                )

                # Remove from open orders list
                self.open_orders = [order for order in self.open_orders if order['orderId'] != order_id]

                logger.info(f"Cancelled order {order_id} on {symbol}: {result}")
                return result

            elif self.exchange_id == 'bybit':
                response = self.client.cancel_active_order(
                    symbol=symbol.upper(),
                    order_id=order_id
                )

                if not response or 'result' not in response:
                    raise Exception(f"Failed to cancel order on Bybit: {response}")

                result = response['result']

                # Remove from open orders list
                self.open_orders = [order for order in self.open_orders if order['order_id'] != order_id]

                logger.info(f"Cancelled order {order_id} on {symbol}: {result}")
                return result

            elif self.exchange_id == 'kucoin':
                result = self.client.cancel_order(order_id)

                # Remove from open orders list
                self.open_orders = [order for order in self.open_orders if order['orderId'] != order_id]

                logger.info(f"Cancelled order {order_id} on {symbol}: {result}")
                return result

            else:
                raise ValueError(f"Cancelling orders not implemented for {self.exchange_id}")

        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            raise

    def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order

        Args:
            symbol: Trading symbol
            order_id: ID of the order

        Returns:
            Order status information
        """
        try:
            if self.exchange_id == 'binance':
                order = self.client.get_order(
                    symbol=symbol.upper(),
                    orderId=order_id
                )

                return order

            elif self.exchange_id == 'bybit':
                response = self.client.query_active_order(
                    symbol=symbol.upper(),
                    order_id=order_id
                )

                if not response or 'result' not in response:
                    raise Exception(f"Failed to get order status from Bybit: {response}")

                return response['result']

            elif self.exchange_id == 'kucoin':
                order = self.client.get_order_details(order_id)

                return order

            else:
                raise ValueError(f"Getting order status not implemented for {self.exchange_id}")

        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
            raise

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders

        Args:
            symbol: Trading symbol (optional)

        Returns:
            List of open orders
        """
        try:
            if self.exchange_id == 'binance':
                if symbol:
                    orders = self.client.get_open_orders(symbol=symbol.upper())
                else:
                    orders = self.client.get_open_orders()

                # Update open orders list
                self.open_orders = orders

                return orders

            elif self.exchange_id == 'bybit':
                response = self.client.query_active_order(
                    symbol=symbol.upper() if symbol else None
                )

                if not response or 'result' not in response:
                    raise Exception(f"Failed to get open orders from Bybit: {response}")

                orders = response['result']

                # Update open orders list
                self.open_orders = orders

                return orders

            elif self.exchange_id == 'kucoin':
                if symbol:
                    orders = self.client.get_active_orders(symbol=symbol.upper())
                else:
                    orders = self.client.get_active_orders()

                # Update open orders list
                self.open_orders = orders

                return orders

            else:
                raise ValueError(f"Getting open orders not implemented for {self.exchange_id}")

        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            raise

    def get_balance(self, asset: Optional[str] = None) -> Union[Dict[str, Any], float]:
        """
        Get account balance

        Args:
            asset: Asset symbol (e.g., 'BTC', 'USDT')

        Returns:
            Balance information or specific asset balance
        """
        try:
            if self.exchange_id == 'binance':
                account = self.client.get_account()

                # If asset specified, return specific balance
                if asset:
                    for balance in account['balances']:
                        if balance['asset'] == asset.upper():
                            return float(balance['free'])

                    # Asset not found
                    return 0.0

                # Return full account info
                return account

            elif self.exchange_id == 'bybit':
                response = self.client.get_wallet_balance()

                if not response or 'result' not in response:
                    raise Exception(f"Failed to get balance from Bybit: {response}")

                balances = response['result']

                # If asset specified, return specific balance
                if asset:
                    asset_upper = asset.upper()
                    if asset_upper in balances:
                        return float(balances[asset_upper]['available_balance'])

                    # Asset not found
                    return 0.0

                # Return full balance info
                return balances

            elif self.exchange_id == 'kucoin':
                if asset:
                    balance = self.client.get_account(asset.upper())
                    return float(balance['available'])
                else:
                    balances = self.client.get_accounts()
                    return balances

            else:
                raise ValueError(f"Getting balance not implemented for {self.exchange_id}")

        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            raise

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker information

        Args:
            symbol: Trading symbol

        Returns:
            Ticker information
        """
        try:
            if self.exchange_id == 'binance':
                ticker = self.client.get_ticker(symbol=symbol.upper())
                return ticker

            elif self.exchange_id == 'bybit':
                response = self.client.latest_information_for_symbol(
                    symbol=symbol.upper()
                )

                if not response or 'result' not in response:
                    raise Exception(f"Failed to get ticker from Bybit: {response}")

                # Get the first item if result is a list
                result = response['result']
                ticker = result[0] if isinstance(result, list) else result

                return ticker

            elif self.exchange_id == 'kucoin':
                ticker = self.client.get_ticker(symbol.upper())
                return ticker

            else:
                raise ValueError(f"Getting ticker not implemented for {self.exchange_id}")

        except Exception as e:
            logger.error(f"Error getting ticker: {str(e)}")
            raise

    def close(self):
        """Close the exchange connection"""
        logger.info(f"Closing {self.exchange_id} connection")