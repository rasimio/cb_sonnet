"""
Reinforcement Learning Model Implementation for TensorTrade
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import time
import json
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam

# Use gymnasium instead of gym for compatibility with newer Stable Baselines 3
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from data.preprocessing import normalize_data, create_sequences

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Custom Gym environment for trading using RL
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0,
                commission: float = 0.001, window_size: int = 60,
                reward_scaling: float = 0.01):
        """
        Initialize the trading environment

        Args:
            data: DataFrame with OHLCV data
            initial_balance: Starting account balance
            commission: Trading commission as a decimal (e.g., 0.001 = 0.1%)
            window_size: Number of price bars to include in state
            reward_scaling: Scaling factor for rewards to stabilize training
        """
        super(TradingEnvironment, self).__init__()

        # Ensure data has at least window_size rows
        if len(data) <= window_size:
            raise ValueError(f"Data must have more than {window_size} rows, but only has {len(data)}")

        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.reward_scaling = reward_scaling

        # Position flags (1: long, 0: neutral, -1: short)
        self.position = 0
        self.position_size = 0.0
        self.entry_price = None
        self.entry_step = None

        # Account variables
        self.balance = initial_balance
        self.equity = initial_balance
        self.prev_equity = initial_balance

        # Episode variables
        self.current_step = window_size
        self.trades = []

        # Features used for the state
        self.features = ['open', 'high', 'low', 'close', 'volume']

        # Create normalized features
        self._prepare_data()

        # Define action and observation space
        # Actions: 0=Do nothing, 1=Buy, 2=Sell, 3=Close position
        self.action_space = spaces.Discrete(4)

        # Observation space: price history + account state
        # Price history: window_size x features
        # Account state: [position, position_size/balance, unrealized_pnl/balance]
        # So total observation shape is (window_size * len(features) + 3,)
        num_features = len(self.features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * num_features + 3,),
            dtype=np.float32
        )

    def _prepare_data(self) -> None:
        """Prepare and normalize data for the environment"""
        # Make a copy to avoid modifying the original
        df = self.data.copy()

        # Add technical indicators
        # Simple moving averages
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()

        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

        # Add these indicators to features
        self.features.extend(['sma_7', 'sma_20', 'ema_12', 'ema_26',
                             'macd', 'macd_signal', 'rsi_14',
                             'bb_middle', 'bb_upper', 'bb_lower'])

        # Drop NaN values
        df = df.dropna()

        # Make sure we still have enough data after dropping NaN values
        if len(df) <= self.window_size:
            raise ValueError(f"After adding technical indicators and dropping NaN values, data has only {len(df)} rows, which is not enough for window size {self.window_size}")

        # Normalize data
        self.normalized_data, self.scaler = normalize_data(df[self.features])

        # Store the original data for reward calculation
        self.df = df

        # Store data length for bounds checking
        self.data_length = len(df)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state

        Returns:
            Initial observation and empty info dict
        """
        # Reset episode variables
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.prev_equity = self.initial_balance
        self.position = 0
        self.position_size = 0.0
        self.entry_price = None
        self.entry_step = None
        self.trades = []

        # Set seed for reproducibility if provided
        if seed is not None:
            np.random.seed(seed)

        return self._get_observation(), {}  # Return observation and empty info dict

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation

        Returns:
            Numpy array with price history and account state
        """
        # Validate current step is within bounds
        if self.current_step < self.window_size:
            logger.warning(f"Current step {self.current_step} is less than window size {self.window_size}, adjusting to window size")
            self.current_step = self.window_size

        if self.current_step >= self.data_length:
            logger.warning(f"Current step {self.current_step} is beyond data length {self.data_length}, capping at last index")
            self.current_step = self.data_length - 1

        # Get price history window
        price_history = self.normalized_data[self.current_step - self.window_size:self.current_step]

        # Flatten price history
        price_obs = price_history.flatten()

        # Get current price for unrealized PnL calculation
        current_price = self.df.iloc[self.current_step]['close']

        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        if self.position == 1 and self.entry_price is not None:  # Long position
            unrealized_pnl = self.position_size * (current_price - self.entry_price) / self.entry_price
        elif self.position == -1 and self.entry_price is not None:  # Short position
            unrealized_pnl = self.position_size * (self.entry_price - current_price) / self.entry_price

        # Normalize account state
        position_norm = self.position  # Already -1, 0, or 1
        position_size_norm = self.position_size / self.balance if self.balance > 0 else 0
        unrealized_pnl_norm = unrealized_pnl / self.balance if self.balance > 0 else 0

        # Combine price history and account state
        account_obs = np.array([position_norm, position_size_norm, unrealized_pnl_norm])

        # Create full observation
        full_obs = np.concatenate([price_obs, account_obs]).astype(np.float32)

        # Validate observation shape matches observation space
        expected_shape = self.observation_space.shape[0]
        if len(full_obs) != expected_shape:
            logger.error(f"Observation shape mismatch: expected {expected_shape}, got {len(full_obs)}")
            # Pad or truncate to match expected shape
            if len(full_obs) < expected_shape:
                padding = np.zeros(expected_shape - len(full_obs), dtype=np.float32)
                full_obs = np.concatenate([full_obs, padding])
            else:
                full_obs = full_obs[:expected_shape]

        return full_obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment

        Args:
            action: Action to take (0=Do nothing, 1=Buy, 2=Sell, 3=Close)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate current step is within bounds
        if self.current_step >= self.data_length:
            logger.warning(f"Step called with current_step={self.current_step} >= data_length={self.data_length}")
            # Return final state with done=True
            return self._get_observation(), 0.0, True, False, {
                'balance': self.balance,
                'equity': self.equity,
                'position': self.position,
                'position_size': self.position_size,
                'reward': 0.0,
                'trades': len(self.trades),
                'message': 'Episode ended: out of data'
            }

        # Get current price data
        current_data = self.df.iloc[self.current_step]
        current_price = current_data['close']

        # Execute action
        self._take_action(action, current_price)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= self.data_length - 1
        truncated = False  # We don't truncate episodes early

        # Calculate reward
        reward = self._calculate_reward()

        # Get new observation
        obs = self._get_observation()

        # Calculate info for logging
        info = {
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'position_size': self.position_size,
            'reward': reward,
            'trades': len(self.trades)
        }

        return obs, reward, terminated, truncated, info

    def _take_action(self, action: int, current_price: float) -> None:
        """
        Execute the specified action

        Args:
            action: Action to take (0=Hold, 1=Buy, 2=Sell, 3=Close)
            current_price: Current asset price
        """
        # Default position size as a percentage of balance
        default_size_pct = 0.5

        # Entry/exit execution
        if action == 1:  # Buy
            if self.position <= 0:  # If no position or short, close and go long
                # Close existing short position if any
                if self.position == -1:
                    # Calculate profit/loss
                    exit_price = current_price * (1 + self.commission)  # Include commission
                    profit = self.position_size * (self.entry_price - exit_price) / self.entry_price

                    # Update balance
                    self.balance += self.position_size + profit

                    # Record trade
                    self.trades.append({
                        'entry_step': self.entry_step,
                        'exit_step': self.current_step,
                        'entry_price': self.entry_price,
                        'exit_price': exit_price,
                        'position': self.position,
                        'profit': profit
                    })

                # Enter long position
                self.position = 1
                self.entry_price = current_price * (1 + self.commission)  # Include commission
                self.entry_step = self.current_step

                # Calculate position size (use percentage of current balance)
                self.position_size = self.balance * default_size_pct
                self.balance -= self.position_size  # Remove position size from balance

        elif action == 2:  # Sell
            if self.position >= 0:  # If no position or long, close and go short
                # Close existing long position if any
                if self.position == 1:
                    # Calculate profit/loss
                    exit_price = current_price * (1 - self.commission)  # Include commission
                    profit = self.position_size * (exit_price - self.entry_price) / self.entry_price

                    # Update balance
                    self.balance += self.position_size + profit

                    # Record trade
                    self.trades.append({
                        'entry_step': self.entry_step,
                        'exit_step': self.current_step,
                        'entry_price': self.entry_price,
                        'exit_price': exit_price,
                        'position': self.position,
                        'profit': profit
                    })

                # Enter short position
                self.position = -1
                self.entry_price = current_price * (1 - self.commission)  # Include commission
                self.entry_step = self.current_step

                # Calculate position size (use percentage of current balance)
                self.position_size = self.balance * default_size_pct
                self.balance -= self.position_size  # Remove position size from balance

        elif action == 3:  # Close position
            if self.position != 0:  # If there's an open position
                # Calculate exit price with commission
                if self.position == 1:  # Long position
                    exit_price = current_price * (1 - self.commission)
                    profit = self.position_size * (exit_price - self.entry_price) / self.entry_price
                else:  # Short position
                    exit_price = current_price * (1 + self.commission)
                    profit = self.position_size * (self.entry_price - exit_price) / self.entry_price

                # Update balance
                self.balance += self.position_size + profit

                # Record trade
                self.trades.append({
                    'entry_step': self.entry_step,
                    'exit_step': self.current_step,
                    'entry_price': self.entry_price,
                    'exit_price': exit_price,
                    'position': self.position,
                    'profit': profit
                })

                # Reset position
                self.position = 0
                self.position_size = 0.0
                self.entry_price = None
                self.entry_step = None

        # Calculate current equity
        self.prev_equity = self.equity
        self.equity = self._calculate_equity(current_price)

    def _calculate_equity(self, current_price: float) -> float:
        """
        Calculate current portfolio equity

        Args:
            current_price: Current asset price

        Returns:
            Current equity value
        """
        equity = self.balance

        # Add value of open position
        if self.position == 1 and self.entry_price is not None:  # Long
            profit = self.position_size * (current_price - self.entry_price) / self.entry_price
            equity += self.position_size + profit
        elif self.position == -1 and self.entry_price is not None:  # Short
            profit = self.position_size * (self.entry_price - current_price) / self.entry_price
            equity += self.position_size + profit

        return equity

    def _calculate_reward(self) -> float:
        """
        Calculate the reward for the current step

        Returns:
            Reward value
        """
        # Reward based on equity change (scaled)
        equity_change = (self.equity - self.prev_equity) / self.initial_balance

        # Scale reward for stability
        reward = equity_change * self.reward_scaling

        # Add small negative reward for doing nothing to encourage action
        if self.equity == self.prev_equity and self.position == 0:
            reward -= 0.0001

        # Add penalty for frequent trading
        if len(self.trades) > 0 and self.trades[-1]['exit_step'] == self.current_step:
            # If we just closed a position that was opened recently, apply penalty
            trade_duration = self.trades[-1]['exit_step'] - self.trades[-1]['entry_step']
            if trade_duration < 5:  # Penalize trades held for less than 5 steps
                reward -= 0.005

        return reward

    def render(self, mode='human'):
        """Render the environment state"""
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, " +
              f"Equity: {self.equity:.2f}, Position: {self.position}")


class RLModel:
    """
    Reinforcement Learning model for trading decisions
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the RL model with the specified configuration

        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        self.model = None
        self.env = None

        # Model hyperparameters with defaults
        self.algo = self.config.get('algorithm', 'ppo')
        self.policy = self.config.get('policy', 'MlpPolicy')
        self.window_size = self.config.get('window_size', 60)
        self.reward_scaling = self.config.get('reward_scaling', 0.01)
        self.learning_rate = self.config.get('learning_rate', 0.0003)

        logger.info(f"Initialized RL model with {self.algo} algorithm")

    def _create_environment(self, data: pd.DataFrame, is_training: bool = True) -> gym.Env:
        """
        Create a trading environment for the RL agent

        Args:
            data: DataFrame with price data
            is_training: Whether this environment is for training

        Returns:
            Gym environment
        """
        try:
            # Create base environment
            env = TradingEnvironment(
                data=data,
                initial_balance=self.config.get('initial_balance', 10000.0),
                commission=self.config.get('commission', 0.001),
                window_size=self.window_size,
                reward_scaling=self.reward_scaling
            )

            # Add monitoring
            log_dir = './logs/rl_training' if is_training else './logs/rl_testing'
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir)

            # Wrap in vectorized environment
            env = DummyVecEnv([lambda: env])

            return env

        except Exception as e:
            logger.error(f"Error creating environment: {str(e)}")
            raise

    def _create_model(self, env: gym.Env) -> Any:
        """
        Create a new RL model

        Args:
            env: Training environment

        Returns:
            RL model instance
        """
        try:
            if self.algo.lower() == 'ppo':
                model = PPO(
                    policy=self.policy,
                    env=env,
                    learning_rate=self.learning_rate,
                    n_steps=self.config.get('n_steps', 2048),
                    batch_size=self.config.get('batch_size', 64),
                    n_epochs=self.config.get('n_epochs', 10),
                    gamma=self.config.get('gamma', 0.99),
                    gae_lambda=self.config.get('gae_lambda', 0.95),
                    clip_range=self.config.get('clip_range', 0.2),
                    verbose=1
                )
            elif self.algo.lower() == 'a2c':
                model = A2C(
                    policy=self.policy,
                    env=env,
                    learning_rate=self.learning_rate,
                    n_steps=self.config.get('n_steps', 5),
                    gamma=self.config.get('gamma', 0.99),
                    verbose=1
                )
            elif self.algo.lower() == 'dqn':
                model = DQN(
                    policy=self.policy,
                    env=env,
                    learning_rate=self.learning_rate,
                    buffer_size=self.config.get('buffer_size', 10000),
                    learning_starts=self.config.get('learning_starts', 1000),
                    batch_size=self.config.get('batch_size', 32),
                    gamma=self.config.get('gamma', 0.99),
                    train_freq=self.config.get('train_freq', 1),
                    gradient_steps=self.config.get('gradient_steps', 1),
                    target_update_interval=self.config.get('target_update_interval', 500),
                    exploration_fraction=self.config.get('exploration_fraction', 0.1),
                    exploration_initial_eps=self.config.get('exploration_initial_eps', 1.0),
                    exploration_final_eps=self.config.get('exploration_final_eps', 0.05),
                    verbose=1
                )
            else:
                raise ValueError(f"Unsupported algorithm: {self.algo}")

            return model

        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise

    def train(self, data: pd.DataFrame, total_timesteps: int = 500000,
             eval_freq: int = 10000, eval_episodes: int = 5) -> Dict[str, Any]:
        """
        Train the RL model on the provided data

        Args:
            data: DataFrame with price data
            total_timesteps: Total number of timesteps to train for
            eval_freq: Frequency of evaluation during training
            eval_episodes: Number of episodes for each evaluation

        Returns:
            Dictionary with training stats
        """
        try:
            # Make sure data has enough samples
            if len(data) <= self.window_size:
                raise ValueError(f"Training data must have more than {self.window_size} samples, but only has {len(data)}")

            # Split data into training and validation
            train_size = int(len(data) * 0.8)
            train_data = data.iloc[:train_size]
            val_data = data.iloc[train_size:]

            logger.info(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")

            # Create environments
            train_env = self._create_environment(train_data, is_training=True)
            eval_env = self._create_environment(val_data, is_training=False)

            # Create callback for evaluation
            eval_callback = EvalCallback(
                eval_env=eval_env,
                n_eval_episodes=eval_episodes,
                eval_freq=eval_freq,
                log_path='./logs/rl_eval',
                best_model_save_path='./saved_models/rl_best_model',
                deterministic=True,
                render=False
            )

            # Create model
            self.model = self._create_model(train_env)

            # Train model
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=eval_callback,
                progress_bar=False  # Disable progress bar to avoid dependency issues
            )

            # Close environments
            train_env.close()
            eval_env.close()

            # Load best model if it exists
            best_model_path = os.path.join('./saved_models/rl_best_model', f"{self.algo}_best_model")
            if os.path.exists(best_model_path + ".zip"):
                if self.algo.lower() == 'ppo':
                    self.model = PPO.load(best_model_path)
                elif self.algo.lower() == 'a2c':
                    self.model = A2C.load(best_model_path)
                elif self.algo.lower() == 'dqn':
                    self.model = DQN.load(best_model_path)

            logger.info(f"Model trained for {total_timesteps} timesteps")

            return {"status": "success", "total_timesteps": total_timesteps}

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading decision using the trained model

        Args:
            data: DataFrame with recent price data

        Returns:
            Dictionary with prediction information
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before making predictions")

        try:
            # Make sure we have enough data
            if len(data) < self.window_size:
                raise ValueError(f"Data must contain at least {self.window_size} samples, but only has {len(data)}")

            # Create environment for prediction
            env = self._create_environment(data, is_training=False)

            # Reset environment
            obs, _ = env.reset()

            # Get action from model
            action, _states = self.model.predict(obs, deterministic=True)

            # Map action to trading decision
            action_map = {
                0: "HOLD",
                1: "BUY",
                2: "SELL",
                3: "CLOSE"
            }

            action_code = int(action[0])
            decision = action_map.get(action_code, "HOLD")  # Default to HOLD if unknown action

            # Get current price
            current_price = data.iloc[-1]['close']

            return {
                "action": decision,
                "action_code": action_code,
                "confidence": 1.0,  # RL models don't provide confidence scores directly
                "current_price": float(current_price)
            }

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Provide a safe default response
            return {
                "action": "HOLD",
                "action_code": 0,
                "confidence": 0.0,
                "current_price": float(data.iloc[-1]['close']) if not data.empty else 0.0,
                "error": str(e)
            }

    def evaluate(self, data: pd.DataFrame, episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate model performance on test data

        Args:
            data: DataFrame with test data
            episodes: Number of evaluation episodes

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before evaluation")

        try:
            # Create environment for evaluation
            eval_env = self._create_environment(data, is_training=False)

            # Run multiple episodes and collect returns
            returns = []
            profits = []
            trade_counts = []
            win_rates = []

            for _ in range(episodes):
                obs, _ = eval_env.reset()
                done = False
                total_reward = 0

                # Track trades for win rate calculation
                trades = []

                while not done:
                    action, _states = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    total_reward += reward[0]
                    done = terminated[0] or truncated[0]

                    # Track trades
                    if 'trades' in info[0] and info[0]['trades'] > len(trades):
                        trades.append(info[0]['trades'])

                # Calculate returns and profit
                env = eval_env.envs[0].unwrapped
                final_equity = env.equity
                total_return = (final_equity - env.initial_balance) / env.initial_balance
                profit = final_equity - env.initial_balance

                # Calculate win rate
                winning_trades = sum(1 for trade in env.trades if trade['profit'] > 0)
                win_rate = winning_trades / len(env.trades) if len(env.trades) > 0 else 0

                returns.append(total_return)
                profits.append(profit)
                trade_counts.append(len(env.trades))
                win_rates.append(win_rate)

            # Close environment
            eval_env.close()

            # Calculate metrics
            metrics = {
                'mean_return': float(np.mean(returns)),
                'mean_profit': float(np.mean(profits)),
                'mean_trade_count': float(np.mean(trade_counts)),
                'mean_win_rate': float(np.mean(win_rates)),
                'sharpe_ratio': float(np.mean(returns) / (np.std(returns) + 1e-10))
            }

            return metrics

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return {
                'error': str(e),
                'mean_return': 0.0,
                'mean_profit': 0.0,
                'mean_trade_count': 0.0,
                'mean_win_rate': 0.0,
                'sharpe_ratio': 0.0
            }

    def save(self, path: str) -> None:
        """
        Save the model to disk

        Args:
            path: Directory path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")

        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)

            # Save model
            model_path = os.path.join(path, f"{self.algo}_model")
            self.model.save(model_path)

            # Save configuration
            with open(os.path.join(path, 'config.json'), 'w') as f:
                # Handle non-serializable objects
                config_copy = {}
                for k, v in self.config.items():
                    if isinstance(v, (dict, list, str, int, float, bool, type(None))):
                        config_copy[k] = v
                    else:
                        config_copy[k] = str(v)

                json.dump(config_copy, f)

            # Save window size and other key parameters separately for easier loading
            with open(os.path.join(path, 'params.json'), 'w') as f:
                params = {
                    'window_size': self.window_size,
                    'reward_scaling': self.reward_scaling,
                    'algorithm': self.algo,
                    'policy': self.policy
                }
                json.dump(params, f)

            logger.info(f"Model saved to {path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load(cls, path: str) -> 'RLModel':
        """
        Load a trained model from disk

        Args:
            path: Directory path where the model is saved

        Returns:
            Loaded RL model instance
        """
        try:
            # Load configuration
            try:
                with open(os.path.join(path, 'config.json'), 'r') as f:
                    config = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load config.json, using default config: {e}")
                config = {}

            # Load additional parameters if available
            try:
                with open(os.path.join(path, 'params.json'), 'r') as f:
                    params = json.load(f)
                    # Update config with params
                    config.update(params)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load params.json: {e}")

            # Create model instance with loaded config
            instance = cls(config)

            # Determine algorithm type
            algo = config.get('algorithm', 'ppo').lower()

            # Load the model based on algorithm
            model_path = os.path.join(path, f"{algo}_model")
            if os.path.exists(model_path + ".zip"):
                if algo == 'ppo':
                    instance.model = PPO.load(model_path)
                elif algo == 'a2c':
                    instance.model = A2C.load(model_path)
                elif algo == 'dqn':
                    instance.model = DQN.load(model_path)
                else:
                    raise ValueError(f"Unsupported algorithm: {algo}")
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}.zip")

            logger.info(f"Model loaded from {path}")
            return instance

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise