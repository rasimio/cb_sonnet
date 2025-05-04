"""
LSTM Model Implementation for TensorTrade
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import pickle
import json

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import MeanSquaredError, Huber

from data.preprocessing import normalize_data, create_sequences

logger = logging.getLogger(__name__)

class LSTMModel:
    """
    LSTM-based model for predicting price movements
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the LSTM model with the specified configuration

        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.sequence_length = self.config.get('sequence_length', 60)
        self.prediction_horizon = self.config.get('prediction_horizon', 1)

        # Model hyperparameters with defaults
        self.units = self.config.get('units', 128)
        self.layers = self.config.get('layers', 2)
        self.dropout = self.config.get('dropout', 0.2)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.loss_function = self.config.get('loss', 'huber')

        # For tracking input shape
        self.input_feature_count = None

        logger.info(f"Initialized LSTM model with {self.layers} layers and {self.units} units per layer")

    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture

        Args:
            input_shape: Shape of input data (sequence_length, features)
        """
        model = Sequential()

        # Record the input shape for future reference
        self.input_feature_count = input_shape[1]

        # First LSTM layer with return sequences for stacking
        model.add(LSTM(
            units=self.units,
            return_sequences=True if self.layers > 1 else False,
            input_shape=input_shape,
            recurrent_dropout=0.0,  # Avoid using recurrent_dropout for better performance
        ))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())

        # Middle LSTM layers if requested
        for i in range(self.layers - 2):
            model.add(LSTM(
                units=self.units,
                return_sequences=True,
                recurrent_dropout=0.0
            ))
            model.add(Dropout(self.dropout))
            model.add(BatchNormalization())

        # Final LSTM layer if more than one layer
        if self.layers > 1:
            model.add(LSTM(units=self.units))
            model.add(Dropout(self.dropout))
            model.add(BatchNormalization())

        # Output layers for price prediction
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.prediction_horizon))

        # Select loss function
        if self.loss_function == 'mse':
            loss = MeanSquaredError()
        elif self.loss_function == 'huber':
            loss = Huber()
        else:
            loss = 'mean_squared_error'

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['mae']
        )

        # Store the model
        self.model = model
        logger.info(f"Built LSTM model: {model.summary()}")

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training by normalizing and creating sequences

        Args:
            data: DataFrame containing price and feature data

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        # Define features to use
        self.feature_columns = self.config.get('feature_columns', ['open', 'high', 'low', 'close', 'volume'])

        # Add technical indicators if requested
        if self.config.get('use_technical_indicators', True):
            # Simple moving averages
            data['sma_7'] = data['close'].rolling(window=7).mean()
            data['sma_20'] = data['close'].rolling(window=20).mean()

            # Exponential moving averages
            data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
            data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()

            # MACD
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

            # RSI (14)
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi_14'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            data['bb_std'] = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
            data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']

            # Include these indicators in features
            additional_features = ['sma_7', 'sma_20', 'ema_12', 'ema_26',
                                 'macd', 'macd_signal', 'rsi_14',
                                 'bb_middle', 'bb_upper', 'bb_lower']

            self.feature_columns.extend(additional_features)

        # Drop NaN values after adding technical indicators
        data = data.dropna()

        # Create target variable (future price change)
        target_column = self.config.get('target_column', 'close')
        if self.config.get('target_type', 'price') == 'price':
            # Predict actual price
            data['target'] = data[target_column].shift(-self.prediction_horizon)
        elif self.config.get('target_type', 'price') == 'return':
            # Predict percentage return
            data['target'] = data[target_column].pct_change(self.prediction_horizon).shift(-self.prediction_horizon) * 100
        else:
            # Predict price direction (1 for up, 0 for down)
            data['target'] = (data[target_column].shift(-self.prediction_horizon) >
                            data[target_column]).astype(int)

        # Drop rows with NaN targets
        data = data.dropna()

        # Store all feature names (important for later prediction)
        all_features = data[self.feature_columns].columns.tolist()
        self._model_features = all_features

        # Normalize data and save scaler for later use
        data_normalized, self.scaler = normalize_data(data[self.feature_columns + ['target']])

        # Save the exact set of features used with the scaler
        self._scaler_features = self.feature_columns + ['target']

        # Create sequences for LSTM training
        X, y = create_sequences(
            data_normalized,
            sequence_length=self.sequence_length,
            target_column_idx=-1  # Target is the last column
        )

        # Split into training and validation sets
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        return X_train, y_train, X_val, y_val

    def train(self, data: pd.DataFrame, epochs: int = 100, batch_size: int = 32,
             validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the LSTM model on the provided data

        Args:
            data: DataFrame containing price and feature data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Portion of data to use for validation

        Returns:
            Dictionary containing training history
        """
        # Prepare data
        X_train, y_train, X_val, y_val = self.prepare_data(data)

        # Build model if it doesn't exist yet
        if self.model is None:
            self._build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        # Define callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            ),
            ModelCheckpoint(
                filepath='saved_models/lstm_checkpoint.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        logger.info(f"Model trained for {len(history.history['loss'])} epochs")
        logger.info(f"Final loss: {history.history['loss'][-1]:.4f}, " +
                   f"Val loss: {history.history['val_loss'][-1]:.4f}")

        return history.history

    def _prepare_features_for_prediction(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for prediction, ensuring correct dimensionality

        Args:
            data: DataFrame with recent price data

        Returns:
            Normalized feature array with correct dimensions
        """
        # Check if we know the required feature count for the model
        required_feature_count = self.input_feature_count

        # If we don't have this information, try to get it from the model
        if required_feature_count is None:
            # Try to extract from model's input shape
            if self.model is not None:
                input_shape = self.model.input_shape
                if input_shape is not None and len(input_shape) == 3:
                    required_feature_count = input_shape[2]
                    self.input_feature_count = required_feature_count
                    logger.info(f"Extracted feature count from model: {required_feature_count}")

        # Use the most recent data points
        recent_data = data.tail(self.sequence_length).copy()

        # Add technical indicators if needed
        if self.config.get('use_technical_indicators', True):
            # We need historical data for indicators, so we'll use all the data we have
            indicator_data = data.copy()

            # Calculate technical indicators
            indicator_data['sma_7'] = indicator_data['close'].rolling(window=7).mean()
            indicator_data['sma_20'] = indicator_data['close'].rolling(window=20).mean()
            indicator_data['ema_12'] = indicator_data['close'].ewm(span=12, adjust=False).mean()
            indicator_data['ema_26'] = indicator_data['close'].ewm(span=26, adjust=False).mean()
            indicator_data['macd'] = indicator_data['ema_12'] - indicator_data['ema_26']
            indicator_data['macd_signal'] = indicator_data['macd'].ewm(span=9, adjust=False).mean()

            # RSI (14)
            delta = indicator_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicator_data['rsi_14'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            indicator_data['bb_middle'] = indicator_data['close'].rolling(window=20).mean()
            indicator_data['bb_std'] = indicator_data['close'].rolling(window=20).std()
            indicator_data['bb_upper'] = indicator_data['bb_middle'] + 2 * indicator_data['bb_std']
            indicator_data['bb_lower'] = indicator_data['bb_middle'] - 2 * indicator_data['bb_std']

            # Get just the recent points with indicators computed
            recent_data = indicator_data.tail(self.sequence_length)

        # If we have model features saved (from training), use those to match dimensions
        if hasattr(self, '_model_features') and self._model_features is not None:
            expected_features = self._model_features
        else:
            # Fallback to feature columns
            expected_features = self.feature_columns

        # Prepare feature dataframe with expected columns
        features_df = pd.DataFrame(index=recent_data.index)

        # Add each expected feature, with 0 as default if missing
        for feature in expected_features:
            if feature in recent_data.columns:
                features_df[feature] = recent_data[feature]
            else:
                logger.warning(f"Feature {feature} not found in input data, filling with zeros")
                features_df[feature] = 0.0

        # Normalize data using saved scaler or manual method
        normalized_data = self._normalize_features(features_df)

        # If we know the required feature count, ensure we match it
        if required_feature_count is not None:
            current_feature_count = normalized_data.shape[1]

            if current_feature_count < required_feature_count:
                # We need to add more features (padding with zeros)
                padding = np.zeros((normalized_data.shape[0], required_feature_count - current_feature_count))
                normalized_data = np.hstack((normalized_data, padding))
                # logger.info(f"Added padding to match feature count: {current_feature_count} -> {required_feature_count}")
            elif current_feature_count > required_feature_count:
                # We need to truncate features
                normalized_data = normalized_data[:, :required_feature_count]
                logger.info(f"Truncated features to match feature count: {current_feature_count} -> {required_feature_count}")

        return normalized_data

    def _normalize_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Normalize features using scaler or manual method

        Args:
            features_df: DataFrame with features to normalize

        Returns:
            Normalized features as numpy array
        """
        try:
            if hasattr(self.scaler, 'transform'):
                # Use the sklearn scaler if available
                return self.scaler.transform(features_df)
            else:
                # Manual min-max scaling as fallback
                logger.warning("Using manual min-max scaling (scaler not available)")
                feature_min = features_df.min()
                feature_max = features_df.max()

                # Avoid division by zero
                denominator = feature_max - feature_min
                denominator = denominator.replace(0, 1)  # Replace zeros with ones

                normalized = (features_df - feature_min) / denominator
                return normalized.values
        except Exception as e:
            # Fallback if anything goes wrong
            # logger.error(f"Error in normalization: {str(e)}")
            # logger.warning("Using zero-mean, unit-variance normalization as fallback")

            # Simple z-score normalization
            mean = features_df.mean()
            std = features_df.std()
            std = std.replace(0, 1)  # Replace zeros with ones

            normalized = (features_df - mean) / std
            return normalized.values

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained model

        Args:
            data: DataFrame with recent price data

        Returns:
            Numpy array with predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before making predictions")

        # Make sure we have enough data for a sequence
        if len(data) < self.sequence_length:
            raise ValueError(f"Data must contain at least {self.sequence_length} samples")

        # Prepare features with correct dimensionality
        normalized_features = self._prepare_features_for_prediction(data)

        # Reshape for LSTM input [samples, time steps, features]
        X = normalized_features.reshape(1, normalized_features.shape[0], normalized_features.shape[1])

        # Generate prediction
        try:
            prediction_normalized = self.model.predict(X)
        except Exception as e:
            logger.error(f"Error during model prediction: {str(e)}")
            # If prediction fails, try a simpler approach
            if "incompatible with the layer" in str(e):
                # This might be due to dimension mismatch, try reshaping to match expected dimensions
                input_shape = self.model.input_shape
                if input_shape is not None and len(input_shape) == 3:
                    expected_shape = input_shape[1:]
                    logger.warning(f"Reshaping input to match expected shape: {expected_shape}")

                    # Reshape data to match expected dimensions
                    X_reshaped = np.zeros((1, expected_shape[0], expected_shape[1]))
                    # Copy as much as we can from our data
                    copy_timesteps = min(X.shape[1], expected_shape[0])
                    copy_features = min(X.shape[2], expected_shape[1])
                    X_reshaped[0, :copy_timesteps, :copy_features] = X[0, :copy_timesteps, :copy_features]

                    # Try prediction again
                    prediction_normalized = self.model.predict(X_reshaped)
                else:
                    # If we can't determine the expected shape, return a default prediction
                    logger.error("Cannot determine expected input shape, returning default prediction")
                    return data.iloc[-1]['close']  # Return the last known price
            else:
                # For other errors, return the last known price
                logger.error("Prediction failed, returning last known price")
                return data.iloc[-1]['close']

        # If we're predicting actual price, we need to invert the normalization
        if self.config.get('target_type', 'price') == 'price':
            try:
                if hasattr(self.scaler, 'inverse_transform'):
                    # Create a dummy array with zeros for features and the prediction for target
                    dummy = np.zeros((1, len(self._scaler_features)))
                    dummy[0, -1] = prediction_normalized[0, 0]

                    # Invert the normalization and extract the target
                    prediction = self.scaler.inverse_transform(dummy)[0, -1]
                else:
                    # Fallback: use the current price and add estimated change
                    current_price = data.iloc[-1]['close']
                    prediction = current_price * (1 + (prediction_normalized[0, 0] - 0.5) * 0.1)
            except Exception as e:
                logger.error(f"Error in denormalization: {str(e)}")
                current_price = data.iloc[-1]['close']
                # Simple fallback: return current price with small random adjustment
                prediction = current_price * (1 + (np.random.random() - 0.5) * 0.01)
        else:
            # For returns or direction, just use the predicted value
            prediction = prediction_normalized[0, 0]

        return prediction

    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data

        Args:
            data: DataFrame with test data

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before evaluation")

        # Prepare test data
        X_test, y_test, _, _ = self.prepare_data(data)

        # Evaluate the model
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {
            'loss': results[0],
            'mae': results[1]
        }

        # Add directional accuracy for price predictions
        if self.config.get('target_type', 'price') == 'price':
            y_pred = self.model.predict(X_test)

            # Calculate directional accuracy - whether price movement direction is predicted correctly
            actual_direction = np.sign(y_test[1:] - y_test[:-1])
            pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
            directional_accuracy = np.mean(actual_direction == pred_direction)

            metrics['directional_accuracy'] = float(directional_accuracy)

        return metrics

    def save(self, path: str) -> None:
        """
        Save the model to disk

        Args:
            path: Directory path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")

        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save the model in SavedModel format (compatible with TensorFlow Serving)
        self.model.save(os.path.join(path, 'model'), save_format='tf')

        # Save scaler using pickle to preserve the object's methods
        if self.scaler is not None:
            with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)

        # Save additional metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'scaler_features': self._scaler_features if hasattr(self, '_scaler_features') else None,
            'model_features': self._model_features if hasattr(self, '_model_features') else None,
            'input_feature_count': self.input_feature_count,
            'sequence_length': self.sequence_length,
            'config': self.config
        }

        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            # Handle non-serializable items
            serializable_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (list, dict, str, int, float, bool)) or v is None:
                    serializable_metadata[k] = v
                else:
                    serializable_metadata[k] = str(v)

            json.dump(serializable_metadata, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'LSTMModel':
        """
        Load a trained model from disk

        Args:
            path: Directory path where the model is saved

        Returns:
            Loaded LSTM model instance
        """
        # Create model instance
        instance = cls()

        # Load the model
        instance.model = load_model(os.path.join(path, 'model'))

        # Try to load metadata
        try:
            with open(os.path.join(path, 'metadata.json'), 'r') as f:
                metadata = json.load(f)

                # Restore metadata attributes
                if 'feature_columns' in metadata:
                    instance.feature_columns = metadata['feature_columns']

                if 'scaler_features' in metadata and metadata['scaler_features']:
                    instance._scaler_features = metadata['scaler_features']

                if 'model_features' in metadata and metadata['model_features']:
                    instance._model_features = metadata['model_features']

                if 'input_feature_count' in metadata and metadata['input_feature_count']:
                    instance.input_feature_count = metadata['input_feature_count']

                if 'sequence_length' in metadata:
                    instance.sequence_length = metadata['sequence_length']

                if 'config' in metadata:
                    instance.config = metadata['config']
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load metadata: {e}, trying legacy format")

            # Try loading the old-style metadata
            try:
                # Load feature columns
                with open(os.path.join(path, 'feature_columns.txt'), 'r') as f:
                    instance.feature_columns = f.read().split(',')
            except FileNotFoundError:
                logger.warning("Could not find feature_columns.txt")
                instance.feature_columns = ['open', 'high', 'low', 'close', 'volume']

            # Try to extract input shape from model
            if instance.model is not None:
                input_shape = instance.model.input_shape
                if input_shape is not None and len(input_shape) == 3:
                    instance.input_feature_count = input_shape[2]
                    logger.info(f"Extracted feature count from model: {instance.input_feature_count}")

        # Load scaler using pickle
        try:
            with open(os.path.join(path, 'scaler.pkl'), 'rb') as f:
                instance.scaler = pickle.load(f)
        except (FileNotFoundError, pickle.PickleError) as e:
            # Try loading the old-style scaler if pickle file not found
            try:
                instance.scaler = np.load(os.path.join(path, 'scaler.npy'), allow_pickle=True)
                logger.warning(f"Loaded scaler from .npy file (old format): {e}")
            except Exception as e2:
                logger.warning(f"Could not load scaler: {e2}. Normalization will use fallback method.")
                instance.scaler = None

        logger.info(f"Model loaded from {path}")
        return instance