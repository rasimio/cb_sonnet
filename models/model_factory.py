"""
Model Factory for TensorTrade

This module provides factory functions for creating and loading
different types of trading models.
"""
import os
import logging
from typing import Dict, Any, Optional, Union

from models.lstm_model import LSTMModel
from models.rl_model import RLModel

logger = logging.getLogger(__name__)


def create_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> Union[LSTMModel, RLModel]:
    """
    Create a new model instance of the specified type

    Args:
        model_type: Type of model to create ('lstm', 'rl', or 'ensemble')
        config: Configuration dictionary for the model

    Returns:
        Model instance
    """
    if config is None:
        config = {}

    logger.info(f"Creating {model_type} model with config: {config}")

    if model_type.lower() == 'lstm':
        return LSTMModel(config)
    elif model_type.lower() == 'rl':
        return RLModel(config)
    elif model_type.lower() == 'ensemble':
        # For ensemble, create both models and wrap them
        return create_ensemble_model(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def load_model(model_path: str) -> Union[LSTMModel, RLModel]:
    """
    Load a trained model from disk

    Args:
        model_path: Path to the model directory

    Returns:
        Loaded model instance
    """
    # Determine model type from directory name
    model_dir = os.path.basename(model_path)
    model_type = model_dir.split('_')[0].lower()

    logger.info(f"Loading {model_type} model from {model_path}")

    if model_type == 'lstm':
        return LSTMModel.load(model_path)
    elif model_type == 'rl':
        return RLModel.load(model_path)
    elif model_type == 'ensemble':
        return load_ensemble_model(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple models
    """

    def __init__(self, lstm_model: LSTMModel, rl_model: RLModel, config: Dict[str, Any] = None):
        """
        Initialize ensemble model with LSTM and RL models

        Args:
            lstm_model: LSTM model instance
            rl_model: RL model instance
            config: Ensemble configuration
        """
        self.lstm_model = lstm_model
        self.rl_model = rl_model
        self.config = config or {}

        # Set weights for each model (default: equal weighting)
        self.lstm_weight = self.config.get('lstm_weight', 0.5)
        self.rl_weight = self.config.get('rl_weight', 0.5)

        # Normalization to ensure weights sum to 1
        total_weight = self.lstm_weight + self.rl_weight
        self.lstm_weight /= total_weight
        self.rl_weight /= total_weight

        logger.info(f"Created ensemble model with weights: LSTM={self.lstm_weight}, RL={self.rl_weight}")

    def train(self, data, **kwargs):
        """
        Train both models

        Args:
            data: Training data
            **kwargs: Additional training parameters
        """
        # First train LSTM model
        logger.info("Training LSTM model in ensemble")
        lstm_history = self.lstm_model.train(data, **kwargs)

        # Then train RL model
        logger.info("Training RL model in ensemble")
        rl_history = self.rl_model.train(data, **kwargs)

        return {
            'lstm': lstm_history,
            'rl': rl_history
        }

    def predict(self, data):
        """
        Generate prediction using ensemble of models

        Args:
            data: Input data

        Returns:
            Ensemble prediction
        """
        # Get LSTM prediction
        lstm_pred = self.lstm_model.predict(data)

        # Get RL prediction
        rl_pred = self.rl_model.predict(data)

        # Determine combined prediction based on model types
        ensemble_method = self.config.get('ensemble_method', 'voting')

        if ensemble_method == 'voting':
            # For voting, convert predictions to buy/sell/hold signals
            lstm_signal = self._get_signal_from_lstm(lstm_pred)
            rl_signal = self._get_signal_from_rl(rl_pred)

            # Voting logic
            if lstm_signal == rl_signal:
                final_signal = lstm_signal
            else:
                # If there's disagreement, use the model with higher confidence or weight
                if self.lstm_weight > self.rl_weight:
                    final_signal = lstm_signal
                else:
                    final_signal = rl_signal

            return {
                'ensemble_signal': final_signal,
                'lstm_prediction': lstm_pred,
                'rl_prediction': rl_pred,
                'lstm_signal': lstm_signal,
                'rl_signal': rl_signal
            }

        elif ensemble_method == 'weighted':
            # This is for weighted averaging of predictions when both models predict same scale values
            # Only works when both models predict the same type of output
            if isinstance(lstm_pred, (int, float)) and isinstance(rl_pred, (int, float)):
                ensemble_pred = lstm_pred * self.lstm_weight + rl_pred * self.rl_weight
                return ensemble_pred
            else:
                # If predictions are not comparable, use voting method
                return self.predict(data)

        else:
            raise ValueError(f"Unsupported ensemble method: {ensemble_method}")

    def _get_signal_from_lstm(self, lstm_pred):
        """Convert LSTM prediction to trading signal"""
        if isinstance(lstm_pred, dict) and 'signal' in lstm_pred:
            return lstm_pred['signal']

        if self.lstm_model.config.get('target_type') == 'price':
            # If predicting price, return BUY if predicted price is higher than current
            return "BUY" if lstm_pred > 0 else "SELL"
        elif self.lstm_model.config.get('target_type') == 'return':
            # If predicting return, return BUY if predicted return is positive
            return "BUY" if lstm_pred > 0 else "SELL"
        else:
            # If predicting direction, return BUY if predicted direction is up
            return "BUY" if lstm_pred > 0.5 else "SELL"

    def _get_signal_from_rl(self, rl_pred):
        """Convert RL prediction to trading signal"""
        if isinstance(rl_pred, dict) and 'action' in rl_pred:
            return rl_pred['action']

        # If RL prediction is a numeric value (unusual), treat positive as BUY
        if isinstance(rl_pred, (int, float)):
            return "BUY" if rl_pred > 0 else "SELL"

        # Default case
        return "HOLD"

    def save(self, path):
        """
        Save the ensemble model

        Args:
            path: Directory path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save LSTM model
        lstm_path = os.path.join(path, 'lstm')
        self.lstm_model.save(lstm_path)

        # Save RL model
        rl_path = os.path.join(path, 'rl')
        self.rl_model.save(rl_path)

        # Save ensemble config
        import json
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.config, f)

        logger.info(f"Ensemble model saved to {path}")


def create_ensemble_model(config: Dict[str, Any] = None) -> EnsembleModel:
    """
    Create an ensemble model with LSTM and RL

    Args:
        config: Configuration dictionary

    Returns:
        Ensemble model instance
    """
    if config is None:
        config = {}

    # Extract sub-configs for each model
    lstm_config = config.get('lstm_config', {})
    rl_config = config.get('rl_config', {})
    ensemble_config = config.get('ensemble_config', {})

    # Create individual models
    lstm_model = LSTMModel(lstm_config)
    rl_model = RLModel(rl_config)

    # Create and return ensemble
    return EnsembleModel(lstm_model, rl_model, ensemble_config)


def load_ensemble_model(model_path: str) -> EnsembleModel:
    """
    Load an ensemble model from disk

    Args:
        model_path: Path to the model directory

    Returns:
        Loaded ensemble model
    """
    # Load LSTM model
    lstm_path = os.path.join(model_path, 'lstm')
    lstm_model = LSTMModel.load(lstm_path)

    # Load RL model
    rl_path = os.path.join(model_path, 'rl')
    rl_model = RLModel.load(rl_path)

    # Load ensemble config
    import json
    config_path = os.path.join(model_path, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create and return ensemble
    return EnsembleModel(lstm_model, rl_model, config)