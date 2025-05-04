"""
Configuration Utilities for TensorTrade

This module provides functions for loading and validating configuration files.
"""
import os
import logging
from typing import Dict, Any, Optional
import yaml

# Setup logger
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        logger.warning("Using default configuration")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        logger.warning("Using default configuration")
        return {}


def validate_config(config: Dict[str, Any], required_fields: Dict[str, Any]) -> bool:
    """
    Validate that the configuration contains all required fields

    Args:
        config: Configuration dictionary
        required_fields: Dictionary with required fields and their default values

    Returns:
        True if valid, False otherwise
    """
    is_valid = True

    for section, fields in required_fields.items():
        if section not in config:
            logger.warning(f"Missing configuration section: {section}")
            config[section] = {}
            is_valid = False

        for field, default_value in fields.items():
            if field not in config[section]:
                logger.warning(f"Missing configuration field: {section}.{field}")
                config[section][field] = default_value
                is_valid = False

    return is_valid


def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation

    Args:
        config: Configuration dictionary
        path: Configuration path using dot notation (e.g., 'api.host')
        default: Default value to return if path not found

    Returns:
        Configuration value or default if not found
    """
    parts = path.split('.')
    current = config

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]

    return current


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries

    Args:
        base_config: Base configuration
        override_config: Configuration with values to override

    Returns:
        Merged configuration
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file

    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")