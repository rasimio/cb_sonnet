"""
Logging Utilities for TensorTrade

This module provides functions for setting up and configuring logging.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from typing import Dict, Any, Optional


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Setup logging configuration

    Args:
        config: Configuration dictionary with logging settings
    """
    if config is None:
        config = {}

    # Get logging config
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_config.get('file', 'logs/tensortrade.log')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    max_file_size = log_config.get('max_file_size', 10 * 1024 * 1024)  # 10 MB
    backup_count = log_config.get('backup_count', 5)

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Create file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Log initial message
    logging.info(f"Logging initialized at level {log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """
    Set the log level for the root logger

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper())
    logging.getLogger().setLevel(log_level)

    # Update handler levels
    for handler in logging.getLogger().handlers:
        handler.setLevel(log_level)

    logging.info(f"Log level set to {level}")


def log_exception(logger: logging.Logger, e: Exception, message: str = "An error occurred") -> None:
    """
    Log an exception with traceback

    Args:
        logger: Logger instance
        e: Exception to log
        message: Message to log with the exception
    """
    import traceback
    logger.error(f"{message}: {str(e)}")
    logger.debug(traceback.format_exc())


def log_system_info() -> None:
    """Log system information for debugging purposes"""
    import platform
    import psutil
    import sys
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    logger = logging.getLogger(__name__)

    try:
        # Log system info
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"CPU: {platform.processor()}")
        logger.info(f"CPU Cores: {psutil.cpu_count(logical=False)}")
        logger.info(f"RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")

        # Log library versions
        logger.info(f"NumPy: {np.__version__}")
        logger.info(f"Pandas: {pd.__version__}")
        logger.info(f"TensorFlow: {tf.__version__}")
        logger.info(f"TF GPU Available: {tf.config.list_physical_devices('GPU')}")
    except Exception as e:
        logger.warning(f"Error logging system info: {str(e)}")