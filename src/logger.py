"""
Logging configuration for Triangle Sports Analytics project.
"""
import logging
import sys
from pathlib import Path
import config


def setup_logger(name: str, level: str = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting across the project.

    Args:
        name: Logger name (typically __name__ from calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               If None, uses config.LOG_LEVEL

    Returns:
        Configured logger instance

    Example:
        >>> from logger import setup_logger
        >>> logger = setup_logger(__name__)
        >>> logger.info("Model training started")
        2026-01-15 10:30:45 - src.train_real_data - INFO - Model training started
    """
    # Use config level if not specified
    if level is None:
        level = config.LOG_LEVEL

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger


def get_simple_logger(name: str) -> logging.Logger:
    """
    Get a simple logger that just prints messages without timestamps.

    Useful for progress messages and user-facing output where timestamps
    would be distracting.

    Args:
        name: Logger name

    Returns:
        Logger with simple format (message only)

    Example:
        >>> logger = get_simple_logger(__name__)
        >>> logger.info("✓ Model trained successfully")
        ✓ Model trained successfully
    """
    logger = logging.getLogger(f"{name}.simple")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

    return logger
