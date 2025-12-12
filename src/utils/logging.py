"""Centralized logging configuration for KnightGPT."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# Remove default logger
logger.remove()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path for file logging
        json_format: Whether to use JSON format for logs
    """
    # Console format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # JSON format for structured logging
    json_log_format = (
        '{{"time": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
        '"level": "{level}", '
        '"name": "{name}", '
        '"function": "{function}", '
        '"line": {line}, '
        '"message": "{message}"}}'
    )

    # Add console handler
    logger.add(
        sys.stderr,
        format=json_log_format if json_format else console_format,
        level=level,
        colorize=not json_format,
        backtrace=True,
        diagnose=True,
    )

    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            format=json_log_format,
            level=level,
            rotation="100 MB",
            retention="1 week",
            compression="gz",
            backtrace=True,
            diagnose=True,
        )

    logger.info(f"Logging configured at level {level}")


def get_logger(name: str):
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)


# Default setup
setup_logging()
