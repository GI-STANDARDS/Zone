"""
Logging configuration for Face Recognition System
"""
import logging
import os
from pathlib import Path
from typing import Optional

from config.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logger(
    name: str = "face_recognition",
    log_file: Optional[Path] = None,
    level: str = LOG_LEVEL,
    format_string: str = LOG_FORMAT
) -> logging.Logger:
    """
    Set up logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Log format string
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log file specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "face_recognition") -> logging.Logger:
    """
    Get logger instance
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize main logger
main_logger = setup_logger(
    name="face_recognition",
    log_file=LOG_FILE,
    level=LOG_LEVEL,
    format_string=LOG_FORMAT
)
