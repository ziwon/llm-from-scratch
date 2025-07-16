"""Logging utilities."""

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# Global console for rich output
console = Console()


def setup_logger(
    name: str = "llm_from_scratch",
    level: str = "INFO",
    log_file: Path | None = None,
    use_rich: bool = True,
) -> logging.Logger:
    """
    Setup logger with rich formatting.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        use_rich: Whether to use rich formatting

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    if use_rich:
        console_handler = RichHandler(console=console, show_time=True, show_path=False)
    else:
        console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if not use_rich:
        console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "llm_from_scratch") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)
