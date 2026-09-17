# ### FILE: app/core/logging.py
"""
Structured Telemetry and Logging Infrastructure for Smart Notes.
Configures uniform console output with timestamps, modules, and log levels.
"""

import logging
import sys
from typing import Optional


def setup_logging(log_level: str = "INFO", debug: bool = False) -> None:
    """Configure system-wide root logger format and handlers."""
    level = logging.DEBUG if debug else getattr(logging, log_level.upper(), logging.INFO)

    log_format = (
        "[%(asctime)s] [%(process)d] [%(levelname)s] [%(name)s:%(lineno)d] "
        "- %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Reset any existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Silence overly chatty third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if debug else logging.WARNING
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger configured with root application parameters."""
    return logging.getLogger(name or "smart_notes")
