"""Centralized logging configuration for trade-bot."""

import logging
import logging.handlers
import os

from trade_bot.core.config import LOG_DIR, LOG_LEVEL

_FMT = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_initialized: set = set()


def get_logger(name: str) -> logging.Logger:
    """Return a logger with both console and rotating-file handlers.

    Safe to call multiple times for the same name — handlers are added only once.
    """
    logger = logging.getLogger(name)
    if name in _initialized:
        return logger

    _initialized.add(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    if not logger.handlers:
        # Console
        ch = logging.StreamHandler()
        ch.setFormatter(_FMT)
        logger.addHandler(ch)

        # Rotating file (10 MB × 5 backups)
        os.makedirs(LOG_DIR, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(LOG_DIR, "trade_bot.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(_FMT)
        logger.addHandler(fh)

    return logger
