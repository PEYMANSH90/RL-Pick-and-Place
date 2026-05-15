"""Logging configuration helper."""

from __future__ import annotations

import logging
import sys


def get_logger(
    name: str | None = None,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%H:%M:%S",
) -> logging.Logger:
    """Configure and return a logger with a stdout StreamHandler.

    Calling this once from a script's ``main()`` configures the root logger
    so every ``logging.getLogger(__name__)`` call in the library receives
    the same handler and format.

    Parameters
    ----------
    name:
        Logger name.  ``None`` configures the root logger (recommended for
        top-level scripts).
    level:
        Minimum severity to display.  Pass ``logging.DEBUG`` for verbose
        output during development.
    fmt / datefmt:
        Log record format strings.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured — avoid duplicate handlers

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
