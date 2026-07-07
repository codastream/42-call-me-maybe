import logging
import os
from rich.logging import RichHandler


def get_logger(level: int = logging.ERROR) -> logging.Logger:
    """Initialize logger at desired log level"""
    if os.getenv("DEBUG") == "True":
        level = logging.DEBUG
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=level,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)]
    )
    return logging.getLogger("matcher_logger")
