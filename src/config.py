import logging
import os
from collections import deque

from rich.logging import RichHandler


class DashboardLogHandler(logging.Handler):
    """In-memory log handler using a ring buffer for display in the dashboard.

    Inherits from `logging.Handler` to capture and store a fixed number of
    recent formatted log entries.

    Attributes:
        records (deque[str]): Double-ended queue storing the most recent formatted log records.
    """

    def __init__(self, maxlen: int = 5) -> None:
        """Initializes the DashboardLogHandler with a maximum buffer capacity.

        Args:
            maxlen (int, optional): Maximum number of log records to retain in memory. Defaults to 5.
        """
        super().__init__()
        self.records: deque[str] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        """Formats and appends a log record to the internal buffer.

        If formatting fails, delegates error handling to `handleError`.

        Args:
            record (logging.LogRecord): The log record to process and store.
        """
        try:
            self.records.append(self.format(record))
        except Exception:
            self.handleError(record)


_console_handler: RichHandler | None = None
_dashboard_handler: DashboardLogHandler | None = None
_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging

    Args:
        level (int, optional): log level. Defaults to logging.INFO.
    """
    global _console_handler, _dashboard_handler, _configured
    if _configured:
        return

    if os.getenv("DEBUG") == "True":
        level = logging.DEBUG

    _console_handler = RichHandler(rich_tracebacks=True, markup=True)
    _dashboard_handler = DashboardLogHandler()

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[_console_handler, _dashboard_handler],
    )
    _configured = True


def get_logger(name: str = "match") -> logging.Logger:
    """Provide logger

    Args:
        name (str, optional): logger name. Defaults to "match".

    Returns:
        logging.Logger: logger
    """
    setup_logging()
    return logging.getLogger(name)


def get_dashboard_handler() -> DashboardLogHandler | None:
    """Return DashboardLogHandler

    Returns:
        DashboardLogHandler | None: handler
    """
    setup_logging()
    if _dashboard_handler:
        return _dashboard_handler
    return None


def suspend_console_logging() -> None:
    """To be called when rich.Live is active"""
    if _console_handler in logging.root.handlers:
        logging.root.removeHandler(_console_handler)


def resume_console_logging() -> None:
    """To be called after rich.Live"""
    if _console_handler is not None and _console_handler not in logging.root.handlers:
        logging.root.addHandler(_console_handler)
