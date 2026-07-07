class DecodingException(Exception):
    """Base exception"""


class DecodingBlockedException(DecodingException):
    """Raised when no available token"""


class DecodingTimeoutException(DecodingException):
    """Raised when timed out"""


class InvalidPayloadException(DecodingException):
    """Raised when input or generated JSON format is invalid"""
