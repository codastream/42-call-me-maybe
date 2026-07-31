
from typing_extensions import Annotated

from pydantic import AfterValidator


def non_empty_str(v: str) -> str:
    """validator for a non empty string

    Args:
        v (str): value
    Raises:
        ValueError: when value is empty

    Returns:
        str: validated value
    """
    if not v or not v.strip():
        raise ValueError("field cannot be empty")
    return v


NonEmptyString = Annotated[str, AfterValidator(non_empty_str)]
