
from typing_extensions import Annotated

from pydantic import AfterValidator


def non_empty_str(v: str) -> str:
    if not v or not v.strip():
        raise ValueError("field cannot be empty")
    return v


NonEmptyString = Annotated[str, AfterValidator(non_empty_str)]
