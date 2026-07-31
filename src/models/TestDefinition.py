from pydantic import BaseModel

from src.models.validation import NonEmptyString


class TestDefinition(BaseModel):
    """Model for a test definition object

    Args:
        BaseModel : pydantic BaseModel
    """
    prompt: NonEmptyString
