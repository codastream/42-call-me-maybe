from pydantic import BaseModel

from src.models.validation import NonEmptyString


class TestDefinition(BaseModel):
    """Model for parsing a test definition"""
    prompt: NonEmptyString
