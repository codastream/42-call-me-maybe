from pydantic import BaseModel


class TestDefinition(BaseModel):
    """Model for parsing a test definition"""
    prompt: str
