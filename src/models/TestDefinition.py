from pydantic import BaseModel


class TestDefinition(BaseModel):
    prompt: str
