from pydantic import BaseModel, Field
from typing import Dict

from src.models.TypeDef import TypeDef


class ParameterDef(BaseModel):
    type: TypeDef


class ReturnDef(BaseModel):
    type: TypeDef


class FunctionDefinition(BaseModel):
    """Model for parsing function definitions"""
    name: str
    description: str
    parameters: Dict[str, ParameterDef] = Field(default_factory=dict)
    returns: ReturnDef
