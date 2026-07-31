from pydantic import BaseModel, Field
from typing import Dict

from src.models.TypeDef import TypeDef
from src.models.validation import NonEmptyString


class ParameterDef(BaseModel):
    type: TypeDef


class ReturnDef(BaseModel):
    type: TypeDef


class FunctionDefinition(BaseModel):
    """Model for parsing function definitions"""
    name: NonEmptyString
    description: NonEmptyString
    parameters: Dict[NonEmptyString, ParameterDef] = Field(default_factory=dict)
    returns: ReturnDef
