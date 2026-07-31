from pydantic import BaseModel, Field
from typing import Dict

from src.models.TypeDef import TypeDef
from src.models.validation import NonEmptyString


class ParameterDef(BaseModel):
    """Model for a parameter object

    Args:
        BaseModel : pydantic BaseModel
    """
    type: TypeDef


class ReturnDef(BaseModel):
    """Model for a returns object

    Args:
        BaseModel : pydantic BaseModel
    """
    type: TypeDef


class FunctionDefinition(BaseModel):
    """Model of a function definition object

    Args:
        BaseModel : pydantic BaseModel
    """
    name: NonEmptyString
    description: NonEmptyString
    parameters: Dict[NonEmptyString, ParameterDef] = Field(default_factory=dict)
    returns: ReturnDef
