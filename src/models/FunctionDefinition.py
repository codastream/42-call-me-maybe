from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict

class TypeDef(Enum):
  NUMBER = "number"
  STRING = "string"

class ParameterDef(BaseModel):
  type: TypeDef

class ReturnDef(BaseModel):
  type: TypeDef

class FunctionDefinition(BaseModel):
  name: str
  description: str
  parameters: Dict[str, ParameterDef] = Field(default_factory=dict)
  returns: ReturnDef
