from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict

class TypeDef(Enum):
  NUMBER = "number"
  STRING = "string"
  BOOLEAN = "bool"

  def _validate_buffer_type(self, buf: str, char: str) -> bool:
    """Determine if buffer + char maintain a consistent type"""

    if self == TypeDef.NUMBER:
      if char in (",", "}"):
        val_to_test = buf[:-1].strip()
        try:
          float(val_to_test)
          return True
        except ValueError:
          return False
      return all(c in "0123456789.-" for c in buf.strip())

    elif self == TypeDef.BOOLEAN:
      if char in (",", "}"):
        val_to_test = buf[:-1].strip()
        return val_to_test in ("true", "false")
      return "true".startswith(buf) or "false".startswith(buf)
      
    elif self == TypeDef.STRING:
      val_to_test = buf.strip()
      if char in (",", "}"):
        closed_str = buf[:-1].strip()
        if closed_str.startswith('"') and closed_str.endswith('"') and len(closed_str) > 1:
          if len(closed_str) == 2 or closed_str[-2] != '\\':
            return True
        return False
    
      if val_to_test.startswith('"'):
        if val_to_test.endswith('"') and len(val_to_test) > 1:
          if len(val_to_test) == 2 or val_to_test[-2] != '\\':
            return False
          return True
    return False

class ParameterDef(BaseModel):
  type: TypeDef

class ReturnDef(BaseModel):
  type: TypeDef

class FunctionDefinition(BaseModel):
  name: str
  description: str
  parameters: Dict[str, ParameterDef] = Field(default_factory=dict)
  returns: ReturnDef
