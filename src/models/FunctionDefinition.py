from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict
import logging


def is_quoted_string(s: str) -> bool:
    """Return true if the string is surrounded by unescaped double quotes"""
    s = s.strip()
    if len(s) < 2:
        return False
    return s.startswith('"') and s.endswith('"') and s[-2] != '\\'


class TypeDef(Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "bool"

    def _validate_buffer_type(self, buf: str, char: str) -> bool:
        """Determine if buffer + char maintain a consistent type"""

        logging.getLogger("matcher_logger")
        val_to_test = buf.strip()
        if not val_to_test:
            return True

        if self == TypeDef.NUMBER:
            if char in (",", "}"):
                val_to_test = buf[:-1].strip()
                if not val_to_test:
                    return False
            return all(c in "0123456789.-" for c in val_to_test)

        elif self == TypeDef.BOOLEAN:
            if char in (",", "}"):
                val_to_test = buf[:-1].strip()
                return val_to_test in ("true", "false")
            return "true".startswith(val_to_test) or "false".startswith(val_to_test)

        elif self == TypeDef.STRING:
            if val_to_test.startswith('"'):
                quote_count = val_to_test.count('"') - val_to_test.count('\\"')
                if quote_count == 1:
                    return True
                if quote_count == 2:
                    return val_to_test.endswith('"') and (len(val_to_test) == 2 or val_to_test[-2] != '\\')
            return False
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
