from enum import Enum


class TypeDef(Enum):
    """Possible json types"""
    INTEGER = "integer"
    FLOAT = "float"
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    NULL = "null"
