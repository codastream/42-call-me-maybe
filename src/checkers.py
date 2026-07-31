import json
from pathlib import Path
from typing import TypeVar
from pydantic import TypeAdapter, ValidationError


T = TypeVar("T")


def check_format(path: Path, adapter: TypeAdapter[T]) -> T:
    """Check that json content follows model

    Raises:
      ValueError: if model does not match content
    """
    try:
        with open(path, "r") as file:
            raw = json.load(file)
            return adapter.validate_python(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Error: {path} does not have a valid JSON structure.")
    except ValidationError:
        raise ValueError(f"Error: {path} does not match function definition schema.")
    except Exception as e:
        raise ValueError(f"Unexpected error while reading {path} : {e}")
