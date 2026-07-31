from pathlib import Path
import os

from pydantic import BaseModel, Field, FilePath, field_validator, model_validator


class AppConfig(BaseModel):
    """Program arguments

    Args:
        BaseModel: base pydantic model
    """

    functions_definition_file: FilePath = Field(..., alias="functions_definition")
    input_file: FilePath = Field(..., alias="input")
    output_file: Path = Field(..., alias="output")

    @field_validator("functions_definition_file", "input_file", mode="before")
    def clean_and_check_extension(cls, v: str) -> Path:
        v = v.strip()
        path = Path(v)
        if path.suffix != ".json":
            raise ValueError(f"file must have a json extension: {path}")
        return path

    @model_validator(mode='after')
    def validate_output_and_create_dir(self) -> "AppConfig":
        out = self.output_file

        if out.is_dir():
            raise ValueError(f"output path is a directory: {out}")
        out_dir = out.parent
        if not out_dir.exists():
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise ValueError(f"cannot create output directory {out_dir}")
        if not os.access(out_dir, os.W_OK):
            raise ValueError(f"{out} has no write permission")
        return self
