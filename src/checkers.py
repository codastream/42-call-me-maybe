import os
import argparse
import json
from pathlib import Path
from typing import Tuple
from pydantic import TypeAdapter, ValidationError

def _check_file_permissions(file_path: str, mode: str) -> bool:
  """Check that a file can be used with requested access permissions

  Args:
      file_path: path to target file
      mode: access mode

  Returns:
      True if the file is accessible with specified mode
      False otherwise
  """
  path = Path(file_path)
  if mode == "r":
    if not path.exists():
      print(f"File {file_path} was not found.")
      return False
    if not os.access(path, os.R_OK):
      print(f"Error : {file_path} cannot be read.")
      return False 
  
  elif mode == "w":
    parent_dir = path.parent
    if not parent_dir.exists():
      print(f"Output directory {parent_dir} does not exist.")
      return False
    if not os.access(parent_dir, os.W_OK):
      print(f"Output directory {parent_dir} does not exist.")
      print(f"Error : {parent_dir} cannot be written into.")
      return False
  return True

def _has_correct_extension(file_path: str, expected_ext: str) -> bool:
  """Check that a file has correct extension"""
  path = Path(file_path)
  if path.suffix != expected_ext:
      print(f"file {file_path} does not have expected extension {expected_ext}")
      return False
  return True

#==============
# MAIN METHODS
#==============

def check_args_paths(args: argparse.Namespace) -> Tuple[str, str, str]:
  """Check file paths and permissions
  
  Raises:
    ValueError: if a file path is invalid or not accessible
  """
  if not args.functions_definition or not args.input or not args.output:
    raise ValueError("Some arguments are missing or empty")

  # check definitions
  defs_path = args.functions_definition.strip()
  is_def_valid = _check_file_permissions(defs_path, "r") and _has_correct_extension(defs_path, ".json")
  if not is_def_valid:
    raise ValueError(f"Invalid function definition file: {defs_path}")

  # check input file
  input_path = args.input.strip()
  is_in_valid = _check_file_permissions(input_path, "r") and _has_correct_extension(input_path, ".json")
  if not is_in_valid:
    raise ValueError(f"Invalid input file: {input_path}")

  # check output file
  output_path = args.output.strip()
  is_out_valid = _check_file_permissions(output_path, "w")
  if not is_out_valid:
    raise ValueError(f"Output path has no write permission: {output_path}")

  return (defs_path, input_path, output_path)

def check_format(path:str, adapter: TypeAdapter) -> bool:
  """Check that json content follows model
  
  Raises:
    ValueError: if model does not match content
  """
  try:
    with open(path, "r") as file:
      raw = json.load(file)
      defs = adapter.validate_python(raw)
      return defs
  except json.JSONDecodeError:
    raise ValueError(f"Error: {path} does not have a valid JSON structure.")
  except ValidationError as e:
    raise ValueError(f"Error: {path} does not match function definition schema.")
  except Exception as e:
    raise ValueError(f"Unexpected error while reading {path} : {e}")
