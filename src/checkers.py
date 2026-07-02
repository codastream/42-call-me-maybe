import os
from pathlib import Path

"""Check that a file can be used with requested access permissions

Args:
    file_path: path to target file
    mode: access mode

Returns:
    True if the file is accessible with specified mode
    False otherwise
"""
def check_file_permissions(file_path: str, mode: str) -> bool:
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

"""Check that a file has correct extension"""
def has_correct_extension(file_path: str, expected_ext: str) -> bool:
  path = Path(file_path)
  if path.suffix != expected_ext:
      print(f"file {file_path} does not have expected extension {expected_ext}")
      return False
  return True
