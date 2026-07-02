
import argparse
import json
from pydantic import TypeAdapter, ValidationError
from collections.abc import Mapping
from src.checkers import check_file_permissions, has_correct_extension
from src.models.FunctionDefinition import FunctionDefinition 
from src.models.Test import TestDefinition
from llm_sdk import Small_LLM_Model

#=================
# FILE VALIDATION
#=================

parser = argparse.ArgumentParser(description="Function calling")
parser.add_argument('-f', '--functions_definition', type=str, help='function definition file', default='data/input/functions_definition.json')
parser.add_argument('-i', '--input', type=str, help='input file', default='data/input/function_calling_tests.json')
parser.add_argument('-o', '--output', type=str, help='output file', default='data/output/function_calls.json')

args = parser.parse_args()

# check definitions
defs_path = args.functions_definition.strip()
is_def_valid = check_file_permissions(defs_path, "r") and has_correct_extension(defs_path, ".json")
if not is_def_valid:
  exit()

# check input file
input_path = args.input.strip()
is_in_valid = check_file_permissions(input_path, "r") and has_correct_extension(input_path, ".json")
if not is_in_valid:
  exit()

# check output file
output_path = args.output.strip()
is_out_valid = check_file_permissions(output_path, "w")
if not is_out_valid:
  exit()

#=================
# MODEL VALIDATION
#=================

functions_list_adapter = TypeAdapter(list[FunctionDefinition])
test_adapter = TypeAdapter(list[TestDefinition])
fun_defs = None
tests = None

try:
  with open(defs_path, "r") as file:
    raw = json.load(file)
    fun_defs = functions_list_adapter.validate_python(raw)
    print(f"functions definitions are valid")
except json.JSONDecodeError:
  print(f"Error: {defs_path} does not have a valid JSON structure.")
  exit(1)
except ValidationError as e:
  print(f"Error: {defs_path} does not match function definition schema.")
  exit(1)
except Exception as e:
  print(f"Unexpected error while reading {defs_path} : {e}")
  exit(1)



try:
  with open(input_path, "r") as file:
    raw = json.load(file)
    tests = test_adapter.validate_python(raw)
    print(f"tests definitions are valid")
except json.JSONDecodeError:
  print(f"Error: {input_path} does not have a valid JSON structure.")
  exit(1)
except ValidationError as e:
  print(f"Error: {input_path} does not match test input schema.")
  exit(1)
except Exception as e:
  print(f"Unexpected error while reading {input_path} : {e}")
  exit(1)


#======================
# EXTRACT VOCABULARY
#======================

model = Small_LLM_Model()
vocab_file_path = model.get_path_to_vocab_file()
try: 
  with open(vocab_file_path, "r", encoding="utf-8") as vocab_file:
    raw_vocab = json.load(vocab_file)
  vocab_map: dict[int, str] = {int(token_id): token_str for token_str, token_id in raw_vocab.items()}
except json.JSONDecodeError:
  print("Error: could not decode model vocabulary file")
except Exception as e:
  print(f"Unknown error: {e}")

func_name = ""
func_args = {}