
import argparse
import json
import logging
import os
from rich import print as rprint
from rich.logging import RichHandler
import numpy as np
from pydantic import TypeAdapter, ValidationError
from src.JSONSchemaMatcher import JSONSchemaMatcher
from src.MatcherState import MatcherState
from src.checkers import check_file_permissions, has_correct_extension
from src.models.FunctionDefinition import FunctionDefinition 
from src.models.Test import TestDefinition
from llm_sdk import Small_LLM_Model
from dotenv import load_dotenv, dotenv_values

#=================
# LOG CONFIG
#=================

load_dotenv()

FORMAT = "%(message)s"
logging.basicConfig(
    level="DEBUG", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

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
    log.info(f"functions definitions are valid")
except json.JSONDecodeError:
  log.error(f"Error: {defs_path} does not have a valid JSON structure.")
  exit(1)
except ValidationError as e:
  log.error(f"Error: {defs_path} does not match function definition schema.")
  exit(1)
except Exception as e:
  log.error(f"Unexpected error while reading {defs_path} : {e}")
  exit(1)

try:
  with open(input_path, "r") as file:
    raw = json.load(file)
    tests = test_adapter.validate_python(raw)
    log.info(f"tests definitions are valid")
except json.JSONDecodeError:
  log.error(f"Error: {input_path} does not have a valid JSON structure.")
  exit(1)
except ValidationError as e:
  log.error(f"Error: {input_path} does not match test input schema.")
  exit(1)
except Exception as e:
  log.error(f"Unexpected error while reading {input_path} : {e}")
  exit(1)

#======================
# EXTRACT VOCABULARY
#======================

model = Small_LLM_Model(local_files_only=True)
vocab_file_path = model.get_path_to_vocab_file()
try: 
  with open(vocab_file_path, "r", encoding="utf-8") as vocab_file:
    raw_vocab = json.load(vocab_file)
  vocab_map: dict[int, str] = {int(token_id): token_str for token_str, token_id in raw_vocab.items()}
  logging.info("Vocab loaded")
except json.JSONDecodeError:
  log.error("Error: could not decode model vocabulary file")
except Exception as e:
  log.error(f"Unexpected error while extracting vocabulary: {e}")

#======================
# UTILS
#======================

def debug_decoded_candidates(context: str, candidates_tokens: list[int], model: Small_LLM_Model) -> None:
  if os.getenv("DEBUG") == "True":
    decoded_candidates = [model.decode([tok]) for tok in candidates_tokens]
    rprint(f"authorized tokens ids for {context}: {candidates_tokens}")
    rprint(f"decoded tokens for {context}[bold cyan]:", decoded_candidates)

def print_step(name:str) -> None:
  if os.getenv("DEBUG") == "True":
    rprint(f"\n[bold yellow on black] === { name.upper() } === [/bold yellow on black]\n")

#======================
# GENERATE RESULTS
#======================

outputs = []
fun_names = [f.name for f in fun_defs]

for test in tests[0:1]:
  current_fun_name = None
  current_param_key = None
  available_fun = "\n".join([f"- {f.name}: {f.description}" for f in fun_defs])
  system_prompt = f"You are a function calling router. Available functions:\n{available_fun}\n. Return a JSON object with the name of the function that matches the user request."
  current_prompt = test.prompt
  chat_prompt = (
    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    f"<|im_start|>user\n{current_prompt}<|im_end|>\n"
    f"<|im_start|>assistant\n"
  )

  logging.info(f"Current prompt = {current_prompt}")

  forced_prefix = f'{{"prompt": "{current_prompt}", "name": "'
  generated = chat_prompt + forced_prefix
  input_ids = model.encode(generated)[0].tolist()

  matcher = JSONSchemaMatcher(fun_defs=fun_defs, initial_prompt=current_prompt.encode('utf-8'))
  matcher.state = MatcherState.EXPECT_FUN_NAME
  matcher.current_buffer = b""
 
  print_step("function name..")

  while matcher.state != MatcherState.FINISH:
    
    logits = np.array(model.get_logits_from_input_ids(input_ids))
    authorized_tokens = []

    for token_id, token_str in vocab_map.items():
      if token_str is None:
        continue
      if matcher.evaluate_token(token_str):
        authorized_tokens.append(token_id)
    if not authorized_tokens:
      log.error(f"Error: decoding blocked : no valid vocabulary for state {matcher.state}")
      break
    debug_decoded_candidates(matcher.state, authorized_tokens, model)

    mask = np.full_like(logits, -float('inf'))
    mask[authorized_tokens] = 0
    filtered_logits = logits + mask

    next_token = int(np.argmax(filtered_logits))
    input_ids.append(next_token)
    token_str = model.decode([next_token])
    matcher.consume_token(token_str)
    generated += token_str

    rprint(f"Generated: [bold green]{generated}[/bold green]")

  json_start_idx = generated.find(f'{{"prompt":')
  if json_start_idx != -1:
    json_str = generated[json_start_idx:]
    try:
      json_obj = json.loads(json_str)
      outputs.append(json_obj)
    except json.JSONDecodeError as e: 
      log.error(f"Error: invalid JSON generated while decoding with prompt {current_prompt}: {e}")
    except Exception as e:
      log.error(f"Unexpected error while decoding: {e}")
  else:
    log.error(f"Error: Could not find user payload prefix")

#======================
# WRITE OUTPUT
#======================

try:
  with open(output_path, "w", encoding="utf=8") as out_file:
    json.dump(outputs, out_file, indent=2, ensure_ascii=False)
    log.info(f"Output has been written in {output_path}")
except Exception as e:
  log.error(f"Unexpected error while writing output: {e}")
