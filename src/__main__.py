
import argparse
import json
import sys

import numpy as np
from pydantic import TypeAdapter
from dotenv import load_dotenv, dotenv_values
from llm_sdk import Small_LLM_Model

from src.utils.convert import extract_and_cache_vocabulary
from src.utils.debug import debug_decoded_candidates, debug_title, debug_prompt
from src.config import get_logger
from src.checkers import check_args_paths, check_format
from src.matcher.JSONSchemaMatcher import JSONSchemaMatcher
from src.matcher.MatcherState import MatcherState
from src.models.FunctionDefinition import FunctionDefinition 
from src.models.Test import TestDefinition

#=================
# CONFIG
#=================

load_dotenv()
log = get_logger()

#=================
# FILE VALIDATION
#=================

parser = argparse.ArgumentParser(description="Function calling")
parser.add_argument('-f', '--functions_definition', type=str, help='function definition file', default='data/input/functions_definition.json')
parser.add_argument('-i', '--input', type=str, help='input file', default='data/input/function_calling_tests.json')
parser.add_argument('-o', '--output', type=str, help='output file', default='data/output/function_calls.json')

args = parser.parse_args()

try:
  defs_path, input_path, output_path = check_args_paths(args)
except ValueError as e:
  log.error(f"Error: {e}")
  sys.exit(1)

#=================
# SCHEMA VALIDATION
#=================

functions_list_adapter = TypeAdapter(list[FunctionDefinition])
tests_list_adapter = TypeAdapter(list[TestDefinition])
fun_defs = None
tests = None

try:
  fun_defs = check_format(defs_path, functions_list_adapter)
  log.info(f"functions definitions are valid")
  tests = check_format(input_path, tests_list_adapter)
  log.info(f"tests definitions are valid")
except ValueError as e:
  log.error(f"Error: {e}")
  sys.exit(1)

#===================================
# MODEL INIT AND VOCABULARY MAPPING
#===================================

model = Small_LLM_Model(local_files_only=True)
vocab_file_path = model.get_path_to_vocab_file()

VOCAB_RAW_BYTES: dict[int, bytes] = {}
VOCAB_PRINT: dict[int, str] = {}

try:
  VOCAB_RAW_BYTES, VOCAB_PRINT = extract_and_cache_vocabulary(vocab_file_path)
  log.info(f"Loaded {len(VOCAB_RAW_BYTES)} optimized tokens into cache maps.")
except (ValueError, Exception) as e:
  log.error(f"Error: {e}")
  sys.exit(1)

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

  log.info(f"Current prompt = {current_prompt}")

  forced_prefix = f'{{"prompt": "{current_prompt}", "name": "'
  generated = chat_prompt + forced_prefix
  input_ids = model.encode(generated)[0].tolist()

  matcher = JSONSchemaMatcher(fun_defs=fun_defs, initial_prompt=current_prompt.encode('utf-8'))
  matcher.state = MatcherState.EXPECT_FUN_NAME
  matcher.current_buffer = b""
 
  debug_title("function name..")

  while matcher.state != MatcherState.FINISH:
    
    logits = np.array(model.get_logits_from_input_ids(input_ids))
    authorized_tokens = []

    for token_id, token_str in VOCAB_PRINT.items():
      if token_str is None:
        continue
      if matcher.evaluate_token(token_str):
        authorized_tokens.append(token_id)
    if not authorized_tokens:
      log.error(f"Error: decoding blocked : no valid vocabulary for state {matcher.state}")
      break

    mask = np.full_like(logits, -float('inf'))
    mask[authorized_tokens] = 0
    filtered_logits = logits + mask

    debug_decoded_candidates(matcher.state, authorized_tokens, logits, filtered_logits, model)

    next_token = int(np.argmax(filtered_logits))
    input_ids.append(next_token)
    token_str = model.decode([next_token])
    matcher.consume_token(token_str)
    generated += token_str

    debug_prompt(generated)

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
