
from src.matcher import AutomatonController
from src.exceptions import DecodingException
from src.decode import execute_with_dashboard
from src.models import FunctionDefinition, TestDefinition
from src.checkers import check_args_paths, check_format
from src.config import get_logger
from src.utils.convert import extract_and_cache_vocabulary
from llm_sdk import Small_LLM_Model
import argparse
import json
import sys
import time
import logging
import os

from pydantic import TypeAdapter
from dotenv import load_dotenv
load_dotenv()


# =================
# CONFIG
# =================

log = get_logger()
print("DEBUG env =", os.getenv("DEBUG"), file=sys.stderr)
print("root level =", logging.getLogger().getEffectiveLevel(), file=sys.stderr)

# =================
# FILE VALIDATION
# =================

parser = argparse.ArgumentParser(description="Function calling")
parser.add_argument('-functions_definition', type=str, help='function definition file',
                    default='data/input/functions_definition.json')
parser.add_argument('-input', type=str, help='input file', default='data/input/function_calling_tests.json')
parser.add_argument('-output', type=str, help='output file', default='data/output/function_calls.json')

args = parser.parse_args()

try:
    defs_path, input_path, output_path = check_args_paths(args)
except Exception as e:
    log.error(f"Error: {e}")
    sys.exit(1)

# =================
# SCHEMA VALIDATION
# =================

functions_list_adapter = TypeAdapter(list[FunctionDefinition])
tests_list_adapter = TypeAdapter(list[TestDefinition])
fun_defs = None
tests = None

try:
    fun_defs = check_format(defs_path, functions_list_adapter)
    log.info("functions definitions are valid")
    tests = check_format(input_path, tests_list_adapter)
    log.info("tests definitions are valid")
except ValueError as e:
    log.error(f"Error: {e}")
    sys.exit(1)

# ===================================
# MODEL INIT AND VOCABULARY MAPPING
# ===================================

model = Small_LLM_Model(local_files_only=True)
vocab_file_path = model.get_path_to_vocab_file()

TOKID_TO_BYTES: dict[int, bytes] = {}
TOKID_TO_PRINT: dict[int, str] = {}

try:
    TOKID_TO_BYTES, TOKID_TO_PRINT = extract_and_cache_vocabulary(vocab_file_path)
    log.info(f"Loaded {len(TOKID_TO_BYTES)} optimized tokens into cache maps.")
except (ValueError, Exception) as e:
    log.error(f"Error: {e}")
    sys.exit(1)

# ======================
# GENERATE RESULTS
# ======================

outputs = []
available_fun = "\n".join([f"- {f.name}: {f.description}" for f in fun_defs])

total_start = time.time()

for test in tests[:]:

    try:
        log.info(f"processing prompt: {test.prompt}")
        controller = AutomatonController(fun_defs=fun_defs, initial_prompt=test.prompt.encode())
        json_obj = execute_with_dashboard(
            model=model,
            current_prompt=test.prompt,
            tokenid_to_bytes=TOKID_TO_BYTES,
            available_fun=available_fun,
            controller=controller,
            timeout=180,
            is_debug=True
        )
        log.debug(f"json obj = {json_obj}")
        outputs.append(json_obj)
    except DecodingException as e:
        log.error(f"Decoding error for prompt '{test.prompt}': {e}")
    except KeyboardInterrupt as e:
        log.error(f"Decoding interrupted (Ctrl + C) : {e}")
    except Exception:
        log.error("Unexpected error")

# ======================
# WRITE OUTPUT
# ======================

total_finish = time.time()
log.info(f"All inputs decoded within {total_finish - total_start:.2f}s")

try:
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(outputs, out_file, indent=2, ensure_ascii=False)
        log.info(f"Output has been written in {output_path}")
except Exception as e:
    log.error(f"Unexpected error while writing output: {e}")
