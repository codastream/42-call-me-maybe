
import argparse
import json
import sys

from pydantic import TypeAdapter
from dotenv import load_dotenv
from llm_sdk import Small_LLM_Model

from src.utils.convert import extract_and_cache_vocabulary
from src.config import get_logger
from src.checkers import check_args_paths, check_format
from src.models import FunctionDefinition, TestDefinition
from src.decode import execute_decoding
from src.exceptions import DecodingException

# =================
# CONFIG
# =================

load_dotenv()
log = get_logger()

# =================
# FILE VALIDATION
# =================

parser = argparse.ArgumentParser(description="Function calling")
parser.add_argument('-f', '--functions_definition', type=str, help='function definition file',
                    default='data/input/functions_definition.json')
parser.add_argument('-i', '--input', type=str, help='input file', default='data/input/function_calling_tests.json')
parser.add_argument('-o', '--output', type=str, help='output file', default='data/output/function_calls.json')

args = parser.parse_args()

try:
    defs_path, input_path, output_path = check_args_paths(args)
except ValueError as e:
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

VOCAB_RAW_BYTES: dict[int, bytes] = {}
VOCAB_PRINT: dict[int, str] = {}

try:
    VOCAB_RAW_BYTES, VOCAB_PRINT = extract_and_cache_vocabulary(vocab_file_path)
    log.info(f"Loaded {len(VOCAB_RAW_BYTES)} optimized tokens into cache maps.")
except (ValueError, Exception) as e:
    log.error(f"Error: {e}")
    sys.exit(1)

# ======================
# GENERATE RESULTS
# ======================

outputs = []
available_fun = "\n".join([f"- {f.name}: {f.description}" for f in fun_defs])

for test in tests[0:1]:

    try:
        log.info(f"processing prompt: {test.prompt}")
        json_obj = execute_decoding(model, fun_defs, VOCAB_PRINT, test.prompt, available_fun, timeout=30)
        log.debug(f"json obj = {json_obj}")
        outputs.append(json_obj)
    except DecodingException as e:
        log.error(f"Decoding error for prompt '{test.prompt}': {e}")

# ======================
# WRITE OUTPUT
# ======================

try:
    with open(output_path, "w", encoding="utf=8") as out_file:
        json.dump(outputs, out_file, indent=2, ensure_ascii=False)
        log.info(f"Output has been written in {output_path}")
except Exception as e:
    log.error(f"Unexpected error while writing output: {e}")
