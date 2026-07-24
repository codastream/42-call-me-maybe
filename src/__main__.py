from llm_sdk import Small_LLM_Model
from src.utils.convert import extract_and_cache_vocabulary
from src.config import get_logger
from src.checkers import check_args_paths, check_format
from src.models import FunctionDefinition, TestDefinition
from src.decode import execute_with_dashboard, execute_decoding
from src.exceptions import DecodingException
from src.matcher import AutomatonController
import argparse
import json
import sys
import time
import os

from typing import Tuple, Any

from pydantic import TypeAdapter
from dotenv import load_dotenv
load_dotenv()


def validate_files() -> Tuple[str, str, str]:
    """Validate input files"""

    log = get_logger()
    parser = argparse.ArgumentParser(description="Function calling")
    parser.add_argument('--functions_definition', type=str, help='function definition file',
                        default='data/input/functions_definition.json')
    parser.add_argument('--input', type=str, help='input file', default='data/input/function_calling_tests.json')
    parser.add_argument('--output', type=str, help='output file', default='data/output/function_calls.json')
    args = parser.parse_args()

    try:
        defs_path, input_path, output_path = check_args_paths(args)
        return defs_path, input_path, output_path
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


def validate_schema(defs_path: str, input_path: str) -> Tuple[list[FunctionDefinition], list[TestDefinition]]:
    """Validate and return function definitions and test prompts"""

    log = get_logger()
    functions_list_adapter = TypeAdapter(list[FunctionDefinition])
    tests_list_adapter = TypeAdapter(list[TestDefinition])
    fun_defs = None
    tests = None

    try:
        fun_defs = check_format(defs_path, functions_list_adapter)
        log.info("functions definitions are valid")
        tests = check_format(input_path, tests_list_adapter)
        log.info("tests definitions are valid")
        return fun_defs, tests
    except ValueError as e:
        log.error(f"Error: {e}")
        sys.exit(1)


def init_model_and_vocabulary() -> Tuple[Small_LLM_Model, dict[int, bytes], dict[int, str]]:
    """Initialize model and vocabulary maps"""
    log = get_logger()
    model = Small_LLM_Model(local_files_only=True)
    vocab_file_path = model.get_path_to_vocab_file()

    dic_id_to_bytes: dict[int, bytes] = {}
    dic_id_to_print: dict[int, str] = {}

    try:
        dic_id_to_bytes, dic_id_to_print = extract_and_cache_vocabulary(vocab_file_path)
        log.info(f"Loaded {len(dic_id_to_bytes)} optimized tokens into cache maps.")
        return model, dic_id_to_bytes, dic_id_to_print
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


def generate_results(fun_defs: list[FunctionDefinition], model: Small_LLM_Model,
                     dic_id_to_bytes: dict[int, bytes]) -> list[dict[Any, Any]]:
    """Decode and return json output"""
    log = get_logger()
    outputs = []
    available_fun = "\n".join([f"- {f.name}: {f.description}" for f in fun_defs])

    total_start = time.time()

    for test in tests[:]:
        try:
            log.info(f"processing prompt: {test.prompt}")
            controller = AutomatonController(fun_defs=fun_defs, initial_prompt=test.prompt.encode())
            if os.getenv("DEBUG") == "True":
                json_obj = execute_with_dashboard(
                    model=model,
                    current_prompt=test.prompt,
                    tokenid_to_bytes=dic_id_to_bytes,
                    available_fun=available_fun,
                    controller=controller,
                    timeout=180,
                    is_debug=True
                )
                log.debug(f"json obj = {json_obj}")
            else:
                json_obj = execute_decoding(
                    model=model,
                    current_prompt=test.prompt,
                    tokenid_to_bytes=dic_id_to_bytes,
                    available_fun=available_fun,
                    controller=controller,
                    timeout=180,
                    is_debug=False
                )
            outputs.append(json_obj)
        except DecodingException as e:
            log.error(f"Decoding error for prompt '{test.prompt}': {e}")
        except KeyboardInterrupt as e:
            log.error(f"Decoding interrupted (Ctrl + C) : {e}")
            exit(1)
        except Exception:
            log.error("Unexpected error")

    total_finish = time.time()
    log.info(f"All inputs decoded within {total_finish - total_start:.2f}s")
    return outputs


def write_outputs(output_path: str, outputs: list[dict[Any, Any]]) -> None:
    """Write outputs"""
    log = get_logger()
    try:
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(outputs, out_file, indent=2, ensure_ascii=False)
            log.info(f"Output has been written in {output_path}")
    except Exception as e:
        log.error(f"Unexpected error while writing output: {e}")


if __name__ == "__main__":
    defs_path, input_path, output_path = validate_files()
    fun_defs, tests = validate_schema(defs_path, input_path)
    model,  dic_id_to_bytes, dic_id_to_print = init_model_and_vocabulary()
    outputs = generate_results(fun_defs, model, dic_id_to_bytes)
    write_outputs(output_path, outputs)
