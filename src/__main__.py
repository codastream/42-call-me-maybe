from src.utils.Trie import TrieNode
from src.matcher import AutomatonController
from src.exceptions import DecodingException
from src.decode import execute_with_dashboard, execute_decoding
from src.models import FunctionDefinition, TestDefinition
from src.models.TypeDef import TypeDef
from src.checkers import check_args_paths, check_format
from src.config import get_logger
from src.utils.convert import extract_and_cache_vocabulary, build_value_buckets
from llm_sdk import Small_LLM_Model
import argparse
import json
import sys
import time
import os
import traceback

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


def init_model_and_vocabulary() -> Tuple[Small_LLM_Model, dict[int, bytes], dict[bytes, int], dict[int, str],
                                         TrieNode, dict[TypeDef, list[int]]]:
    """Initialize model and vocabulary maps"""
    log = get_logger()
    model = Small_LLM_Model(local_files_only=True)
    vocab_file_path = model.get_path_to_vocab_file()

    dic_id_to_bytes: dict[int, bytes] = {}
    dic_id_to_print: dict[int, str] = {}
    dic_bytes_to_id: dict[bytes, int] = {}

    try:
        dic_id_to_bytes, dic_id_to_print, dic_bytes_to_id = extract_and_cache_vocabulary(vocab_file_path)
        log.info(f"Loaded {len(dic_id_to_bytes)} optimized tokens into cache maps.")
        root = TrieNode.build_vocab_trie(dic_id_to_bytes)
        value_type_buckets = build_value_buckets(dic_id_to_bytes)
        return model, dic_id_to_bytes, dic_bytes_to_id, dic_id_to_print, root, value_type_buckets
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


def generate_results(fun_defs: list[FunctionDefinition], tests: list[TestDefinition],  model: Small_LLM_Model,
                     dic_id_to_bytes: dict[int, bytes], dic_bytes_to_id: dict[bytes, int],
                     trie_root: TrieNode, value_buckets: dict[TypeDef, list[int]]) -> list[dict[Any, Any]]:
    """Decode and return json output"""
    log = get_logger()
    outputs = []
    available_fun = "\n".join([f"- {f.name}: {f.description}" for f in fun_defs])

    total_start = time.time()

    for test in tests[8:]:
        try:
            log.info(f"processing prompt: {test.prompt}")
            controller = AutomatonController(fun_defs=fun_defs, initial_prompt=test.prompt.encode())
            if os.getenv("DEBUG") == "True":
                json_obj = execute_with_dashboard(
                    model=model,
                    current_prompt=test.prompt,
                    tokenid_to_bytes=dic_id_to_bytes,
                    tokenbytes_to_id=dic_bytes_to_id,
                    available_fun=available_fun,
                    controller=controller,
                    trie_root=trie_root,
                    value_buckets=value_buckets,
                    timeout=180,
                    is_debug=True
                )
                log.debug(f"json obj = {json_obj}")
            else:
                json_obj = execute_decoding(
                    model=model,
                    current_prompt=test.prompt,
                    tokenid_to_bytes=dic_id_to_bytes,
                    tokenbytes_to_id=dic_bytes_to_id,
                    available_fun=available_fun,
                    controller=controller,
                    trie_root=trie_root,
                    value_buckets=value_buckets,
                    timeout=180,
                    is_debug=False
                )
            outputs.append(json_obj)
        except DecodingException as e:
            log.error(f"Decoding error for prompt '{test.prompt}': {e}")
        except KeyboardInterrupt as e:
            log.error(f"Decoding interrupted (Ctrl + C) : {e}")
            exit(1)
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            traceback.print_exc()

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
    model,  dic_id_to_bytes, dic_bytes_to_id, dic_id_to_print, root, value_type_buckets = init_model_and_vocabulary()
    outputs = generate_results(fun_defs, tests, model, dic_id_to_bytes, dic_bytes_to_id, root, value_type_buckets)
    write_outputs(output_path, outputs)
