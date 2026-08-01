from src.utils.Trie import TrieNode
from src.matcher.AutomatonController import AutomatonController
from src.exceptions import DecodingException
from src.decode import execute_with_dashboard, execute_decoding
from src.models.FunctionDefinition import FunctionDefinition
from src.models.TestDefinition import TestDefinition
from src.models.AppConfig import AppConfig
from src.models.DecodingContext import DecodingContext
from src.models.TypeDef import TypeDef
from src.checkers import check_format
from src.config import get_logger
from src.utils.convert import extract_and_cache_vocabulary, build_value_buckets
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from pathlib import Path
import argparse
import json
import sys
import time
import os
import traceback

from typing import Tuple, Any, Annotated

from pydantic import TypeAdapter, ValidationError, Field
from dotenv import load_dotenv
load_dotenv()


def validate_files() -> Tuple[Path, Path, Path]:
    """Parses CLI arguments and validates input/output file paths.

    Returns:
        Tuple[Path, Path, Path]: Resolved paths to functions definition file,
            input test prompts file, and output JSON file.

    Raises:
        SystemExit: If CLI argument validation fails.
    """
    log = get_logger()
    parser = argparse.ArgumentParser(description="Function calling")
    parser.add_argument('--functions_definition', type=str, help='function definition file',
                        default='data/input/functions_definition.json')
    parser.add_argument('--input', type=str, help='input file', default='data/input/function_calling_tests.json')
    parser.add_argument('--output', type=str, help='output file', default='data/output/function_calls.json')
    args = parser.parse_args()

    try:
        config = AppConfig.model_validate(vars(args))
        return config.functions_definition_file, config.input_file, config.output_file
    except ValidationError as e:
        log.error(f"Configuration error\n: {e}")
        sys.exit(1)


def validate_schema(defs_path: Path, input_path: Path) -> Tuple[list[FunctionDefinition], list[TestDefinition]]:
    """Validates schemas for function definitions and test prompt files.

    Args:
        defs_path (Path): Path to the function definitions JSON file.
        input_path (Path): Path to the test prompts JSON file.

    Returns:
        Tuple[list[FunctionDefinition], list[TestDefinition]]: Parsed and validated
            function definitions and test prompt instances.

    Raises:
        SystemExit: If JSON content fails schema validation.
    """
    log = get_logger()
    NonEmptyFunctionList = Annotated[list[FunctionDefinition], Field(min_length=1)]
    NonEmptyTestList = Annotated[list[TestDefinition], Field(min_length=1)]
    functions_list_adapter: TypeAdapter[NonEmptyFunctionList] = TypeAdapter(NonEmptyFunctionList)
    tests_list_adapter: TypeAdapter[NonEmptyTestList] = TypeAdapter(NonEmptyTestList)
    fun_defs = None
    tests = None

    try:
        fun_defs = check_format(defs_path, functions_list_adapter)
        log.info("functions definitions are valid")
        tests = check_format(input_path, tests_list_adapter)
        log.info("tests definitions are valid")
        return fun_defs, tests
    except ValueError as e:
        log.error(f"Schema validation error for inputs\n: {e}")
        sys.exit(1)


def init_model_and_vocabulary() -> Tuple[Small_LLM_Model, dict[int, bytes], dict[bytes, int], dict[int, str],
                                         TrieNode, dict[TypeDef, set[int]]]:
    """Initializes the LLM, extracts token vocabulary mappings, and precomputes search structures.

    Returns:
        Tuple[Small_LLM_Model, dict[int, bytes], dict[bytes, int], dict[int, str], TrieNode, dict[TypeDef, set[int]]]:
            - Small_LLM_Model: Loaded LLM instance.
            - dict[int, bytes]: Token ID to byte sequence mapping.
            - dict[bytes, int]: Byte sequence to Token ID mapping.
            - dict[int, str]: Token ID to printable string representation.
            - TrieNode: Root of vocabulary byte Trie.
            - dict[TypeDef, set[int]]: Pre-categorized token IDs grouped by literal TypeDef.

    Raises:
        SystemExit: If vocabulary loading or model initialization fails.
    """
    log = get_logger()
    model = Small_LLM_Model()
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
                     trie_root: TrieNode, value_buckets: dict[TypeDef, set[int]]) -> list[dict[Any, Any]]:
    """Executes constrained decoding on all input prompts and gathers output JSON results.

    Args:
        fun_defs (list[FunctionDefinition]): Available function definitions.
        tests (list[TestDefinition]): Prompts to process.
        model (Small_LLM_Model): Target language model.
        dic_id_to_bytes (dict[int, bytes]): Mapping from token ID to raw bytes.
        dic_bytes_to_id (dict[bytes, int]): Mapping from raw bytes to token ID.
        trie_root (TrieNode): Vocabulary Trie root.
        value_buckets (dict[TypeDef, set[int]]): Token IDs pre-categorized by type.

    Returns:
        list[dict[Any, Any]]: List of generated function call payloads parsed as dictionaries.
    """
    log = get_logger()
    outputs = []
    available_fun = "\n".join([f"- {f.name}: {f.description}" for f in fun_defs])

    total_start = time.time()

    for test in tests[:]:
        try:
            log.info(f"processing prompt: {test.prompt}")
            controller = AutomatonController(fun_defs=fun_defs, value_buckets=value_buckets)
            context = DecodingContext(
                model=model,
                tokenid_to_bytes=dic_id_to_bytes,
                tokenbytes_to_id=dic_bytes_to_id,
                trie_root=trie_root,
                value_buckets=value_buckets,
                controller=controller,
                current_prompt=test.prompt,
                available_fun=available_fun
            )

            if os.getenv("DEBUG") == "True":
                json_obj = execute_with_dashboard(
                    ctx=context,
                    timeout=180,
                    is_debug=True
                )
                log.debug(f"json obj = {json_obj}")
            else:
                json_obj = execute_decoding(
                    ctx=context,
                    timeout=180,
                    is_debug=True
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


def write_outputs(output_path: Path, outputs: list[dict[Any, Any]]) -> None:
    """Writes JSON decoding results to the specified output file path.

    Args:
        output_path (Path): File path where outputs should be saved.
        outputs (list[dict[Any, Any]]): List of generated result payloads.
    """
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
