import json
import logging
import unittest
import time
import argparse
import sys
from typing import Any, Callable

from pydantic import TypeAdapter
import numpy as np
from llm_sdk import Small_LLM_Model

from src.matcher import JSONSchemaMatcher, MatcherState
from src.models import FunctionDefinition, TestDefinition
from src.utils.convert import extract_and_cache_vocabulary
from src.decode import execute_decoding
from src.exceptions import DecodingException, DecodingBlockedException, DecodingTimeoutException, InvalidPayloadException

log = logging.getLogger("unittest")


class TestConstrainedDecoding(unittest.TestCase):
    """Base class"""

    funcs_path: str
    input_path: str
    expected_path: str
    timeout_limit: float
    fun_defs: list[FunctionDefinition]
    expected_map: dict[str, dict[str, Any]]
    model: Small_LLM_Model
    VOCAB_PRINT: dict[int, str]
    available_fun: str

    @classmethod
    def setUpClass(cls) -> None:
        """Shared init"""

        cls.funcs_path = 'data/input/functions_definition.json'
        cls.input_path = 'data/input/function_calling_tests.json'
        cls.expected_path = 'data/input/function_calling_expected.json'
        cls.timeout_limit = 180

        with open(cls.funcs_path, 'r', encoding='utf-8') as f:
            cls.fun_defs = TypeAdapter(list[FunctionDefinition]).validate_python(json.load(f))

        cls.available_fun = "\n".join([f"- {f.name}: {f.description}" for f in cls.fun_defs])

        with open(cls.expected_path, 'r', encoding='utf-8') as f:
            expected_data = json.load(f)
            log.debug(f"expected_data = {expected_data}")
            cls.expected_map = {
                prompt: data
                for item in expected_data
                for prompt, data in item.items()
              }

        cls.model = Small_LLM_Model(local_files_only=True)
        _, cls.VOCAB_PRINT = extract_and_cache_vocabulary(cls.model.get_path_to_vocab_file())


def make_test_method(test_definition: TestDefinition) -> Callable[[Any], None]:
    """Factory for unit test method"""

    def test_method(self: "TestConstrainedDecoding") -> None:
        current_prompt = test_definition.prompt

        expected_json = self.expected_map.get(current_prompt)
        self.assertIsNotNone(
            expected_json,
            msg=f"Error: Please declare an expected result for prompt {current_prompt}"
        )

        start_time = time.time()

        try:
          log.info(f"processing prompt: {current_prompt}")
          
          matcher = JSONSchemaMatcher(fun_defs=self.fun_defs, initial_prompt=current_prompt.encode('utf-8'))
          
          received_json = execute_decoding(
            model=self.model,
            fun_defs=self.fun_defs,
            vocab_print=self.VOCAB_PRINT,
            current_prompt=current_prompt,
            available_fun=self.available_fun,
            matcher=matcher,
            timeout=self.timeout_limit
          )
          
          elapsed_time = time.time() - start_time
          log.debug(f"received = {received_json}")
          log.info(f"test executed in {elapsed_time}")
          
          ## Function name
          self.assertEqual(expected_json["name"], received_json["name"], "Incorrect function name")
          
          ## Parameters
          self.assertEqual(
            len(expected_json["parameters"]), 
            len(received_json["parameters"]), 
            "Mismatched number of parameters generated"
          )

          for param_key, expected_val in expected_json["parameters"].items():
            self.assertIn(
              param_key,
              received_json["parameters"],
              f"Missing expected parameter: '{param_key}'"
            )
          received_params = received_json["parameters"].copy()
          if "regex" in received_params and isinstance(received_params["regex"], str):
            val = received_params["regex"]
            if val.startswith("(") and val.endswith(")"):
              received_params["regex"] = val[1:-1]
          
          self.assertEqual(expected_json["parameters"], received_params, "Incorrect parameters mapping")

        except DecodingTimeoutException as e:
          self.fail(f"❌ TIMEOUT detected after {self.timeout_limit}s")
        except DecodingBlockedException as e:
          self.fail(f"❌ DECODING BLOCKED. No valid token at step {matcher.state}")
        except InvalidPayloadException as e:
          self.fail(f"❌ INVALID JSON. {e}")
    
    return test_method

def csv_and_json_injector() -> None:
    """Load test file and inject methods before execution"""
    input_path = 'data/input/function_calling_tests.json'

    try:
      with open(input_path, 'r', encoding="utf-8") as f:
          tests = TypeAdapter(list[TestDefinition]).validate_python(json.load(f))
      for idx, test_def in enumerate(tests):
        clean_name = "".join([c if c.isalnum() else "_" for c in test_def.prompt[:30]])
        if idx < 10:
          method_name = f"test_prompt_0{idx}_{clean_name}"
        else:
          method_name = f"test_prompt_{idx}_{clean_name}"
        test_function = make_test_method(test_def)
        setattr(TestConstrainedDecoding, method_name, test_function)
    except Exception as e:
      logging.basicConfig(level=logging.DEBUG)
      log.exception(f"❌ Critical error during dynamic test injection: {e}")

csv_and_json_injector()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--log-level', type=str, default='ERROR',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args, remaining_argv = parser.parse_known_args()

    numeric_level = getattr(logging, args.log_level.upper(), logging.ERROR)
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')
    log.setLevel(numeric_level)

    sys.argv = [sys.argv[0]] + remaining_argv
    
    unittest.main()
