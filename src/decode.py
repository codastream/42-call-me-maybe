import logging
import time
import json
from typing import cast, Any

from src.matcher import JSONSchemaMatcher, MatcherState
from src.exceptions import DecodingBlockedException, DecodingTimeoutException, InvalidPayloadException
from src.utils import debug_decoded_candidates, debug_prompt
from src.models import FunctionDefinition

import numpy as np
from llm_sdk import Small_LLM_Model


def _init_generated(available_fun: str, current_prompt: str) -> str:
    log = logging.getLogger("matcher_logger")
    log.debug(f"Current prompt = {current_prompt}")

    system_prompt = f"You are a function calling router. \
      Available functions:\n{available_fun}\n. \
      Return a JSON object with the name of the function that matches the user request."
    chat_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{current_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    forced_prefix = f'{{"prompt": "{current_prompt}", "name": "'
    generated = chat_prompt + forced_prefix
    return generated


def _output_generated_json(generated: str) -> dict:
    log = logging.getLogger("matcher_logger")
    prefix = '{\"prompt\":'
    json_start_idx = generated.find(prefix)
    if json_start_idx == -1:
        raise InvalidPayloadException(f"could not find {prefix} in generated output")
    json_str = generated[json_start_idx:]
    log.debug(f"json_str : {json_str}")
    try:
        return cast(dict[Any, Any], (json.loads(json_str)))
    except json.JSONDecodeError as e:
        raise InvalidPayloadException(f"invalid JSON generated : {e}. Content: {json_str}")


def execute_decoding(model: Small_LLM_Model, fun_defs: list[FunctionDefinition], vocab_print: dict[int, str],
                     current_prompt: str, available_fun: str,
                     timeout: float = 10.0) -> dict:
    """Execute decoding for a given prompt

    Returns:
      JSON dict

    Raises:
      Decoding Exception when timeout, no authorized tokens or invalid json
    """

    generated = _init_generated(available_fun, current_prompt)
    input_ids = model.encode(generated)[0].tolist()

    matcher = JSONSchemaMatcher(fun_defs=fun_defs, initial_prompt=current_prompt.encode('utf-8'))
    matcher.state = MatcherState.EXPECT_FUN_NAME
    matcher.current_buffer = b""

    start_time = time.time()

    while matcher.state != MatcherState.FINISH:

        if (time.time() - start_time) > timeout:
            raise DecodingTimeoutException(f"timeout reached ({timeout}s)")

        logits = np.array(model.get_logits_from_input_ids(input_ids))

        authorized_tokens = [t_id for t_id, t_str in vocab_print.items() if t_str and matcher.evaluate_token(t_str)]
        if not authorized_tokens:
            raise DecodingBlockedException(f"automata blocked at {matcher.state} : no authorized token")

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

    return _output_generated_json(generated)
