import logging
import time
import json
import codecs
from typing import cast, Any

from src.matcher import AutomatonController
from src.exceptions import DecodingBlockedException, DecodingTimeoutException, InvalidPayloadException
from src.utils import debug_decoded_candidates
from src.models import FunctionDefinition
from src.utils.DebugDashboard import DebugDashboard
from src.matcher.AutomatonDef import AState
from rich.live import Live
from src.utils.StepController import StepController
from src.config import get_logger, suspend_console_logging, resume_console_logging, get_dashboard_handler
from src.matcher import ValueMatcher

import numpy as np
from llm_sdk import Small_LLM_Model

log = get_logger()

def _init_generated(available_fun: str, current_prompt: str) -> str:
    log.debug(f"Current prompt = {current_prompt}")

    system_prompt = f"You are a function calling router. \
      Available functions:\n{available_fun}\n. \
      Return a JSON object with the name of the function that matches the user request."
    chat_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{current_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    escaped_prompt = json.dumps(current_prompt)[1:-1]
    forced_prefix = f'{{"prompt": "{escaped_prompt}", "name": "'
    generated = chat_prompt + forced_prefix
    return generated


def _output_generated_json(generated: str) -> dict:
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


def _get_dashboard_generated(generated: str) -> str:
    content_start_idx = generated.index('"prompt":')
    if content_start_idx != -1:
        return generated[content_start_idx:]
    return generated

def execute_decoding(model: Small_LLM_Model, fun_defs: list[FunctionDefinition],
                     tokenid_to_print: dict[int, str],
                     tokenid_to_bytes: dict[int, bytes],
                     current_prompt: str, available_fun: str, matcher: AutomatonController,
                     timeout: float = 10.0, is_debug: bool = True) -> dict:
    """Execute decoding for a given prompt

    Returns:
      JSON dict

    Raises:
      Decoding Exception when timeout, no authorized tokens or invalid json
    """

    generated = _init_generated(available_fun, current_prompt)
    input_ids = model.encode(generated)[0].tolist()
    utf8_decoder = codecs.getincrementaldecoder('utf-8')(errors='strict')

    loops = 0
    # total_rejected_tokens = 0
    # total_possible_tokens = len(tokenid_to_bytes.keys())
    top1_rejected_count = 0
    selected_ranks = []
    MIN_WIDTH, MIN_HEIGHT = 80, 24
    dashboard = DebugDashboard(pipeline_stages=AState._member_names_)
    if dashboard.console.size.width < MIN_WIDTH or dashboard.console.size.height < MIN_HEIGHT:
      raise RuntimeError(
          f"Terminal too small ({dashboard.console.size}), "
          f"minimum {MIN_WIDTH}x{MIN_HEIGHT}"
      )
    step_ctrl = StepController(enabled=is_debug)

    start_time = time.time()

    # suspend_console_logging()
    try:
      with Live(dashboard.layout, refresh_per_second=10, auto_refresh=False, screen=False) as live:
          while not matcher.is_finished:
              loops += 1
              if (time.time() - start_time) > timeout:
                  raise DecodingTimeoutException(f"timeout reached ({timeout}s)")
              logits = np.array(model.get_logits_from_input_ids(input_ids))


              authorized_tokens_ids = matcher.evaluate_tokens(tokenid_to_bytes)
              
              # [t_id for t_id, t_str in tokenid_to_print.items(
              # ) if t_str and matcher.evaluate_token_bytes(tokenid_to_bytes[t_id])]
              if not authorized_tokens_ids:
                  # resume_console_logging()
                  raise DecodingBlockedException(f"automata blocked at {matcher.state_label} : no authorized token")

              # step_rejected = total_possible_tokens - len(authorized_tokens_ids)
              # total_rejected_tokens += step_rejected

              raw_top_1_id = int(np.argsort(logits)[-1])
              if raw_top_1_id not in authorized_tokens_ids:
                  top1_rejected_count += 1
              rejected_pct = top1_rejected_count / loops

              mask = np.full_like(logits, -float('inf'))
              mask[authorized_tokens_ids] = 0
              filtered_logits = logits + mask
              # debug_decoded_candidates(matcher.state_label, authorized_tokens_ids, logits, filtered_logits, model)

              known_ids = set(tokenid_to_bytes.keys())
              top_global_ids = [int(t_id) for t_id in np.argsort(logits)[::-1] if int(t_id) in known_ids][:10]
              sorted_global_ids = np.argsort(logits)[::-1]
              top_tokens_data = []
              for t_id in top_global_ids:
                  rank = int(np.where(sorted_global_ids == t_id)[0][0]) + 1
                  disp_token_bytes = tokenid_to_bytes[t_id]
                  try:
                    disp_readable = disp_token_bytes.decode("utf-8", errors="replace")
                  except Exception:
                    disp_readable = repr(disp_token_bytes)
                  top_tokens_data.append({
                      'token': disp_readable,
                      'logit': float(logits[t_id]),
                      'filtered': float(filtered_logits[t_id]),
                      'rank': rank
                  })

              next_token_id = int(np.argmax(filtered_logits))

              actual_rank = int(np.where(sorted_global_ids == next_token_id)[0][0]) + 1
              selected_ranks.append(actual_rank)
              avg_rank = sum(selected_ranks) / len(selected_ranks)

              input_ids.append(next_token_id)
              token_bytes = tokenid_to_bytes[next_token_id]
              matcher.consume_token_bytes(token_bytes)
              readable_chunk = utf8_decoder.decode(token_bytes)
              old_generated = generated
              generated += readable_chunk

              # debug_automaton_state(matcher)
              # debug_stack(matcher)
              # debug_prompt(generated)

              live.update(
                  dashboard.update(
                      current_stage=matcher.state._name_,
                      current_stage_label=matcher.state_label,
                      active_pipeline=getattr(matcher, "pipeline", []),
                      top_tokens=top_tokens_data,
                      loops=loops,
                      rejected_pct=rejected_pct,
                      top1_rejected=top1_rejected_count,
                      avg_rank=avg_rank,
                      generated_text=_get_dashboard_generated(old_generated),
                      generated_added_text=readable_chunk,
                      step_hint="[n]ext [c]ontinue [j]ump steps [q]uit",
                      logs=list(get_dashboard_handler().records),
                  )
              )
              live.refresh()
              step_ctrl.wait()
              if step_ctrl.should_quit:
                break
    except KeyboardInterrupt:
        live.stop()
        raise

    return _output_generated_json(generated)
