import time
import json
import traceback
from typing import cast, Any, Callable, Tuple

from rich.live import Live
import numpy as np
import numpy.typing as npt
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.matcher.AutomatonController import AutomatonController
from src.matcher.AutomatonDef import AState
from src.matcher import StaticSequenceMatcher
from src.exceptions import DecodingBlockedException, DecodingTimeoutException, InvalidPayloadException
from src.utils.DebugDashboard import DebugDashboard, DecodingStepState
from src.utils.StepController import StepController
from src.utils.CustomUTF8Decoder import CustomUTF8Decoder
from src.utils.Trie import TrieNode
from src.config import get_logger, get_dashboard_handler
from src.models.TypeDef import TypeDef


log = get_logger()


def _init_generated(available_fun: str, current_prompt: str) -> str:
    """Force part of the output"""
    log.debug(f"Current prompt = {current_prompt}")

    system_prompt = f"You are a function calling router. \
      Available functions:\n{available_fun}\n. \
      Return a JSON object with the name of the function that matches the user request.\
      The parameter VALUES must be extracted DIRECTLY and LITERALLY from the user prompt when possible."
    chat_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{current_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    escaped_prompt = json.dumps(current_prompt, ensure_ascii=False)[1:-1]
    forced_prefix = f'{{"prompt": "{escaped_prompt}", "name": "'
    generated = chat_prompt + forced_prefix
    return generated


def _output_generated_json(generated: str) -> dict[Any, Any]:
    """Return a valid JSON"""
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
    """Extract relevant generated text for debugging"""
    content_start_idx = generated.find('"prompt":')
    if content_start_idx != -1:
        return generated[content_start_idx:]
    return generated


def execute_with_dashboard(model: Small_LLM_Model,
                           current_prompt: str,
                           available_fun: str,
                           controller: AutomatonController,
                           tokenid_to_bytes: dict[int, bytes],
                           tokenbytes_to_id: dict[bytes, int],
                           trie_root: TrieNode,
                           value_buckets: dict[TypeDef, set[int]],
                           timeout: float = 10.0,
                           is_debug: bool = False,
                           ) -> dict[Any, Any]:
    """Wrapper for execution with dashboard"""

    dashboard = DebugDashboard(pipeline_stages=AState._member_names_)
    step_ctrl = StepController(enabled=True)

    with Live(dashboard.layout, refresh_per_second=10, auto_refresh=False, screen=False) as live:

        def ui_callback(state: DecodingStepState, ctrl: AutomatonController) -> bool:

            handler = get_dashboard_handler()
            logs = list(handler.records) if handler else []
            live.update(
                dashboard.update(
                    current_stage=ctrl.state._name_,
                    active_pipeline=getattr(ctrl, "pipeline", []),
                    top_tokens=state.stat_top_tokens_data,
                    loops=state.stat_loops,
                    rejected_pct=state.stat_top1_rejected_pct,
                    top1_rejected=state.stat_top1_rejected_count,
                    avg_rank=state.stat_avg_rank,
                    generated_text=_get_dashboard_generated(state.old_generated),
                    generated_added_text=state.readable_chunk,
                    step_hint="[n]ext [c]ontinue [j]ump steps [q]uit",
                    logs=logs,
                )
            )
            live.refresh()
            step_ctrl.wait()
            return bool(step_ctrl.should_quit)

        return execute_decoding(
            model=model,
            tokenid_to_bytes=tokenid_to_bytes,
            tokenbytes_to_id=tokenbytes_to_id,
            current_prompt=current_prompt,
            available_fun=available_fun,
            controller=controller,
            value_buckets=value_buckets,
            trie_root=trie_root,
            timeout=timeout,
            is_debug=is_debug,
            on_step=ui_callback
        )


def _format_token_bytes(token_b: bytes) -> str:
    """Format tokens for display with Rich"""
    try:
        decoded = token_b.decode("utf-8")
        if decoded == " ":
            return "␣ (space)"
        elif decoded == "\n":
            return "\\n (newline)"
        elif decoded == "\t":
            return "\\t (tab)"
        return repr(decoded)[1:-1]
    except UnicodeDecodeError:
        return f"bytes: {token_b!r}"


def _update_metrics(
    logits: npt.NDArray[np.int64],
    authorized_tokens_ids: list[int],
    filtered_logits: npt.NDArray[np.int64],
    generated: str,
    readable_chunk: str,
    next_token_id: int,
    tokenid_to_bytes: dict[int, bytes],
    stat_loops: int,
    stat_top1_rejected_count: int,
    stat_selected_ranks: list[int],
    controller: AutomatonController
) -> DecodingStepState:
    """Consolidate metrics into a DecodingStepState Object"""

    raw_top_1_id = int(np.argsort(logits)[-1])
    if raw_top_1_id not in authorized_tokens_ids:
        stat_top1_rejected_count += 1

    stat_top1_rejected_pc = stat_top1_rejected_count / stat_loops

    known_ids = set(tokenid_to_bytes.keys())
    top_global_ids = [int(t_id) for t_id in np.argsort(logits)[::-1] if int(t_id) in known_ids][:25]
    sorted_global_ids = np.argsort(logits)[::-1]
    top_tokens_data = []
    for t_id in top_global_ids:
        rank = int(np.where(sorted_global_ids == t_id)[0][0]) + 1
        disp_token_bytes = tokenid_to_bytes[t_id]
        try:
            disp_readable = _format_token_bytes(disp_token_bytes)
        except Exception:
            disp_readable = repr(disp_token_bytes)
        top_tokens_data.append({
            'token': disp_readable,
            'logit': float(logits[t_id]),
            'filtered': float(filtered_logits[t_id]),
            'rank': rank
        })
        # log.debug(f"token:|{disp_readable}|\t\tfiltered:{float(filtered_logits[t_id])}")
    actual_rank = int(np.where(sorted_global_ids == next_token_id)[0][0]) + 1
    stat_selected_ranks.append(actual_rank)
    stat_avg_rank = sum(stat_selected_ranks) / len(stat_selected_ranks)

    state = DecodingStepState(
        stat_loops=stat_loops,
        stat_top1_rejected_count=stat_top1_rejected_count,
        stat_top1_rejected_pct=stat_top1_rejected_pc,
        stat_top_tokens_data=top_tokens_data,
        stat_avg_rank=stat_avg_rank,
        logits=logits,
        filtered_logits=filtered_logits,
        authorized_token_ids=authorized_tokens_ids,
        next_token_id=next_token_id,
        old_generated=generated,
        readable_chunk=readable_chunk,
        current_stage=controller.state._name_,
        controller=controller,
        tokenid_to_bytes=tokenid_to_bytes,
        selected_ranks=stat_selected_ranks
    )
    return state


def _add_quote_if_starting_val(controller: AutomatonController, tokenbytes_to_id: dict[bytes, int],
                               input_ids: list[int]) -> bool:
    try:
        p_type = controller.get_current_parameter_type()
        if not p_type or controller.current_buffer_b != b"":
            return False
        if p_type == TypeDef.STRING:
            quote_token_id = tokenbytes_to_id[b'"']
            input_ids.append(quote_token_id)
            controller.consume_token_bytes(b'"')
            log.debug("automatically added \" as a string value prefix")
            return True
        return False
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        traceback.print_exc()
        return False


def _add_expected_sequence(model: Small_LLM_Model, controller: AutomatonController, tokenbytes_to_id: dict[bytes, int],
                           input_ids: list[int]) -> Tuple[bool, str]:
    top_matcher = controller._top
    if not isinstance(top_matcher, StaticSequenceMatcher):
        return False, ""
    matcher = top_matcher
    target_text = matcher.target.decode(errors="surrogateescape")
    forced_t_ids = model.encode(target_text)[0].tolist()
    input_ids.extend(forced_t_ids)
    controller.consume_token_bytes(matcher.target)
    log.debug(f"automatically added {matcher.target!r}")
    return True, target_text


def execute_decoding(model: Small_LLM_Model,
                     tokenid_to_bytes: dict[int, bytes],
                     tokenbytes_to_id: dict[bytes, int],
                     current_prompt: str,
                     available_fun: str,
                     controller: AutomatonController,
                     trie_root: TrieNode,
                     value_buckets: dict[TypeDef, set[int]],
                     timeout: float = 10.0,
                     is_debug: bool = False,
                     on_step: Callable[[DecodingStepState, AutomatonController], bool] | None = None) -> dict[Any, Any]:
    """Core decoding loop"""

    generated = _init_generated(available_fun, current_prompt)
    input_ids = model.encode(generated)[0].tolist()
    custom_utf8_decoder = CustomUTF8Decoder()
    start_time = time.time()
    stat_loops = 0
    stat_top1_rejected_count = 0
    stat_selected_ranks: Any = []

    while not controller.is_finished:
        if (time.time() - start_time) > timeout:
            raise DecodingTimeoutException(f"timeout reached ({timeout}s)")

        if _add_quote_if_starting_val(controller=controller, tokenbytes_to_id=tokenbytes_to_id, input_ids=input_ids):
            generated += '"'
            continue

        is_static_state, extra = _add_expected_sequence(
            model, controller=controller, tokenbytes_to_id=tokenbytes_to_id, input_ids=input_ids)
        if is_static_state:
            generated += extra
            continue

        logits = np.array(model.get_logits_from_input_ids(input_ids))
        authorized_tokens_ids = controller.evaluate_tokens(tokenid_to_bytes, trie_root, value_buckets)

        if not authorized_tokens_ids:
            raise DecodingBlockedException(f"automata blocked at {controller.state_label} : no authorized token")

        mask = np.full_like(logits, -float('inf'))
        mask[authorized_tokens_ids] = 0
        filtered_logits = logits + mask

        next_token_id = int(np.argmax(filtered_logits))

        input_ids.append(next_token_id)
        token_bytes = tokenid_to_bytes[next_token_id]
        controller.consume_token_bytes(token_bytes)
        readable_chunk = custom_utf8_decoder.decode(token_bytes)

        if is_debug:
            stat_loops += 1
            dec_state = _update_metrics(
                logits=logits,
                authorized_tokens_ids=authorized_tokens_ids,
                filtered_logits=filtered_logits,
                generated=generated,
                readable_chunk=readable_chunk,
                next_token_id=next_token_id,
                tokenid_to_bytes=tokenid_to_bytes,
                stat_loops=stat_loops,
                stat_top1_rejected_count=stat_top1_rejected_count,
                stat_selected_ranks=stat_selected_ranks,
                controller=controller
            )
            stat_selected_ranks = dec_state.selected_ranks
            stat_top1_rejected_count = dec_state.stat_top1_rejected_count
            if on_step:
                should_quit = on_step(dec_state, controller)
                if should_quit:
                    break
        generated += readable_chunk
    extra = custom_utf8_decoder.flush()
    log.debug(f"extra decoder bytes: {extra}")
    return _output_generated_json(generated)
