import time
import json
import traceback
from typing import cast, Any, Callable, Tuple

from rich.live import Live
import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.matcher.AutomatonController import AutomatonController
from src.matcher.AutomatonDef import AState
from src.matcher.StaticSequenceMatcher import StaticSequenceMatcher
from src.exceptions import DecodingBlockedException, DecodingTimeoutException, InvalidPayloadException
from src.utils.DebugDashboard import DebugDashboard
from src.utils.StepController import StepController
from src.utils.CustomUTF8Decoder import CustomUTF8Decoder
from src.config import get_logger, get_dashboard_handler
from src.models.TypeDef import TypeDef
from src.models.DecodingStepState import DecodingStepState
from src.models.DecodingContext import DecodingContext
from src.models.DecodingMetrics import DecodingMetrics


log = get_logger()


def _init_generated(available_fun: str, current_prompt: str) -> str:
    """Initializes the generated prompt for the model with available functions and user input.

    Constructs a system prompt instructing the model to act as a function-calling router,
    and formats the full prompt with the user's input and a forced JSON prefix.

    Args:
        available_fun (str): String listing available functions for the model.
        current_prompt (str): The user's input prompt.

    Returns:
        str: The fully formatted prompt, including the forced JSON prefix for function calling.
    """
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
    """Extracts and validates the generated JSON from the output string.

    Locates the JSON substring in the generated output and parses it into a dictionary.
    Raises an exception if the JSON is invalid or missing.

    Args:
        generated (str): The raw generated output string containing JSON.

    Returns:
        dict[Any, Any]: The parsed JSON as a dictionary.

    Raises:
        InvalidPayloadException: If the JSON prefix is missing or the JSON is malformed.
    """
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
    """Extracts the relevant portion of the generated string for debugging display.

    Isolates the substring starting from the JSON prompt key for dashboard logging.

    Args:
        generated (str): The full generated output string.

    Returns:
        str: The substring starting from the JSON prompt key, or the full string if not found.
    """
    content_start_idx = generated.find('"prompt":')
    if content_start_idx != -1:
        return generated[content_start_idx:]
    return generated


def execute_with_dashboard(
    ctx: DecodingContext,
    timeout: float = 10.0,
    is_debug: bool = False,
) -> dict[Any, Any]:
    """Executes the decoding process with a live debugging dashboard.

    Sets up a `DebugDashboard` and `StepController` to visualize the decoding steps.
    Uses the `rich.live.Live` context to update the dashboard dynamically.

    Args:
        ctx (DecodingContext): The decoding context containing model, controller, and prompts.
        timeout (float, optional): Maximum execution time in seconds. Defaults to 10.0.
        is_debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        dict[Any, Any]: The final decoded output as a dictionary.

    Notes:
        The dashboard updates in real-time with the current stage, pipeline, and logs.
        User can interact with the dashboard using keyboard inputs (e.g., next, continue).
    """
    dashboard = DebugDashboard(pipeline_stages=AState._member_names_)
    step_ctrl = StepController(enabled=True)

    with Live(dashboard.layout, refresh_per_second=10, auto_refresh=False, screen=False) as live:

        def ui_callback(state: DecodingStepState, ctrl: AutomatonController) -> bool:

            handler = get_dashboard_handler()
            state.current_stage = ctrl.state._name_
            state.active_pipeline = getattr(ctrl, "pipeline", [])
            live.update(
                dashboard.update(
                    state=state,
                    step_hint="[n]ext [c]ontinue [j]ump steps [q]uit",
                    logs=list(handler.records) if handler else []
                )
            )
            live.refresh()
            step_ctrl.wait()
            return bool(step_ctrl.should_quit)

        return execute_decoding(
            ctx=ctx,
            timeout=timeout,
            is_debug=is_debug,
            on_step=ui_callback
        )


def _format_token_bytes(token_b: bytes) -> str:
    """Formats a token (in bytes) into a human-readable string for display.

    Handles special cases like spaces, newlines, and tabs, and falls back to a
    raw bytes representation if UTF-8 decoding fails.

    Args:
        token_b (bytes): The token as bytes.

    Returns:
        str: A formatted string representing the token (e.g., "␣ (space)" for space).
    """
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


def _update_metrics(state: DecodingStepState) -> DecodingStepState:
    """Updates the decoding metrics in the provided state.

    Consolidates metrics such as:
    - Count of rejected top-1 tokens.
    - Rank of the selected token.
    - Top 25 tokens with their logits, filtered logits, and ranks.

    Args:
        state (DecodingStepState): The current decoding state, including context and metrics.

    Returns:
        DecodingStepState: The updated state with consolidated metrics and top tokens data.
    """
    ctx = state.context
    metrics = state.metrics
    raw_top_1_id = int(np.argsort(state.logits)[-1])
    if raw_top_1_id not in state.authorized_token_ids:
        metrics.top1_rejected_count += 1

    known_ids = set(ctx.tokenid_to_bytes.keys())
    top_global_ids = [int(t_id) for t_id in np.argsort(state.logits)[::-1] if int(t_id) in known_ids][:25]
    sorted_global_ids = np.argsort(state.logits)[::-1]
    top_tokens_data = []
    for t_id in top_global_ids:
        rank = int(np.where(sorted_global_ids == t_id)[0][0]) + 1
        disp_token_bytes = ctx.tokenid_to_bytes[t_id]
        try:
            disp_readable = _format_token_bytes(disp_token_bytes)
        except Exception:
            disp_readable = repr(disp_token_bytes)
        top_tokens_data.append({
            'token': disp_readable,
            'logit': float(state.logits[t_id]),
            'filtered': float(state.filtered_logits[t_id]),
            'rank': rank
        })
    actual_rank = int(np.where(sorted_global_ids == state.next_token_id)[0][0]) + 1
    metrics.selected_ranks.append(actual_rank)

    state.context = ctx
    state.metrics = metrics
    state.top_tokens_data = top_tokens_data
    return state


def _add_quote_if_starting_val(
    controller: AutomatonController,
    tokenbytes_to_id: dict[bytes, int],
    input_ids: list[int]
) -> bool:
    """Automatically adds a quote token if the current parameter type is a string and the buffer is empty.

    Checks if the current parameter type is `TypeDef.STRING` and prepends a quote token
    to the input IDs if needed.

    Args:
        controller (AutomatonController): The automaton controller managing the decoding state.
        tokenbytes_to_id (dict[bytes, int]): Mapping of token bytes to their IDs.
        input_ids (list[int]): The list of input token IDs to modify.

    Returns:
        bool: True if a quote was added, False otherwise.
    """
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


def _add_expected_sequence(
    model: Small_LLM_Model,
    controller: AutomatonController,
    tokenbytes_to_id: dict[bytes, int],
    input_ids: list[int]
) -> Tuple[bool, str]:
    """Forces the addition of a static sequence's tokens to the input IDs
    if the top matcher is a `StaticSequenceMatcher`.

    Encodes the target text of the matcher and appends the corresponding token IDs to the input.
    Updates the controller's consumed bytes.

    Args:
        model (Small_LLM_Model): The language model used for encoding.
        controller (AutomatonController): The automaton controller.
        tokenbytes_to_id (dict[bytes, int]): Mapping of token bytes to their IDs.
        input_ids (list[int]): The list of input token IDs to modify.

    Returns:
        Tuple[bool, str]:
            - bool: True if a sequence was added, False otherwise.
            - str: The target text of the matcher, or an empty string if no sequence was added.
    """
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


def execute_decoding(
    ctx: DecodingContext,
    timeout: float = 10.0,
    is_debug: bool = False,
    on_step: Callable[[DecodingStepState, AutomatonController], bool] | None = None
) -> dict[Any, Any]:
    """Core decoding loop for generating and validating model outputs.

    Manages the decoding process, including:
    - Timeout handling.
    - Automatic addition of quotes or static sequences.
    - Token selection based on logits and authorized tokens.
    - Metrics updates (if in debug mode).
    - Callback execution for step-by-step debugging.

    Args:
        ctx (DecodingContext): The decoding context, including model, prompts, and controller.
        timeout (float, optional): Maximum execution time in seconds. Defaults to 10.0.
        is_debug (bool, optional): Whether to enable debug mode. Defaults to False.
        on_step (Callable[[DecodingStepState, AutomatonController], bool] | None, optional):
            Callback function for step-by-step debugging. Defaults to None.

    Returns:
        dict[Any, Any]: The final decoded output as a dictionary.

    Raises:
        DecodingTimeoutException: If the decoding exceeds the timeout.
        DecodingBlockedException: If no authorized tokens are available.
    """

    generated = _init_generated(ctx.available_fun, ctx.current_prompt)
    input_ids = ctx.model.encode(generated)[0].tolist()
    custom_utf8_decoder = CustomUTF8Decoder()
    start_time = time.time()

    metrics = DecodingMetrics()

    while not ctx.controller.is_finished:
        if (time.time() - start_time) > timeout:
            raise DecodingTimeoutException(f"timeout reached ({timeout}s)")

        if _add_quote_if_starting_val(
                controller=ctx.controller,
                tokenbytes_to_id=ctx.tokenbytes_to_id,
                input_ids=input_ids):
            generated += '"'
            continue

        is_static_state, extra = _add_expected_sequence(
            ctx.model, controller=ctx.controller, tokenbytes_to_id=ctx.tokenbytes_to_id, input_ids=input_ids)
        if is_static_state:
            generated += extra
            continue

        logits = np.array(ctx.model.get_logits_from_input_ids(input_ids))
        authorized_token_ids = ctx.controller.evaluate_tokens(ctx.tokenid_to_bytes, ctx.trie_root, ctx.value_buckets)

        if not authorized_token_ids:
            raise DecodingBlockedException(f"automata blocked at {ctx.controller.state_label} : no authorized token")

        mask = np.full_like(logits, -float('inf'))
        mask[authorized_token_ids] = 0
        filtered_logits = logits + mask

        next_token_id = int(np.argmax(filtered_logits))

        input_ids.append(next_token_id)
        token_bytes = ctx.tokenid_to_bytes[next_token_id]
        ctx.controller.consume_token_bytes(token_bytes)
        readable_chunk = custom_utf8_decoder.decode(token_bytes)

        if is_debug:
            metrics.loops += 1
            state = DecodingStepState(
                context=ctx,
                metrics=metrics,
                logits=logits,
                filtered_logits=filtered_logits,
                authorized_token_ids=authorized_token_ids,
                next_token_id=next_token_id,
                readable_chunk=readable_chunk,
                generated_text=_get_dashboard_generated(generated),
                current_stage=ctx.controller.state._name_,
                top_tokens_data=list(),
                active_pipeline=ctx.controller.pipeline
            )
            state = _update_metrics(state)

            if on_step:
                should_quit = on_step(state, ctx.controller)
                if should_quit:
                    break
        generated += readable_chunk
    extra = custom_utf8_decoder.flush()
    log.debug(f"extra decoder bytes: {extra}")
    return _output_generated_json(generated)
