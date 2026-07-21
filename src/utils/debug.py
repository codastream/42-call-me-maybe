import numpy as np
from rich.table import Table
from rich.text import Text
from rich.markup import escape
from rich import print as rprint
from llm_sdk import Small_LLM_Model
from typing import Any

from src.matcher import TokenMatcher

IS_DEBUG = True
PIPELINE_STAGES = ["FUN_NAME", "PARAM_PREFIX", "PARAM_KEY_OR_VAL", "CLOSE"]


def isExpectedMatcher(m: TokenMatcher, expected: str) -> bool:
    name = type(m).__name__
    if name == expected:
        return True
    return False


def isChoiceMatcher(m: TokenMatcher) -> bool:
    return isExpectedMatcher(m, "ChoiceMatcher")


def isStaticSequenceMatcher(m: TokenMatcher) -> bool:
    return isExpectedMatcher(m, "StaticSequenceMatcher")


def isParamMatcher(m: TokenMatcher) -> bool:
    return isExpectedMatcher(m, "ParamMatcher")


def _matcher_stage_label(m: TokenMatcher) -> str:
    """Display state name according to TokenMatcher type"""
    if isChoiceMatcher(m):
        return "FUN_NAME"
    if isParamMatcher(m):
        return "PARAM_KEY_OR_VAL"
    if isStaticSequenceMatcher(m):
        return "PARAM_PREFIX" if b'parameters' in m.target else "CLOSE"
    return "UNKNOWN"


def debug_automaton_state(matcher: Any) -> None:
    """Display current automaton state"""

    if not IS_DEBUG:
        return
    if hasattr(matcher, "_current_matcher"):
        current = matcher._current_matcher
    current_label = _matcher_stage_label(current) if current else "FINISHED"
    current_sub = None
    if isParamMatcher(matcher):
        current_sub = current.sub_state.name

    graph = Text()
    for stage in PIPELINE_STAGES:
        style = "bold black on yellow" if stage == current_label else "dim_white"
        graph.append(f" {stage} ", style=style)
        graph.append(" -> ", style="dim")
    graph.append(" FINISHED ", style="bold black on green" if matcher.is_finished else "dim white")
    rprint(graph)
    if current_sub:
        rprint(f" [dim]sub state:[/dim] [bold magenta]{current_sub}[/bold magenta]")


def debug_stack(matcher: Any) -> None:
    """Print matcher pipeline"""

    if not IS_DEBUG:
        return
    if hasattr(matcher, "pipeline"):
        pipeline: list[TokenMatcher] = matcher.pipeline
    rprint("Expected constraints")
    for m in pipeline:
        rprint("[white]---[/white]")
        if hasattr(m, "acceptable_targets"):
            for t in m.acceptable_targets:
                print(f"{t}")
        elif hasattr(m, "_allowed_keys"):
            for p in m._allowed_keys():
                print(f"{p}")
        elif hasattr(m, "target"):
            print(f"{m.target}")


def debug_prompt(generated: str) -> None:
    """Print constructed prompt"""
    if not IS_DEBUG:
        return
    content_start_idx = generated.index('"prompt":')
    if content_start_idx != -1:
        rprint(f"Generated: [bold green]{escape(generated[content_start_idx:])}[/bold green]")


def debug_decoded_candidates(context: str, authorized_tokens: list[int], logits: np.ndarray,
                             filtered_logits: np.ndarray, model: Small_LLM_Model) -> None:
    """Print first candidate tokens"""

    if not IS_DEBUG:
        return
    logits_arr = np.array(logits).flatten()
    filtered_arr = np.array(filtered_logits).flatten()
    top_global_ids = np.argsort(logits_arr)[::-1][:5]
    vmax = logits_arr[top_global_ids[0]] if len(top_global_ids) > 0 else 0

    table = Table(
        title=f"Top tokens at step {context}",
        box=None,
        show_header=True,
        header_style="bold cyan",
        min_width=50
    )
    table.add_column("Token", justify="left", min_width=15)
    table.add_column("Logit", justify="right", min_width=10)
    table.add_column("Filtered Logit", justify="right", min_width=10)
    table.add_column("ID", justify="right", min_width=8)

    for t_id in top_global_ids:

        t_id = int(t_id)
        t_str = model.decode([t_id])
        t_str_rep = repr(t_str)[1:-1]
        logit_val = float(logits_arr[t_id])
        filtered_val = float(filtered_arr[t_id])

        if filtered_val < -1000 and np.isclose(logit_val, vmax):
            repr_display = f"[bold white on red]{t_str_rep}[/bold white on red]"
            logit_display = f"{logit_val:.2f}"
            filtered_logit_display = f"[bold white on red]{filtered_val:.2f}[/bold white on red]"
        else:
            repr_display = t_str_rep
            logit_display = f"{logit_val:.2f}"
            filtered_logit_display = f"{filtered_val:.2f}"

            if np.isclose(logit_val, vmax):
                repr_display = f"[bold green]{t_str_rep}[/bold green]"

        table.add_row(repr_display, logit_display, filtered_logit_display, str(t_id))

    rprint(table)


def debug_title(name: str) -> None:
    """Print with special title format"""
    if not IS_DEBUG:
        return
    rprint(f"\n[bold yellow on black] === { name.upper() } === [/bold yellow on black]\n")
