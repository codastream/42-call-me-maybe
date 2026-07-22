from typing import Any, List
from dataclasses import dataclass

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.console import Console
import numpy.typing as npt
import numpy as np


from src.matcher.AutomatonDef import AState
from src.matcher.TokenMatcher import TokenMatcher
from src.matcher.AutomatonController import AutomatonController


@dataclass
class DecodingStepState:
    stat_loops: int
    stat_top1_rejected_count: int
    stat_top1_rejected_pct: float
    stat_top_tokens_data: Any
    stat_avg_rank: float
    logits: npt.NDArray[np.int64]
    filtered_logits: npt.NDArray[np.int64]
    authorized_token_ids: list[int]
    next_token_id: int
    readable_chunk: str
    old_generated: str
    current_stage: str
    controller: AutomatonController
    tokenid_to_bytes: dict[int, bytes]
    selected_ranks: list[int]


class DebugDashboard:

    def __init__(self, pipeline_stages: List[str]):
        """Initializer"""
        self.console = Console()
        self.stages = pipeline_stages
        self.layout = Layout()
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Declare layout"""
        h = self.console.size.height
        top_size = 14
        bottom_size = h - top_size
        token_size = 9
        stats_size = 6
        logs_size = 10
        json_size = bottom_size - token_size - stats_size - logs_size

        self.layout.split_column(
            Layout(name="top", size=top_size),
            Layout(name="bottom", ratio=bottom_size)
        )

        self.layout["top"].split_column(
            Layout(name="help", size=5),
            Layout(name="automaton_row", size=9),
        )

        self.layout["automaton_row"].split_row(
            Layout(name="automaton", ratio=5),
            Layout(name="stack", ratio=2)
        )

        self.layout["bottom"].split_row(
            Layout(name="right_metrics", ratio=1)
        )

        self.layout["right_metrics"].split_column(
            Layout(name="token_table", size=token_size),
            Layout(name="summary_stats", size=stats_size),
            Layout(name="json_preview", size=json_size),
            Layout(name="logs", size=logs_size)
        )

        self.layout["summary_stats"].split_row(
            Layout(name="stat_loops", ratio=1),
            Layout(name="stat_rejected_count", ratio=1),
            Layout(name="stat_rejected_pct", ratio=1),
            Layout(name="stat_avg_rank", ratio=1),
        )

        self.layout["summary_stats"].split

    def _build_automaton_graph(self, current_stage: str) -> None:
        """Display automaton states"""
        graph = Text()
        for stage in self.stages:
            style = "bold black on yellow" if stage == current_stage\
                else "bold black on green" if stage == current_stage == AState.FINISH._name_\
                else "dim white"
            graph.append(f" {stage} ", style=style)
            if stage == AState.PARAM_KEY._name_:
                connector = " <-> "
            if stage != AState.PARAM_KEY._name_:
                connector = " > "
            if stage != AState.FINISH._name_:
                graph.append(connector, style="dim")
        centered = Align(graph, align="center", vertical="middle")
        self.layout["automaton"].update(Panel(centered, title="Automaton Flow Graph", border_style="white"))

    def _build_matcher_status(self, active_pipeline: List[TokenMatcher], max_visible: int = 6) -> None:
        """Display current matcher status"""
        stack_text = Text()
        visible = active_pipeline[-1:]
        for _, m in enumerate(active_pipeline):
            stack_text.append(f"{m.display_name()}\n", style="bold yellow")
            stack_text.append(f"{m.display_state()}\n", style="white")

        hidden = len(active_pipeline) - len(visible)
        if hidden > 0:
            stack_text.append(f"\n+{hidden} other matchers \n", style="bold white")

        self.layout["stack"].update(Panel(stack_text, title="Active Matcher", border_style="white"))

    def _get_rank(self, item: dict[str, Any]) -> int:
        """Get rank value"""
        return int(item['rank'])

    def _build_logit_table(self, top_tokens: List[dict[str, Any]]) -> None:
        """Display first logits and their masks"""
        table = Table(box=None, show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Token", justify="left")
        table.add_column("Logit", justify="right")
        table.add_column("Filtered Logit", justify="right")
        table.add_column("Rank", justify="right")

        sorted_items = sorted(top_tokens, key=self._get_rank)
        first_token_rejected = False
        if sorted_items:
            first_token_rejected = sorted_items[0]['filtered'] <= -1000

        for item in sorted_items:
            token_str = repr(item['token'])[1:-1]
            if first_token_rejected and item['rank'] == 1:
                style = "bold white on red"
            elif item['rank'] == 1 and item['filtered'] > -1000:
                style = "bold white on green"
            else:
                style = "bold green" if item['filtered'] > -1000 else "red"

            token_display = Text(token_str, style=style)

            filtered_val = f"{item['filtered']:.2f}" if item['filtered'] > -1000 else "-inf"
            table.add_row(
                token_display,
                f"{item['logit']:.2f}",
                filtered_val,
                str(item['rank'])
            )
        self.layout["token_table"].update(Panel(table, title="Top token logits", border_style="white"))

    def _build_stats(self, loops: int, rejected_pct: float,
                     top1_rejected: int,
                     avg_rank: float) -> None:
        """Display basic figures : loops, average selected rank and rejected first token"""

        p_loops = Panel(
            Align(Text(str(loops), style="bold yellow"), align="center", vertical="middle"),
            title="Iterations", border_style="white"
        )

        p_reject = Panel(
            Align(Text(str(top1_rejected), style="red"), align="center", vertical="middle"),
            title="nb top1 rejected", border_style="white"
        )
        p_reject_pct = Panel(
            Align(Text(f"{rejected_pct * 100:.1f}%", style="red"), align="center", vertical="middle"),
            title="% top1 rejected", border_style="white"
        )
        p_rank = Panel(
            Align(Text(f"{avg_rank:.2f}", style="bold yellow"), align="center", vertical="middle"),
            title="Avg selected rank", border_style="white"
        )

        self.layout["stat_loops"].update(p_loops)
        self.layout["stat_rejected_count"].update(p_reject)
        self.layout["stat_rejected_pct"].update(p_reject_pct)
        self.layout["stat_avg_rank"].update(p_rank)

    def _build_help(self, step_hint: str = "") -> None:
        """Display usage message"""
        if not step_hint:
            return
        help = Text()
        help.append(f"\n{step_hint}", style="bold cyan")
        centered = Align(help, align="center", vertical="top")
        self.layout["help"].update(Panel(centered, title="Help", border_style="white"))

    def _build_logs(self, logs: list[str]) -> None:
        """Display logs"""
        if not logs:
            return
        log_content = Text("\n".join(logs), style="grey")
        self.layout["logs"].update(Panel(log_content, title="Logs", border_style="white"))

    def _build_generated(self, generated_text: str, added_text: str) -> None:
        """Display generated JSON"""
        generated = Text(generated_text, style="green")
        generated.append(f"{added_text}", style="white on green")

        self.layout["json_preview"].update(
            Panel(generated,
                  title="Generated Output String",
                  border_style="white")
        )

    def update(self, current_stage: str,
               active_pipeline: List[Any],
               top_tokens: List[dict[str, Any]],
               loops: int,
               rejected_pct: float,
               top1_rejected: int,
               avg_rank: float,
               generated_text: str,
               generated_added_text: str,
               step_hint: str,
               logs: list[str]
               ) -> Layout:
        """Update dashboard"""

        self._build_automaton_graph(current_stage)
        self._build_matcher_status(active_pipeline)
        self._build_logit_table(top_tokens)
        self._build_stats(loops, rejected_pct, top1_rejected, avg_rank)
        self._build_generated(generated_text, generated_added_text)
        self._build_help(step_hint)
        self._build_logs(logs)

        return self.layout
