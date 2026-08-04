from typing import Any, List

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.console import Console


from src.matcher.AutomatonDef import AState
from src.matcher.TokenMatcher import TokenMatcher
from src.models.DecodingStepState import DecodingStepState


class DebugDashboard:
    """Dashboard for decoding process visualization
    """

    def __init__(self, pipeline_stages: List[str]):
        """Initializer"""
        self.console = Console()
        self.stages = pipeline_stages
        self.layout = Layout()
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Segment console screen"""

        h = self.console.size.height
        top_size = 5
        middle_size = 32
        bottom_size = h - top_size - middle_size

        self.layout.split_column(
            Layout(name="top", size=top_size),
            Layout(name="middle", size=middle_size),
            Layout(name="bottom", size=bottom_size)
        )

        self.layout["top"].split_column(
            Layout(name="help", size=5),
        )

        self.layout["middle"].split_row(
            Layout(name="middle-left", ratio=5),
            Layout(name="middle-right", ratio=2)
        )

        self.layout["middle-left"].split_column(
            Layout(name="automaton", size=5),
            Layout(name="token_table", size=27),
        )

        self.layout["middle-right"].split_column(
            Layout(name="stack", ratio=1),
            Layout(name="stat_loops", size=3),
            Layout(name="stat_rejected_count", size=3),
            Layout(name="stat_rejected_pct", size=3),
            Layout(name="stat_avg_rank", size=3),
            Layout(name="stat_avg_logit", size=3)
        )

        self.layout["bottom"].split_column(
            Layout(name="json_preview", ratio=1),
            Layout(name="logs", ratio=2)
        )

    def _build_automaton_graph(self, current_stage: str) -> None:
        """Display automaton states

        Args:
            current_stage (str): current stage
        """

# FUN_NAME_VAL ──˃ PARAMS_OBJ_KEY ─˃ PARAM_KEY ─˃ PARAM_VAL ˂─˃ CLOSE ───˃  FINISH
#             ╰──˃ EMPTY_PARAMS_AND_CLOSE ─────────────────────────────╯

        graph = Text("\n")
        for stage in [AState.FUN_NAME_VAL,
                      AState.PARAMS_OBJ_KEY,
                      AState.PARAM_KEY,
                      AState.PARAM_VAL,
                      AState.CLOSE,
                      AState.FINISH]:
            style = "bold black on yellow" if stage.name == current_stage\
                else "dim white"
            graph.append(f" {stage.name} ", style=style)
            if stage == AState.PARAM_KEY:
                connector = " <─> "
            else:
                connector = " ─> "
            if stage != AState.FINISH:
                graph.append(connector, style="dim")

        graph.append("\n            ╰───> ", style="dim")
        if current_stage == AState.EMPTY_PARAMS_AND_CLOSE.name:
            style = "bold black on yellow"
        else:
            style = "dim white"
        graph.append(f" {AState.EMPTY_PARAMS_AND_CLOSE.name} ", style=style)
        graph.append("───────────────────────────────────╯", style="dim")

        centered = Align(graph, align="center", vertical="middle")
        self.layout["automaton"].update(Panel(centered, title="Automaton Flow Graph", border_style="white"))

    def _build_matcher_status(self, active_pipeline: List[TokenMatcher], max_visible: int = 6) -> None:
        """Display current matcher status

        Args:
            active_pipeline (List[TokenMatcher]): list of matchers on stack
            max_visible (int, optional): (deprecated) maximum number of visible elements
        """
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
        """Build a table with logits representation, weight, filtered weight and rank

        Args:
            top_tokens (List[dict[str, Any]]): first tokens
        """
        table = Table(box=None, show_header=True, header_style="bold cyan", expand=False)
        w = self.console.width
        left_w = (w * 5) // 7
        w_token = int(left_w * 0.40)
        w_logit = int(left_w * 0.20)
        w_filtered = int(left_w * 0.20)
        w_rank = left_w - (w_token + w_logit + w_filtered)
        table.add_column("Token", justify="left", width=w_token, no_wrap=True)
        table.add_column("Logit", justify="right", width=w_logit, no_wrap=True)
        table.add_column("Filtered Logit", justify="right", width=w_filtered, no_wrap=True)
        table.add_column("Rank", justify="right", width=w_rank, no_wrap=True)

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
                     avg_rank: float, avg_logit: float) -> None:
        """Display decoding stats

        Args:
            loops (int): number of iterations
            rejected_pct (float): % of rejected top-1 tokens
            top1_rejected (int): nb of rejected top-1 tokens
            avg_rank (float): average rank of selected token
            avg_logit (float): average logit of selected token
        """
        reject_style = "bold red" if top1_rejected > 0 else "bold green"
        reject_pc_style = "bold red" if rejected_pct > 10 else "bold green"
        rank_style = "bold green" if avg_rank <= 2 else ("bold yellow" if avg_rank <= 5 else "bold red")
        logit_style = "bold green" if avg_logit > 20 else "bold red"
        p_loops = Panel(
            Align(Text(str(loops), style="bold yellow"), align="center", vertical="middle"),
            title="Iterations", border_style="white"
        )

        p_reject = Panel(
            Align(Text(str(top1_rejected), style=reject_style), align="center", vertical="middle"),
            title="nb top1 rejected", border_style="white"
        )
        p_reject_pct = Panel(
            Align(Text(f"{rejected_pct * 100:.1f}%", style=reject_pc_style), align="center", vertical="middle"),
            title="% top1 rejected", border_style="white"
        )
        p_rank = Panel(
            Align(Text(f"{avg_rank:.2f}", style=rank_style), align="center", vertical="middle"),
            title="Avg selected rank", border_style="white"
        )
        p_logit = Panel(
            Align(Text(f"{avg_logit:.2f}", style=logit_style), align="center", vertical="middle"),
            title="Avg selected logit", border_style="white"
        )

        self.layout["stat_loops"].update(p_loops)
        self.layout["stat_rejected_count"].update(p_reject)
        self.layout["stat_rejected_pct"].update(p_reject_pct)
        self.layout["stat_avg_rank"].update(p_rank)
        self.layout["stat_avg_logit"].update(p_logit)

    def _build_help(self, step_hint: str = "") -> None:
        """Display usage message

        Args:
            step_hint (str, optional): usage hints. Defaults to "".
        """
        if not step_hint:
            return
        help = Text()
        help.append(f"\n{step_hint}", style="bold cyan")
        centered = Align(help, align="center", vertical="top")
        self.layout["help"].update(Panel(centered, title="Help", border_style="white"))

    def _build_logs(self, logs: list[str]) -> None:
        """Display logs

        Args:
            logs (list[str]): logs
        """
        if not logs:
            return
        log_content = Text("\n".join(logs), style="grey")
        self.layout["logs"].update(Panel(log_content, title="Logs", border_style="white"))

    def _build_generated(self, generated_text: str, added_text: str) -> None:
        """Display generated JSON

        Args:
            generated_text (str): part of the output already generated
            added_text (str): new text chunk
        """
        generated = Text(generated_text, style="green")
        generated.append(f"{added_text}", style="white on green")

        self.layout["json_preview"].update(
            Panel(generated,
                  title="Generated Output String",
                  border_style="white")
        )

    def update(self,
               state: DecodingStepState,
               step_hint: str,
               logs: list[str]
               ) -> Layout:
        """Update dashboard according to step state

        Args:
            state (DecodingStepState): step state
            step_hint (str): dashboard usage note
            logs (list[str]): logs

        Returns:
            Layout: updated layout
        """
        metrics = state.metrics
        self._build_automaton_graph(state.current_stage)
        self._build_matcher_status(state.active_pipeline)
        self._build_logit_table(state.top_tokens_data)
        self._build_stats(metrics.loops, metrics.top1_rejected_pct, metrics.top1_rejected_count, metrics.avg_rank,
                          metrics.avg_logit)
        self._build_generated(state.generated_text, state.readable_chunk)
        self._build_help(step_hint)
        self._build_logs(logs)

        return self.layout
