from typing import Any, List
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console
from src.matcher.AutomatonDef import AState

class DebugDashboard:

    def __init__(self, pipeline_stages: List[str]):
        self.console = Console()
        self.stages = pipeline_stages
        self.layout = Layout()
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Declare layout"""
        h             = self.console.size.height
        top_size      = 9
        bottom_size   = h - top_size
        token_size    = 9
        stats_size    = 4
        logs_size     = 10
        json_size     = bottom_size - token_size - stats_size - logs_size

        self.layout.split_column(
            Layout(name="top", size=top_size),
            Layout(name="bottom", ratio=bottom_size)
        )

        self.layout["top"].split_column(
            Layout(name="help", size=4),
            Layout(name="automaton_row", size=5),
        )

        self.layout["automaton_row"].split_row(
            Layout(name="automaton", ratio=4),
            Layout(name="stack", ratio=1)
        )

        self.layout["bottom"].split_row(
            Layout(name="right_metrics", ratio=1)
        )

        self.layout["right_metrics"].split_column(
            Layout(name="token_table", size=token_size),
            Layout(name="summary_stats",size=stats_size),
            Layout(name="json_preview", size=json_size),
            Layout(name="logs", size=logs_size)
        )

    def _build_automaton_graph(self, current_stage: str) -> None:
        graph = Text()
        for stage in self.stages:
            style = "bold black on yellow" if stage == current_stage else "bold black on green" if stage == current_stage == AState.FINISH._name else "dim white"
            graph.append(f" {stage} ", style=style)
            if stage == AState.PARAM_KEY._name_:
                connector = " <-> "
            if stage != AState.PARAM_KEY._name_:
                connector = " > "
            if stage != AState.FINISH._name_:
              graph.append(connector, style="dim")
        self.layout["automaton"].update(Panel(graph, title="Automaton Flow Graph", border_style="white"))

    def _build_matcher_status(self, active_pipeline: List[Any], max_visible: int = 6) -> None:
        stack_text = Text()
        visible = active_pipeline[:max_visible]
        for idx, m in enumerate(active_pipeline):
            color = "orange" if idx == 0 else "dim white"
            stack_text.append(f"{m.display_name()}\n", style=f"bold {color}")
            stack_text.append(f"{m.display_state()}\n", style=f"{color}")

        hidden = len(active_pipeline) - len(visible)
        if hidden > 0:
            stack_text.append(f"\n… +{hidden} other matchers \n", style="bold red")
            
        self.layout["stack"].update(Panel(stack_text, title="Active Matcher Stack", border_style="white"))

    def _build_logit_table(self, top_tokens: List[dict]):
        table = Table(box=None, show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Token", justify="left")
        table.add_column("Logit", justify="right")
        table.add_column("Filtered Logit", justify="right")
        table.add_column("Rank", justify="right")

        sorted_items = sorted(top_tokens, key=lambda x: x['rank'])
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
        stats_text = Text()
        stats_text.append(f"Loop: #{loops} | ", style="white")
        stats_text.append(f"# top-1 rejected: ", style="white")
        stats_text.append(f"{top1_rejected}", style="bold yellow")
        stats_text.append(f" | % rejected: ", style="white")
        stats_text.append(f"{rejected_pct * 100:.1f}%", style="bold yellow")
        stats_text.append(f" | avg selected rank: ", style="white")
        stats_text.append(f"{avg_rank:.2f}", style="bold yellow")
        self.layout["summary_stats"].update(Panel(stats_text, border_style="white"))

    def _build_help(self, step_hint: str = "") -> None:
        if not step_hint:
            return
        help = Text()
        help.append(f"\n{step_hint}", style="bold cyan")
        self.layout["help"].update(Panel(help, title="Help", border_style="white"))

    def _build_logs(self, logs: list[str]) -> None:
        if not logs:
            return
        log_content = Text("\n".join(logs), style="grey")
        self.layout["logs"].update(Panel(log_content, title="Logs", border_style="white"))

    def _build_generated(self, generated_text: str, added_text: str) -> None:
        generated = Text(generated_text, style="green")
        generated.append(f"{added_text}", style="white on green")
        
        self.layout["json_preview"].update(
            Panel(generated,
                  title="Generated Output String",
                  border_style="white")
        )

    def update(self, current_stage: str,
               current_stage_label:str,
               active_pipeline: List[Any],
               top_tokens: List[dict],
               loops: int,
               rejected_pct: float,
               top1_rejected: int,
               avg_rank: float,
               generated_text: str,
               generated_added_text: str,
               step_hint: str,
               logs: list[str]
               ) -> Layout:

        self._build_automaton_graph(current_stage)
        self._build_matcher_status(active_pipeline)
        self._build_logit_table(top_tokens)
        self._build_stats(loops, rejected_pct, top1_rejected, avg_rank)
        self._build_generated(generated_text, generated_added_text)
        self._build_help(step_hint)
        self._build_logs(logs)

        return self.layout
