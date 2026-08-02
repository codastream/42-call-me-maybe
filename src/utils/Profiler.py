

from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import time
from typing import Any, Callable, Generator, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])


class Profiler:
    """Accumulate execution time by label

    Attributes:
    timings (dict[str, float]): Total accumulated time in seconds, keyed
        by label.
    counts (dict[str, int]): Number of times each label was tracked,
        keyed by label.
    """

    def __init__(self) -> None:
        """Initializes empty timing and count accumulators."""
        self.timings: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        """Clears all accumulated timings and counts.

        Note:
            Should be called at the start of each new prompt's decoding
            loop to avoid mixing measurements across prompts.
        """
        self.timings.clear()
        self.counts.clear()

    @contextmanager
    def track(self, label: str) -> Generator[None, None, None]:
        """Time a block of code and accumulate the elapsed time under a label.

        Args:
            label (str): Identifier under which the elapsed time and call
                count are accumulated.

        Yields:
            None: Control returns to the caller's `with` block; timing is
            recorded on exit (including on exception, via `finally`).
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[label] += time.perf_counter() - start
            self.counts[label] += 1

    def decorate(self, label: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Builds a decorator that times every call to the wrapped function.

        Args:
            label (str | None): Identifier to accumulate time under. If
                None, the wrapped function's qualified name (`__qualname__`)
                is used instead.

        Returns:
            Callable[[F], F]: A decorator that wraps a function with timing
            instrumentation and returns a function of the same signature.
        """
        def deco(fn: Callable[P, R]) -> Callable[P, R]:
            name = label or fn.__qualname__

            @wraps(fn)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                with self.track(name):
                    return fn(*args, **kwargs)
            return wrapper
        return deco

    def report(self) -> str:
        """Builds a human-readable summary table of accumulated timings.

        Note:
            Timings are inclusive of nested tracked calls, so the sum of
            all rows' percentages is not guaranteed to equal 100%.

        Returns:
            str: A formatted, newline-joined table with columns for label,
            total time (ms), call count, average time per call (ms), and
            percentage of total tracked time. Rows are sorted by total time
            descending.
        """
        total = sum(self.timings.values()) or 1e-9
        rows = sorted(self.timings.items(), key=lambda kv: -kv[1])
        label_width = max((len(label) for label in self.timings), default=5)
        label_width = max(label_width, len("label"))
        lines = [f"{'label':<{label_width}} {'ms':>10} {'calls':>8} {'ms/call':>10} {'%':>6}"]
        for label, secs in rows:
            n = self.counts[label]
            lines.append(
                f"{label:42s} {secs * 1000:10.2f} {n:8d} {secs * 1000 / n:10.4f} {secs / total * 100:5.1f}%"
            )
        return "\n".join(lines)


profiler = Profiler()
