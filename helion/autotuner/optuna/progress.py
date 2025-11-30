"""Progress tracking and visualization for Optuna optimization.

This module provides progress bar management and tracking for both
single-study and multi-study optimization modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from rich.console import Console
    from rich.progress import Progress
    from rich.progress import TaskID

    from .trial_tracker import TrialTracker


class SingleStudyProgressTracker:
    """Progress tracker for single-study optimization.

    Manages three progress bars for the pipeline stages:
    - Generation: Configs generated
    - Compilation: Configs compiled
    - Benchmarking: Configs benchmarked
    """

    def __init__(
        self,
        progress: Progress,
        generate_task_id: TaskID,
        compile_task_id: TaskID,
        benchmark_task_id: TaskID,
        trial_tracker: TrialTracker,
    ) -> None:
        """Initialize the progress tracker.

        Args:
            progress: Rich Progress instance.
            generate_task_id: Task ID for generation progress.
            compile_task_id: Task ID for compilation progress.
            benchmark_task_id: Task ID for benchmarking progress.
            trial_tracker: TrialTracker for getting best performance.
        """
        self.progress = progress
        self.gen_id = generate_task_id
        self.comp_id = compile_task_id
        self.bench_id = benchmark_task_id
        self.trial_tracker = trial_tracker

    def on_config_generated(self) -> None:
        """Called when a config is generated."""
        self.progress.update(self.gen_id, advance=1)

    def on_compilation_complete(self) -> None:
        """Called when a config finishes compilation."""
        self.progress.update(self.comp_id, advance=1)

    def on_benchmark_complete(self) -> None:
        """Called when a config finishes benchmarking."""
        self.progress.update(self.bench_id, advance=1)
        # Update best value display
        if self.trial_tracker.best_config is not None:
            unit = self.trial_tracker.get_performance_unit()
            best_str = f"best: {self.trial_tracker.best_perf:.3f}{unit}"
            self.progress.update(self.bench_id, best=best_str)


class MultiStudyProgressTracker:
    """Progress tracker for multi-study optimization.

    Manages a single progress bar showing study completion.
    """

    def __init__(
        self,
        progress: Progress,
        task_id: TaskID,
        trial_tracker: TrialTracker,
    ) -> None:
        """Initialize the progress tracker.

        Args:
            progress: Rich Progress instance.
            task_id: Task ID for study progress.
            trial_tracker: TrialTracker for getting best performance.
        """
        self.progress = progress
        self.task_id = task_id
        self.trial_tracker = trial_tracker

    def on_study_complete(self) -> None:
        """Called when a study completes."""
        self.progress.update(self.task_id, advance=1)
        # Update best value display
        if self.trial_tracker.best_config is not None:
            unit = self.trial_tracker.get_performance_unit()
            best_str = f"best: {self.trial_tracker.best_perf:.3f}{unit}"
            self.progress.update(self.task_id, best=best_str)


def create_single_study_progress(
    n_trials: int,
    trial_tracker: TrialTracker,
    sequential_trials_completed: int = 0,
) -> tuple[Progress, SingleStudyProgressTracker]:
    """Create progress context and tracker for single-study mode.

    Args:
        n_trials: Total number of trials.
        trial_tracker: TrialTracker instance.
        sequential_trials_completed: Number of trials already completed sequentially.

    Returns:
        Tuple of (Progress context, tracker instance).
    """
    from rich.console import Console
    from rich.progress import BarColumn
    from rich.progress import MofNCompleteColumn
    from rich.progress import Progress
    from rich.progress import TextColumn

    console = Console()

    progress_ctx = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="yellow", finished_style="green"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[best]}"),
        console=console,
    )
    progress_ctx.__enter__()

    # Three separate progress bars for pipeline stages
    generate_task_id = progress_ctx.add_task(
        "Generated",
        total=n_trials,
        completed=sequential_trials_completed,
        best="",
    )
    compile_task_id = progress_ctx.add_task(
        "Compiled ",
        total=n_trials,
        completed=sequential_trials_completed,
        best="",
    )
    benchmark_task_id = progress_ctx.add_task(
        "Benchmarked",
        total=n_trials,
        completed=sequential_trials_completed,
        best="",
    )

    tracker = SingleStudyProgressTracker(
        progress_ctx,
        generate_task_id,
        compile_task_id,
        benchmark_task_id,
        trial_tracker,
    )

    return progress_ctx, tracker


def create_multi_study_progress(
    n_studies: int,
    trial_tracker: TrialTracker,
) -> tuple[Progress, MultiStudyProgressTracker]:
    """Create progress context and tracker for multi-study mode.

    Args:
        n_studies: Total number of studies.
        trial_tracker: TrialTracker instance.

    Returns:
        Tuple of (Progress context, tracker instance).
    """
    from rich.console import Console
    from rich.progress import BarColumn
    from rich.progress import MofNCompleteColumn
    from rich.progress import Progress
    from rich.progress import TextColumn

    console = Console()

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="yellow", finished_style="green"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[best]}"),
        console=console,
    )
    progress.__enter__()

    task_id = progress.add_task(
        "Studies",
        total=n_studies,
        best="",
    )

    tracker = MultiStudyProgressTracker(progress, task_id, trial_tracker)

    return progress, tracker


def setup_rich_logging(console: Console, logger: Any) -> tuple[Any, list[Any]]:
    """Setup rich logging handler for prettier output during progress display.

    Args:
        console: Rich Console instance to use for logging.
        logger: Logger instance to configure.

    Returns:
        Tuple of (RichHandler instance, original handlers list).
    """
    from rich.logging import RichHandler

    # Temporarily remove existing handlers to avoid duplicate logs
    original_handlers = list(logger._active_handlers)
    for handler in original_handlers:
        logger.remove_handler(handler)

    rich_handler = RichHandler(console=console, show_time=False, show_path=False)
    rich_handler.setFormatter(
        original_handlers[0].formatter if original_handlers else None
    )
    logger.add_handler(rich_handler)

    return rich_handler, original_handlers


def teardown_rich_logging(
    logger: Any, rich_handler: Any, original_handlers: list[Any]
) -> None:
    """Restore original logging configuration after progress display.

    Args:
        logger: Logger instance to restore.
        rich_handler: RichHandler to remove.
        original_handlers: Original handlers to restore.
    """
    logger.remove_handler(rich_handler)
    for handler in original_handlers:
        logger.add_handler(handler)


__all__ = [
    "MultiStudyProgressTracker",
    "SingleStudyProgressTracker",
    "create_multi_study_progress",
    "create_single_study_progress",
    "setup_rich_logging",
    "teardown_rich_logging",
]
