"""Inner study runner for Optuna optimization.

This module provides the core trial execution logic that is shared between
single-study and multi-study optimization modes, eliminating duplication.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

    from ...runtime.config import Config
    from ..base_search import BaseSearch
    from .config_mapper import OptunaConfigMapper
    from .trial_tracker import TrialTracker


class InnerStudyRunner:
    """Runs trials for a single Optuna study.

    This class encapsulates the core trial execution logic that is shared
    between single-study mode (running one study) and multi-study mode
    (running N studies in parallel).

    The runner handles:
    - Trial generation with Optuna's ask-and-tell interface
    - Config generation from trials
    - Result reporting to Optuna
    - Optional meta-trial reporting for pruning (multi-study mode)
    - Integration with StreamExecutor for pipelined execution
    """

    def __init__(
        self,
        search: BaseSearch,
        study: optuna.Study,
        config_mapper: OptunaConfigMapper,
        trial_tracker: TrialTracker,
        n_trials: int,
        max_configs_ahead: int,
        catch_exceptions: bool,
        meta_trial: optuna.Trial | None = None,
        report_interval: int = 1,
        timeout: float | None = None,
        start_time: float | None = None,
    ) -> None:
        """Initialize the inner study runner.

        Args:
            search: BaseSearch instance for compilation/benchmarking.
            study: Optuna study to run trials for.
            config_mapper: Config mapper for trial suggestions.
            trial_tracker: Trial tracker for state management.
            n_trials: Number of trials to run.
            max_configs_ahead: Max configs to generate ahead of completion.
            catch_exceptions: Whether to catch and handle exceptions.
            meta_trial: Optional meta-trial for pruning (multi-study mode).
            report_interval: Report to meta-trial every N trials (if meta_trial set).
            timeout: Optional timeout in seconds for the study.
            start_time: Optional start time (time.time()) for timeout tracking.
                If timeout is set but start_time is None, current time is used.
        """
        import time

        self.search = search
        self.study = study
        self.config_mapper = config_mapper
        self.trial_tracker = trial_tracker
        self.n_trials = n_trials
        self.max_configs_ahead = max_configs_ahead
        self.catch_exceptions = catch_exceptions
        self.meta_trial = meta_trial
        self.report_interval = report_interval
        self.timeout = timeout
        self.start_time = start_time if start_time is not None else time.time()

        # Track best performance within this study (for meta-trial reporting)
        self.inner_best = (
            math.inf if trial_tracker.direction == "minimize" else -math.inf
        )
        self.inner_trials_completed = 0
        self.stopped_by_timeout = False

    def run(self) -> int:
        """Run trials using stream or sequential execution (blocking).

        Returns:
            Number of trials completed.
        """
        if self.search.settings.autotune_precompile:
            return self._run_with_stream()
        return self._run_sequential()

    async def run_async(self) -> int:
        """Run trials using stream execution (async).

        Returns:
            Number of trials completed.
        """
        if self.search.settings.autotune_precompile:
            from ..pipeline import StreamExecutor

            callbacks = self._create_stream_callbacks()
            executor = StreamExecutor(
                self.search,
                callbacks,
                max_configs_ahead=self.max_configs_ahead,
                progress_tracker=None,  # Progress tracked at executor level
            )
            return await executor.run_async()
        return self._run_sequential()

    def _check_timeout(self) -> bool:
        """Check if study has exceeded timeout.

        Returns:
            True if timeout has been exceeded, False otherwise.
        """
        import time

        if self.timeout is None:
            return False

        elapsed = time.time() - self.start_time
        return elapsed >= self.timeout

    def _create_stream_callbacks(self):
        """Create callbacks for stream execution."""
        from ..pipeline import StreamCallbacks

        def generate_next(generated: int) -> tuple[optuna.Trial, Config] | None:
            """Generate next trial and config."""
            import optuna

            if generated >= self.n_trials:
                return None

            # Check timeout
            if self._check_timeout():
                self.stopped_by_timeout = True
                return None

            trial = self.study.ask()
            try:
                config = self.config_mapper.suggest_config(trial)

                # CRITICAL: Check for duplicates BEFORE compiling/benchmarking
                # This prevents multiple studies from working on the same config in parallel
                if not self.trial_tracker.check_and_mark_config(config):
                    # Duplicate config - skip it to avoid wasted GPU time
                    # Mark trial as failed so sampler learns to avoid this region
                    self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    # Return None to skip this trial and generate a new one
                    return None

                return trial, config
            except Exception as e:
                self._handle_trial_failure(trial, e)
                return None

        def report_result(trial: optuna.Trial, result: object) -> None:
            """Report single result."""
            # Import here to avoid circular dependency issues at module load time
            import helion.autotuner.base_search

            if not isinstance(result, helion.autotuner.base_search.BenchmarkResult):
                return

            # Store accuracy error
            self.trial_tracker.store_accuracy_error(trial.number, result.accuracy_error)

            # Report result based on status
            if result.status == "ok" and math.isfinite(result.perf):
                self._report_success(trial, result.perf, result.config)
            elif result.status == "ok" and not math.isfinite(result.perf):
                self._handle_trial_failure(
                    trial,
                    RuntimeError(f"Trial produced invalid performance: {result.perf}"),
                )
            elif result.status == "timeout":
                self._handle_trial_failure(
                    trial,
                    RuntimeError("Trial timed out during compilation/benchmarking"),
                )
            else:
                # status is "error" - actual error was logged in pipeline
                self._handle_trial_failure(
                    trial,
                    RuntimeError(
                        "Trial failed during pipeline execution (see logs above)"
                    ),
                )

        def should_continue(completed: int) -> bool:
            """Check if optimization should continue."""
            # Check trial limit
            if completed >= self.n_trials:
                return False

            # Check timeout
            if self._check_timeout():
                self.stopped_by_timeout = True
                return False

            return True

        return StreamCallbacks(
            generate_next=generate_next,
            report_result=report_result,
            should_continue=should_continue,
        )

    def _run_with_stream(self) -> int:
        """Run trials using stream execution (blocking)."""
        from ..pipeline import StreamExecutor

        callbacks = self._create_stream_callbacks()
        executor = StreamExecutor(
            self.search,
            callbacks,
            max_configs_ahead=self.max_configs_ahead,
            progress_tracker=None,
        )
        return executor.run()

    def _run_sequential(self) -> int:
        """Run trials sequentially without precompilation.

        Returns:
            Number of trials completed.
        """

        for _ in range(self.n_trials):
            # Check timeout before each trial
            if self._check_timeout():
                self.stopped_by_timeout = True
                break

            trial = self.study.ask()
            try:
                config = self.config_mapper.suggest_config(trial)

                # CRITICAL: Check for duplicates BEFORE compiling/benchmarking
                if not self.trial_tracker.check_and_mark_config(config):
                    # Duplicate config - skip it
                    import optuna

                    self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    continue
            except Exception as e:
                self._handle_trial_failure(trial, e)
                continue

            try:
                # Compile and benchmark
                fn = self.search.kernel.compile_config(config, allow_print=False)
                perf = self.search.benchmark_function(config, fn)
                accuracy_error = self.search.last_accuracy_error

                if math.isfinite(perf):
                    # Store accuracy error
                    self.trial_tracker.store_accuracy_error(
                        trial.number, accuracy_error
                    )

                    # Report success
                    self._report_success(trial, perf, config)
                else:
                    self._handle_trial_failure(
                        trial,
                        RuntimeError(f"Trial produced invalid performance: {perf}"),
                    )
            except Exception as e:
                self._handle_trial_failure(trial, e)

        return self.inner_trials_completed

    def _report_success(self, trial: optuna.Trial, perf: float, config: Config) -> None:
        """Report a successful trial result.

        Args:
            trial: Optuna trial.
            perf: Performance value.
            config: Configuration that achieved this performance.
        """
        import optuna

        # Store in trial tracker
        self.trial_tracker.store_trial_config(trial.number, config)
        self.trial_tracker.update_best(perf, config)

        # NOTE: Config was already marked as evaluated at generation time
        # to prevent duplicate work across parallel studies

        # Tell study about result
        self.study.tell(trial, perf)
        self.inner_trials_completed += 1

        # Update inner best
        is_better = (
            self.trial_tracker.direction == "minimize" and perf < self.inner_best
        ) or (self.trial_tracker.direction == "maximize" and perf > self.inner_best)

        if is_better:
            self.inner_best = perf

        # Report to meta-trial if we have one (multi-study mode)
        if self.meta_trial is not None:
            if self.inner_trials_completed % self.report_interval == 0:
                self.meta_trial.report(self.inner_best, self.inner_trials_completed)

                # Check if meta-pruner wants to stop this study
                if self.meta_trial.should_prune():
                    unit = self.trial_tracker.get_performance_unit()
                    self.search.log(
                        f"Study '{self.study.study_name}' (meta-trial {self.meta_trial.number}) "
                        f"pruned at trial {self.inner_trials_completed} (best: {self.inner_best:.3f}{unit})"
                    )
                    raise optuna.TrialPruned

    def _handle_trial_failure(self, trial: optuna.Trial, error: Exception) -> None:
        """Handle a failed trial.

        Args:
            trial: The Optuna trial that failed.
            error: The exception that caused the failure.
        """
        import traceback

        import optuna

        if self.catch_exceptions:
            # Log the error with type and message
            error_msg = (
                f"{error.__class__.__name__}: {error}"
                if error.__class__.__name__ != "RuntimeError"
                else str(error)
            )
            self.search.log(
                f"Trial {trial.number} failed in study '{self.study.study_name}': {error_msg}"
            )

            # Log full traceback if verbose mode enabled
            if hasattr(self.search, "settings") and getattr(
                self.search.settings, "verbose", False
            ):
                tb_str = "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                )
                self.search.log(f"Traceback:\n{tb_str}")

            self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            raise error


__all__ = ["InnerStudyRunner"]
