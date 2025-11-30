"""Single-study execution strategy for Optuna optimization.

This module implements the single-study optimization mode where configs are
generated, compiled, and benchmarked using stream processing for maximum
hardware utilization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ... import exc

if TYPE_CHECKING:
    import optuna

    from ...runtime.config import Config
    from ..base_search import BaseSearch
    from .config import OptunaSearchParams
    from .config_mapper import OptunaConfigMapper
    from .trial_tracker import TrialTracker


class SingleStudyExecutor:
    """Executes single-study Optuna optimization.

    This class encapsulates all logic for running a single optimization study,
    including stream-based execution, progress tracking, and trial management.
    """

    def __init__(
        self,
        search: BaseSearch,
        params: OptunaSearchParams,
        study: optuna.Study,
        config_mapper: OptunaConfigMapper,
        trial_tracker: TrialTracker,
    ) -> None:
        """Initialize the single-study executor.

        Args:
            search: BaseSearch instance for compilation/benchmarking.
            params: Optimization parameters.
            study: Optuna study instance.
            config_mapper: Config mapper for trial suggestions.
            trial_tracker: Trial tracker for state management.
        """
        self.search = search
        self.params = params
        self.study = study
        self.config_mapper = config_mapper
        self.trial_tracker = trial_tracker

    def run(self) -> Config:
        """Run single-study optimization.

        Returns:
            Best configuration found.

        Raises:
            InvalidAPIUsage: If pipelining is enabled but device is not a GPU.
            AutotuneError: If no trials completed successfully.
        """
        import time

        # Track start time for timeout
        start_time = time.time()

        # Pipelining only makes sense on GPU
        if (
            self.search.settings.autotune_precompile
            and self.search.kernel.env.device.type != "cuda"
        ):
            raise exc.InvalidAPIUsage(
                "Pipelined compilation (autotune_precompile) requires CUDA device. "
                f"Current device: {self.search.kernel.env.device.type}. "
                "Disable autotune_precompile when benchmarking on CPU."
            )

        # Run sequential trial to establish search space (needed for multivariate TPE)
        sequential_trials_completed = 0
        if self.search.settings.autotune_precompile:
            sequential_trials_completed = self._run_sequential_trial_if_needed(start_time)

        # Set up progress tracking if enabled
        progress_ctx = None
        progress_tracker = None
        rich_handler = None
        original_handlers = []

        if self.params.show_progress_bar:
            from .progress import create_single_study_progress
            from .progress import setup_rich_logging

            progress_ctx, progress_tracker = create_single_study_progress(
                self.params.n_trials,
                self.trial_tracker,
                sequential_trials_completed,
            )

            # Set up rich logging
            rich_handler, original_handlers = setup_rich_logging(
                progress_ctx.live.console,  # type: ignore
                self.search.log,
            )

        # Create inner study runner to execute remaining trials
        from .study_runner import InnerStudyRunner

        # Calculate remaining trials after sequential trial
        remaining_trials = self.params.n_trials - sequential_trials_completed

        runner = InnerStudyRunner(
            search=self.search,
            study=self.study,
            config_mapper=self.config_mapper,
            trial_tracker=self.trial_tracker,
            n_trials=remaining_trials,
            max_configs_ahead=self.params.max_configs_ahead,
            catch_exceptions=self.params.catch_exceptions,
            meta_trial=None,  # No meta-trial for single-study
            timeout=self.params.timeout,
            start_time=start_time,
        )

        try:
            # Run trials with progress tracking
            from ..pipeline import StreamExecutor

            if self.search.settings.autotune_precompile:
                # Use stream execution
                callbacks = runner._create_stream_callbacks()
                executor = StreamExecutor(
                    self.search,
                    callbacks,
                    max_configs_ahead=self.params.max_configs_ahead,
                    progress_tracker=progress_tracker,
                )
                executor.run()
            else:
                # Sequential execution
                runner._run_sequential()
        finally:
            if progress_ctx is not None:
                progress_ctx.__exit__(None, None, None)
            if rich_handler is not None:
                from .progress import teardown_rich_logging

                teardown_rich_logging(self.search.log, rich_handler, original_handlers)

        # Log timeout if applicable
        if runner.stopped_by_timeout:
            import time

            elapsed = time.time() - start_time
            self.search.log(
                f"\nOptimization stopped: timeout reached ({elapsed:.1f}s / {self.params.timeout:.1f}s)"
            )

        self._log_final_statistics()

        return self._get_best_config()

    def _run_sequential_trial_if_needed(self, start_time: float) -> int:
        """Run one sequential trial to establish search space if needed.

        For multivariate TPE to work in parallel, the search space must be
        established first. We need at least 1 completed trial.

        Args:
            start_time: Start time of optimization (for timeout tracking).

        Returns:
            Number of sequential trials completed (0 or 1).
        """
        import optuna

        completed_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(completed_trials) > 0 or self.params.n_trials == 0:
            return 0

        self.search.log("Running 1 sequential trial to establish search space...")

        # Use InnerStudyRunner to run single trial
        from .study_runner import InnerStudyRunner

        runner = InnerStudyRunner(
            search=self.search,
            study=self.study,
            config_mapper=self.config_mapper,
            trial_tracker=self.trial_tracker,
            n_trials=1,
            max_configs_ahead=1,  # Sequential
            catch_exceptions=self.params.catch_exceptions,
            meta_trial=None,
            timeout=self.params.timeout,
            start_time=start_time,
        )

        return runner._run_sequential()

    def _log_final_statistics(self) -> None:
        """Log final optimization statistics."""
        import optuna

        trials = self.study.trials
        completed = sum(
            1 for t in trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        failed = sum(1 for t in trials if t.state == optuna.trial.TrialState.FAIL)
        pruned = sum(1 for t in trials if t.state == optuna.trial.TrialState.PRUNED)

        self.search.log("\nOptimization complete!")
        if completed > 0:
            best_trial = self.study.best_trial
            unit = self.trial_tracker.get_performance_unit()
            self.search.log(f"  Best trial: {best_trial.number}")
            self.search.log(f"  Best value: {best_trial.value:.3f} {unit}")
        else:
            self.search.log("  No trials completed successfully")
        self.search.log(f"  Total trials: {len(trials)}")
        self.search.log(f"  Completed trials: {completed}")
        self.search.log(f"  Pruned trials: {pruned}")
        self.search.log(f"  Failed trials: {failed}")

    def _get_best_config(self) -> Config:
        """Get the best configuration from optimization.

        Returns:
            Best configuration.

        Raises:
            AutotuneError: If no trials completed successfully.
        """
        import optuna

        completed_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(completed_trials) == 0:
            raise exc.AutotuneError(
                "Optimization failed: no trials completed successfully. "
                "All trials either failed or timed out."
            )

        # Return best config if we have one
        if self.trial_tracker.best_config is not None:
            return self.trial_tracker.best_config

        # All trials had inf performance - return config from best trial
        best_trial = self.study.best_trial
        self.search.log(
            "Warning: All trials had infinite performance "
            "(likely all failed accuracy checks). Returning first config."
        )

        # Try to get config from cache first
        cached_config = self.trial_tracker.get_trial_config(best_trial.number)
        if cached_config is not None:
            return cached_config

        # Reconstruct from trial params
        return self.config_mapper.suggest_config(best_trial)


__all__ = ["SingleStudyExecutor"]
