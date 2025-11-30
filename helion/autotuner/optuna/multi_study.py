"""Multi-study execution strategy for Optuna optimization.

This module implements the multi-study optimization mode where multiple
independent studies run in parallel with different seeds, coordinated by
a meta-study with adaptive resource allocation via pruning.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ... import exc

if TYPE_CHECKING:
    import optuna

    from ...runtime.config import Config
    from ..base_search import BaseSearch
    from .config import OptunaSearchParams
    from .config_mapper import OptunaConfigMapper
    from .trial_tracker import TrialTracker


class MultiStudyExecutor:
    """Executes multi-study Optuna optimization with meta-pruning.

    This class manages parallel execution of multiple optimization studies,
    each with a different seed, coordinated by a meta-study that decides
    which studies to continue based on their intermediate results.
    """

    def __init__(
        self,
        search: BaseSearch,
        params: OptunaSearchParams,
        meta_study: optuna.Study,
        config_mapper: OptunaConfigMapper,
        trial_tracker: TrialTracker,
    ) -> None:
        """Initialize the multi-study executor.

        Args:
            search: BaseSearch instance for compilation/benchmarking.
            params: Optimization parameters.
            meta_study: Meta-study coordinating inner studies.
            config_mapper: Config mapper for trial suggestions.
            trial_tracker: Trial tracker for state management.
        """
        self.search = search
        self.params = params
        self.meta_study = meta_study
        self.config_mapper = config_mapper
        self.trial_tracker = trial_tracker

    def run(self) -> Config:
        """Run multi-study optimization.

        Returns:
            Best configuration found across all inner studies.

        Raises:
            AutotuneError: If no trials completed successfully.
        """
        import time

        # Track start time for timeout (shared across all inner studies)
        self.start_time = time.time()
        self.stopped_by_timeout = False

        # Calculate max_configs_ahead per study
        max_configs_ahead_per_study = max(
            1, self.params.max_configs_ahead // self.params.n_studies
        )

        self.search.log(
            f"Starting multi-study optimization with {self.params.n_studies} parallel studies"
        )
        self.search.log(f"  Trials per study: {self.params.n_trials}")
        self.search.log(f"  Max configs ahead per study: {max_configs_ahead_per_study}")
        self.search.log(f"  Inner sampler: {self.params.sampler_name}")
        self.search.log(f"  Meta sampler: {self.params.meta_sampler_name}")
        self.search.log(f"  Meta pruner: {self.params.meta_pruner_name}")
        self.search.log(f"  Deduplication: {'enabled' if self.trial_tracker.enable_deduplication else 'disabled'}")
        if self.params.timeout is not None:
            self.search.log(f"  Timeout: {self.params.timeout:.1f}s")

        # Run all studies
        asyncio.run(self._run_all_studies_async(max_configs_ahead_per_study))

        # Log timeout if applicable
        if self.stopped_by_timeout:
            import time

            elapsed = time.time() - self.start_time
            self.search.log(
                f"\nMulti-study optimization stopped: timeout reached ({elapsed:.1f}s / {self.params.timeout:.1f}s)"
            )

        # Log final statistics
        self._log_final_statistics()

        if self.trial_tracker.best_config is None:
            raise exc.AutotuneError(
                "Multi-study optimization failed: no trials completed successfully across all studies."
            )

        return self.trial_tracker.best_config

    async def _run_all_studies_async(self, max_configs_ahead_per_study: int) -> None:
        """Run all inner studies concurrently.

        Args:
            max_configs_ahead_per_study: Max configs ahead for each study.
        """
        # Create tasks for all studies
        tasks = [
            asyncio.create_task(
                self._run_inner_study_async(max_configs_ahead_per_study)
            )
            for _ in range(self.params.n_studies)
        ]

        # Set up progress tracking if enabled
        if self.params.show_progress_bar:
            await self._run_with_progress(tasks)
        else:
            await self._run_without_progress(tasks)

    async def _run_with_progress(self, tasks: list[asyncio.Task]) -> None:
        """Run studies with progress bar.

        Args:
            tasks: List of study tasks.
        """
        from .progress import create_multi_study_progress
        from .progress import setup_rich_logging
        from .progress import teardown_rich_logging

        progress, tracker = create_multi_study_progress(
            self.params.n_studies,
            self.trial_tracker,
        )

        # Set up rich logging to redirect logs to progress console
        rich_handler, original_handlers = setup_rich_logging(
            progress.live.console,  # type: ignore
            self.search.log,
        )

        try:
            # Process results as they complete
            for coro in asyncio.as_completed(tasks):
                try:
                    await coro
                except Exception as e:
                    self.search.log(f"Study execution failed: {e}")
                finally:
                    tracker.on_study_complete()
        finally:
            progress.__exit__(None, None, None)
            teardown_rich_logging(self.search.log, rich_handler, original_handlers)

    async def _run_without_progress(self, tasks: list[asyncio.Task]) -> None:
        """Run studies without progress bar.

        Args:
            tasks: List of study tasks.
        """
        for coro in asyncio.as_completed(tasks):
            try:
                await coro
            except Exception as e:
                self.search.log(f"Study execution failed: {e}")

    async def _run_inner_study_async(
        self, max_configs_ahead: int
    ) -> tuple[int, float | None]:
        """Run a single inner optimization study.

        Args:
            max_configs_ahead: Max configs ahead for this study.

        Returns:
            Tuple of (meta_trial_number, best_performance).
            best_performance is None if the study failed or was pruned.
        """
        import optuna

        from .factories import create_sampler
        from .factories import create_study
        from .study_runner import InnerStudyRunner

        # Ask meta-study for a trial
        meta_trial = self.meta_study.ask()
        seed = meta_trial.number

        self.search.log(f"Starting study 'inner_study_{seed}' (seed={seed})")

        try:
            # Create inner study with seed
            # Note: Deduplication is handled at generation time via check_and_mark_config()
            # We only need constraints for accuracy checking here
            constraints_func = (
                self.trial_tracker.create_constraints_func(
                    enable_dedup_constraint=False  # Dedup now handled at generation time
                )
                if self.search.settings.autotune_accuracy_check
                else None
            )

            sampler = create_sampler(
                self.params.sampler_name,
                self.params.sampler_kwargs,
                seed=seed,
                constraints_func=constraints_func,
            )

            inner_study = create_study(
                study_name=f"inner_study_{seed}",
                sampler=sampler,
                direction=self.params.direction,
                storage=None,  # Inner studies use in-memory storage
                pruner=None,  # No pruning within inner studies
                load_if_exists=False,
            )

            # Create inner study runner
            runner = InnerStudyRunner(
                search=self.search,
                study=inner_study,
                config_mapper=self.config_mapper,
                trial_tracker=self.trial_tracker,
                n_trials=self.params.n_trials,
                max_configs_ahead=max_configs_ahead,
                catch_exceptions=self.params.catch_exceptions,
                meta_trial=meta_trial,
                report_interval=self.params.report_interval,
                timeout=self.params.timeout,
                start_time=self.start_time,  # Shared across all inner studies
            )

            # Run the study
            if self.search.settings.autotune_precompile:
                await runner.run_async()
            else:
                runner.run()

            # Track if any inner study stopped due to timeout
            if runner.stopped_by_timeout:
                self.stopped_by_timeout = True

            # Check if any trials completed
            if runner.inner_trials_completed == 0:
                # No trials completed - mark study as failed
                self.search.log(
                    f"Study '{inner_study.study_name}' (seed={seed}) failed: "
                    f"no trials completed successfully"
                )
                if self.params.catch_exceptions:
                    self.meta_study.tell(meta_trial, state=optuna.trial.TrialState.FAIL)
                    return seed, None
                raise exc.AutotuneError(
                    f"Study '{inner_study.study_name}' (seed={seed}) failed: "
                    f"no trials completed successfully"
                )

            # Get final inner_best from the study
            inner_best = runner.inner_best

            # Study completed successfully
            self.meta_study.tell(meta_trial, inner_best)
            return seed, inner_best

        except optuna.TrialPruned:
            # Study was pruned by meta-pruner
            self.meta_study.tell(meta_trial, state=optuna.trial.TrialState.PRUNED)
            return seed, None
        except Exception as e:
            # Study failed
            study_name = f"inner_study_{seed}"
            try:
                # Try to get the actual study name if study was created
                study_name = inner_study.study_name
            except NameError:
                # Study wasn't created yet, use the expected name
                pass
            self.search.log(f"Study '{study_name}' (seed={seed}) failed: {e}")
            if self.params.catch_exceptions:
                self.meta_study.tell(meta_trial, state=optuna.trial.TrialState.FAIL)
                return seed, None
            raise

    def _log_final_statistics(self) -> None:
        """Log final optimization statistics."""
        import optuna

        trials = self.meta_study.trials
        completed = sum(
            1 for t in trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        pruned = sum(1 for t in trials if t.state == optuna.trial.TrialState.PRUNED)
        failed = sum(1 for t in trials if t.state == optuna.trial.TrialState.FAIL)

        self.search.log("\nMulti-study optimization complete!")
        self.search.log(f"  Total studies: {len(trials)}")
        self.search.log(f"  Completed: {completed}")
        self.search.log(f"  Pruned: {pruned}")
        self.search.log(f"  Failed: {failed}")

        if self.trial_tracker.best_config is not None:
            unit = self.trial_tracker.get_performance_unit()
            self.search.log(
                f"  Best performance: {self.trial_tracker.best_perf:.3f}{unit}"
            )

        # Log deduplication statistics
        if self.trial_tracker.enable_deduplication:
            stats = self.trial_tracker.get_deduplication_stats()
            self.search.log(f"  Unique configs evaluated: {stats['unique_configs_evaluated']}")
            self.search.log(f"  Duplicate configs skipped: {stats['duplicate_configs_skipped']}")


__all__ = ["MultiStudyExecutor"]
