"""Optuna-based autotuner for Helion kernels.

This module provides an autotuner that uses Optuna, a sophisticated hyperparameter
optimization framework, to search for optimal kernel configurations.

Key features:
- Supports multiple sampling algorithms (TPE, CMA-ES, GP, Random, etc.)
- Pruning for early stopping of unpromising trials
- Persistent studies via database URLs
- Parallel trial execution
- Multi-objective optimization support
- Advanced search space features (categorical, numerical, conditional parameters)

Example usage:
    # Use via environment variable
    HELION_AUTOTUNER=OptunaSearch python your_script.py

    # Or directly
    from helion.autotuner import OptunaSearch
    autotuner = OptunaSearch(bound_kernel, args, n_trials=100)
    best_config = autotuner.autotune()
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from typing import Sequence

from ... import exc
from ..base_search import BaseSearch
from .config import OptunaSearchParams
from .config import apply_env_overrides
from .config_mapper import OptunaConfigMapper
from .factories import create_pruner
from .factories import create_sampler
from .factories import create_storage
from .factories import create_study
from .factories import get_storage_description
from .trial_tracker import TrialTracker

if TYPE_CHECKING:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages import RDBStorage

    from ...runtime.config import Config
    from ...runtime.kernel import BoundKernel

try:
    HAS_OPTUNA = True
except ImportError as e:
    HAS_OPTUNA = False
    _IMPORT_ERROR = e


class OptunaSearch(BaseSearch):
    """Optuna-based autotuner for Helion kernels.

    This autotuner uses Optuna's sophisticated optimization algorithms to search
    for optimal kernel configurations. Optuna provides state-of-the-art sampling
    algorithms like TPE (Tree-structured Parzen Estimator), CMA-ES, and Gaussian
    Processes, along with pruning capabilities for early stopping.

    The autotuner supports two modes:
    1. Single-study mode (n_studies=1): Standard Optuna optimization with one study
    2. Multi-study mode (n_studies>1): Runs multiple studies with different seeds,
       using a meta-pruner for adaptive resource allocation to the most promising studies

    The autotuner automatically maps Helion's configuration space to Optuna's
    search space, handling block sizes, loop orders, and other kernel parameters.

    Args:
        bound_kernel: The kernel to autotune.
        args: Arguments to pass to the kernel.
        params: OptunaSearchParams instance with configuration options.

    Example (single-study):
        >>> # Using JournalStorage (recommended for parallel execution)
        >>> params = OptunaSearchParams(
        ...     n_trials=200,
        ...     sampler_name="tpe",
        ...     storage="optuna_journal.log",  # File path uses JournalStorage
        ...     study_name="my_kernel_optimization",
        ... )
        >>> autotuner = OptunaSearch(bound_kernel, args, params=params)
        >>> best_config = autotuner.autotune()
        >>> autotuner.close()  # Clean up resources

    Example (multi-study with meta-pruning):
        >>> params = OptunaSearchParams(
        ...     n_trials=50,  # 50 trials per study
        ...     n_studies=10,  # 10 studies with different seeds
        ...     sampler_name="tpe",
        ...     meta_sampler_name="random",
        ...     meta_pruner_name="median",  # Prune studies performing below median
        ...     show_progress_bar=True,
        ... )
        >>> autotuner = OptunaSearch(bound_kernel, args, params=params)
        >>> best_config = autotuner.autotune()
        >>> autotuner.close()
    """

    def __init__(
        self,
        bound_kernel: BoundKernel,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
        args: Sequence[object],
        *,
        params: OptunaSearchParams | None = None,
    ) -> None:
        """Initialize OptunaSearch.

        Args:
            bound_kernel: The kernel to autotune.
            args: Arguments to pass to the kernel.
            params: OptunaSearchParams with Optuna-specific settings. If None, uses defaults.

        Raises:
            AutotuneError: If Optuna is not installed.
        """
        if not HAS_OPTUNA:
            raise exc.AutotuneError(
                "OptunaSearch requires optuna. "
                "Install with: pip install helion[optuna] or pip install optuna"
            ) from _IMPORT_ERROR

        super().__init__(bound_kernel, args)  # pyright: ignore[reportUnknownMemberType]

        # Store or create default params
        self.params = params if params is not None else OptunaSearchParams()

        # Apply environment variable overrides
        apply_env_overrides(self.params)

        # Initialize core components
        self.config_mapper = OptunaConfigMapper(self.kernel)
        self.trial_tracker = TrialTracker(
            direction=self.params.direction,
            enable_deduplication=self.params.enable_deduplication or False,
        )

        # Study will be created in _autotune
        self.study: optuna.Study | None = None
        self.meta_study: optuna.Study | None = None
        self.storage: RDBStorage | JournalStorage | None = None

    def _create_single_study(self) -> optuna.Study:
        """Create or load an Optuna study for single-study mode.

        Returns:
            Configured Optuna study instance.
        """
        # Create sampler with constraints
        constraints_func = (
            self.trial_tracker.create_constraints_func()
            if self.settings.autotune_accuracy_check
            else None
        )

        sampler = create_sampler(
            self.params.sampler_name,
            self.params.sampler_kwargs,
            constraints_func=constraints_func,
        )

        # Create storage if using persistence
        self.storage = create_storage(self.params.storage)
        storage_desc = get_storage_description(self.storage)

        # Generate study name
        study_name = (
            self.params.study_name or f"helion_{self.kernel.kernel.fn.__name__}"
        )

        # Log study configuration
        self.log(f"Creating Optuna study: {study_name}")
        self.log(f"  Sampler: {sampler.__class__.__name__}")
        self.log(f"  Storage: {storage_desc}")
        self.log(f"  Trials: {self.params.n_trials}")
        self.log(f"  Max configs ahead: {self.params.max_configs_ahead}")

        # Create study
        study = create_study(
            study_name=study_name,
            sampler=sampler,
            direction=self.params.direction,
            storage=self.storage,
            pruner=None,  # No pruning within single studies
            load_if_exists=self.params.load_if_exists,
        )

        # Initialize from existing trials if resuming
        if self.params.load_if_exists and len(study.trials) > 0:
            self.log(f"Resuming existing study with {len(study.trials)} trials")

            n_completed = self.trial_tracker.initialize_from_study(
                study, self.config_mapper.suggest_config
            )

            if n_completed > 0:
                unit = self.trial_tracker.get_performance_unit()
                self.log(
                    f"  Found {n_completed} completed trials with finite performance"
                )
                self.log(f"  Best existing: {self.trial_tracker.best_perf:.3f}{unit}")

        return study

    def _create_meta_study(self) -> optuna.Study:
        """Create or load a meta-study for multi-study mode.

        Returns:
            Configured Optuna meta-study instance.
        """
        meta_sampler = create_sampler(
            self.params.meta_sampler_name or "random",
            self.params.meta_sampler_kwargs,
        )

        meta_pruner = create_pruner(
            self.params.meta_pruner_name,
            self.params.meta_pruner_kwargs,
        )

        # Create storage if using persistence
        self.storage = create_storage(self.params.storage)

        # Create meta-study
        meta_study_name = (
            f"{self.params.study_name or 'helion'}_meta"
            if self.params.storage
            else None
        )

        return create_study(
            study_name=meta_study_name or "meta_study",
            sampler=meta_sampler,
            direction=self.params.direction,
            storage=self.storage,
            pruner=meta_pruner,
            load_if_exists=self.params.load_if_exists if self.params.storage else False,
        )

    def _autotune(self) -> Config:
        """Run Optuna optimization.

        Dispatches to either single-study or multi-study mode based on params.

        Returns:
            Best configuration found.
        """
        if self.params.n_studies > 1:
            return self._autotune_multi_study()
        return self._autotune_single_study()

    def _autotune_single_study(self) -> Config:
        """Run single-study optimization.

        Returns:
            Best configuration found.
        """
        from .single_study import SingleStudyExecutor

        self.study = self._create_single_study()

        executor = SingleStudyExecutor(
            search=self,
            params=self.params,
            study=self.study,
            config_mapper=self.config_mapper,
            trial_tracker=self.trial_tracker,
        )

        return executor.run()

    def _autotune_multi_study(self) -> Config:
        """Run multi-study optimization.

        Returns:
            Best configuration found.
        """
        from .multi_study import MultiStudyExecutor

        self.meta_study = self._create_meta_study()

        executor = MultiStudyExecutor(
            search=self,
            params=self.params,
            meta_study=self.meta_study,
            config_mapper=self.config_mapper,
            trial_tracker=self.trial_tracker,
        )

        return executor.run()

    def get_study(self) -> optuna.Study | None:
        """Get the Optuna study object for further analysis.

        For single-study mode (n_studies=1), returns the main study.
        For multi-study mode (n_studies>1), returns None (use get_meta_study instead).

        This can be used to access trial history, visualize optimization,
        or perform additional analysis using Optuna's visualization tools.

        Returns:
            The Optuna study object (single-study mode), or None.

        Example:
            >>> autotuner = OptunaSearch(
            ...     bound_kernel, args, params=OptunaSearchParams(n_studies=1)
            ... )
            >>> best_config = autotuner.autotune()
            >>> study = autotuner.get_study()
            >>> # Visualize optimization history
            >>> import optuna.visualization as vis
            >>> vis.plot_optimization_history(study).show()
        """
        return self.study

    @property
    def best_config(self) -> Config | None:
        """Get the best configuration found so far.

        Returns:
            Best configuration, or None if no valid configs have been found.
        """
        if hasattr(self, "trial_tracker"):
            return self.trial_tracker.best_config
        return None

    @property
    def best_perf_so_far(self) -> float:
        """Get the best performance found so far.

        Returns:
            Best performance value.
        """
        if hasattr(self, "trial_tracker"):
            return self.trial_tracker.best_perf
        # During initialization, return the temporary value if it exists
        return getattr(self, "_best_perf_temp", float("inf"))

    @best_perf_so_far.setter
    def best_perf_so_far(self, value: float) -> None:
        """Set the best performance (for compatibility with BaseSearch).

        This setter is called by benchmark_function during trial execution.
        We intentionally do NOT update trial_tracker.best_perf here, because:
        1. trial_tracker.best_perf is managed by update_best() which also sets best_config
        2. Directly setting best_perf without best_config causes inconsistent state
        3. _report_success will call update_best() after benchmarking completes

        Args:
            value: New best performance value (ignored for Optuna).
        """
        # Do nothing - trial_tracker.best_perf is managed by update_best()
        pass

    def get_meta_study(self) -> optuna.Study | None:
        """Get the meta-study object for analysis (multi-study mode only).

        This is only available when n_studies > 1. The meta-study contains information
        about which inner studies were run and which were pruned.

        Returns:
            The meta-study object, or None if not in multi-study mode or optimization hasn't run yet.

        Example:
            >>> params = OptunaSearchParams(n_trials=50, n_studies=10)
            >>> autotuner = OptunaSearch(bound_kernel, args, params=params)
            >>> best_config = autotuner.autotune()
            >>> meta_study = autotuner.get_meta_study()
            >>> # Analyze which studies were pruned
            >>> pruned = [
            ...     t
            ...     for t in meta_study.trials
            ...     if t.state == optuna.trial.TrialState.PRUNED
            ... ]
        """
        return self.meta_study

    def close(self) -> None:
        """Close database connections and cleanup resources.

        This method should be called when you're done with the OptunaSearch instance,
        especially if you're using database storage. It ensures that all database
        connections are properly closed.

        Example:
            >>> autotuner = OptunaSearch(bound_kernel, args, params=params)
            >>> try:
            ...     best_config = autotuner.autotune()
            ... finally:
            ...     autotuner.close()
        """
        if self.storage is not None:
            # Check if this is RDBStorage (has engine/scoped_session)
            if hasattr(self.storage, "scoped_session"):
                # Close RDBStorage connections
                with contextlib.suppress(Exception):
                    # Remove scoped sessions
                    self.storage.scoped_session.remove()
                with contextlib.suppress(Exception):
                    # Dispose of the SQLAlchemy engine
                    self.storage.engine.dispose()
            # JournalStorage doesn't need explicit cleanup
            self.storage = None

        # Clear study references to help with cleanup
        self.study = None
        self.meta_study = None

    def __del__(self) -> None:
        """Cleanup database connections when the object is deleted."""
        self.close()


__all__ = ["OptunaSearch", "OptunaSearchParams"]
