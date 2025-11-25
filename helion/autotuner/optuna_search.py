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

import dataclasses
import math
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from . import exc
from .base_search import BaseSearch
from .config_fragment import BooleanFragment
from .config_fragment import EnumFragment
from .config_fragment import IntegerFragment
from .config_fragment import ListOf
from .config_fragment import PermutationFragment
from .config_fragment import PowerOfTwoFragment
from .config_generation import ConfigGeneration

if TYPE_CHECKING:
    from optuna import pruners
    from optuna.samplers import BaseSampler

    from ..runtime.kernel import BoundKernel
    from .config import Config

try:
    import optuna

    HAS_OPTUNA = True
except ImportError as e:
    HAS_OPTUNA = False
    _IMPORT_ERROR = e


@dataclasses.dataclass
class OptunaSearchParams:
    """Parameters for OptunaSearch.

    Attributes:
        n_trials: Number of trials to run. If None, runs indefinitely until stopped.
        timeout: Stop study after this number of seconds. None means no timeout.
        sampler: Optuna sampler to use. Can be a string name or a BaseSampler instance.
            Supported strings: 'tpe' (default), 'cmaes', 'gp', 'random', 'grid'.
        sampler_kwargs: Additional kwargs passed to sampler constructor (if sampler is a string).
        pruner: Optuna pruner to use. Can be a string name or pruner instance.
            Supported strings: 'median', 'hyperband', 'percentile', 'threshold', None.
        pruner_kwargs: Additional kwargs passed to pruner constructor (if pruner is a string).
        study_name: Name of the Optuna study. Used for persistence and logging.
        storage: Database URL for study persistence. Examples:
            - None (default): In-memory study (not persistent)
            - 'sqlite:///autotuner.db': SQLite database
            - 'postgresql://user:pass@host/db': PostgreSQL database
        load_if_exists: If True, load existing study from storage. If False, create new.
        direction: Optimization direction. 'maximize' for throughput (default), 'minimize' for latency.
        show_progress_bar: Whether to show Optuna's built-in progress bar.
        catch_exceptions: If True, catch exceptions during trials and mark as failed.
        n_jobs: Number of parallel jobs. 1 means sequential, -1 means use all CPUs.
            Note: Helion already parallelizes compilation, so this controls trial-level parallelism.
    """

    n_trials: int = 100
    timeout: float | None = None
    sampler: str | BaseSampler = "tpe"
    sampler_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    pruner: str | Any | None = "median"
    pruner_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    study_name: str | None = None
    storage: str | None = None
    load_if_exists: bool = True
    direction: str = "maximize"
    show_progress_bar: bool = True
    catch_exceptions: bool = True
    n_jobs: int = 1


class OptunaSearch(BaseSearch):
    """Optuna-based autotuner for Helion kernels.

    This autotuner uses Optuna's sophisticated optimization algorithms to search
    for optimal kernel configurations. Optuna provides state-of-the-art sampling
    algorithms like TPE (Tree-structured Parzen Estimator), CMA-ES, and Gaussian
    Processes, along with pruning capabilities for early stopping.

    The autotuner automatically maps Helion's configuration space to Optuna's
    search space, handling block sizes, loop orders, and other kernel parameters.

    Args:
        bound_kernel: The kernel to autotune.
        args: Arguments to pass to the kernel.
        params: OptunaSearchParams instance with configuration options.
        **kwargs: Additional keyword arguments passed to BaseSearch.

    Example:
        >>> params = OptunaSearchParams(
        ...     n_trials=200,
        ...     sampler="tpe",
        ...     storage="sqlite:///helion_autotune.db",
        ...     study_name="my_kernel_optimization",
        ... )
        >>> autotuner = OptunaSearch(bound_kernel, args, params=params)
        >>> best_config = autotuner.autotune()
    """

    def __init__(
        self,
        bound_kernel: BoundKernel,
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

        super().__init__(bound_kernel, args)

        # Store or create default params
        self.params = params if params is not None else OptunaSearchParams()

        # Apply environment variable overrides
        self._apply_env_overrides()

        # Initialize config generation for encoding/decoding
        self.config_gen = ConfigGeneration(self.config_spec)

        # Study will be created in _autotune
        self.study: optuna.Study | None = None

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to parameters."""
        # Override n_trials from env if set
        if env_trials := os.environ.get("HELION_AUTOTUNE_OPTUNA_TRIALS"):
            self.params.n_trials = int(env_trials)

        # Override timeout from env if set
        if env_timeout := os.environ.get("HELION_AUTOTUNE_OPTUNA_TIMEOUT"):
            self.params.timeout = float(env_timeout)

        # Override sampler from env if set
        if env_sampler := os.environ.get("HELION_AUTOTUNE_OPTUNA_SAMPLER"):
            self.params.sampler = env_sampler

        # Override storage from env if set
        if env_storage := os.environ.get("HELION_AUTOTUNE_OPTUNA_STORAGE"):
            self.params.storage = env_storage

        # Override study name from env if set
        if env_study := os.environ.get("HELION_AUTOTUNE_OPTUNA_STUDY"):
            self.params.study_name = env_study

    def _create_sampler(self) -> BaseSampler:
        """Create Optuna sampler based on configuration.

        Returns:
            Configured Optuna sampler instance.

        Raises:
            ValueError: If sampler name is not recognized.
        """
        if not isinstance(self.params.sampler, str):
            # Already a sampler instance
            return self.params.sampler

        sampler_name = self.params.sampler.lower()
        kwargs = self.params.sampler_kwargs

        if sampler_name == "tpe":
            return optuna.samplers.TPESampler(**kwargs)
        if sampler_name == "cmaes":
            return optuna.samplers.CmaEsSampler(**kwargs)
        if sampler_name == "gp":
            return optuna.samplers.GPSampler(**kwargs)
        if sampler_name == "random":
            return optuna.samplers.RandomSampler(**kwargs)
        if sampler_name == "grid":
            # GridSampler requires search_space parameter
            if "search_space" not in kwargs:
                raise ValueError(
                    "GridSampler requires 'search_space' in sampler_kwargs"
                )
            return optuna.samplers.GridSampler(**kwargs)
        raise ValueError(
            f"Unknown sampler: {sampler_name}. "
            f"Supported: 'tpe', 'cmaes', 'gp', 'random', 'grid'"
        )

    def _create_pruner(self) -> pruners.BasePruner | None:
        """Create Optuna pruner based on configuration.

        Returns:
            Configured Optuna pruner instance or None.

        Raises:
            ValueError: If pruner name is not recognized.
        """
        if self.params.pruner is None:
            return None

        if not isinstance(self.params.pruner, str):
            # Already a pruner instance
            return self.params.pruner

        pruner_name = self.params.pruner.lower()
        kwargs = self.params.pruner_kwargs

        if pruner_name == "median":
            return optuna.pruners.MedianPruner(**kwargs)
        if pruner_name == "hyperband":
            return optuna.pruners.HyperbandPruner(**kwargs)
        if pruner_name == "percentile":
            return optuna.pruners.PercentilePruner(**kwargs)
        if pruner_name == "threshold":
            return optuna.pruners.ThresholdPruner(**kwargs)
        raise ValueError(
            f"Unknown pruner: {pruner_name}. "
            f"Supported: 'median', 'hyperband', 'percentile', 'threshold', None"
        )

    def _suggest_config(self, trial: optuna.Trial) -> Config:
        """Suggest a configuration using Optuna trial.

        This method maps Helion's configuration space to Optuna's suggestion API,
        creating appropriate suggestions for each parameter type.

        Args:
            trial: Optuna trial object.

        Returns:
            Suggested configuration.
        """

        def suggest_fragment_value(
            fragment: (
                PowerOfTwoFragment
                | IntegerFragment
                | EnumFragment
                | BooleanFragment
                | PermutationFragment
            ),
            param_name: str,
        ) -> object:
            """Suggest a value for a single fragment.

            Args:
                fragment: The fragment to suggest a value for.
                param_name: The parameter name to use for Optuna.

            Returns:
                The suggested value.
            """
            match fragment:
                case PowerOfTwoFragment(low=low, high=high):
                    # Power of two: suggest in log space
                    min_exp = int(math.log2(low))
                    max_exp = int(math.log2(high))
                    exp = trial.suggest_int(f"{param_name}_exp", min_exp, max_exp)
                    return 2**exp

                case IntegerFragment(low=low, high=high):
                    # Integer range
                    return trial.suggest_int(param_name, low, high)

                case EnumFragment(choices=choices):
                    # Categorical choice
                    return trial.suggest_categorical(param_name, choices)

                case BooleanFragment():
                    # Boolean as categorical
                    return trial.suggest_categorical(param_name, [False, True])

                case PermutationFragment(length=n):
                    # Permutation: use rank-based encoding
                    # For each position i, we select which of the remaining elements (rank 0 to n-i-1)
                    # This avoids the issue of dynamic categorical choices
                    available = list(range(n))
                    perm = []
                    for i in range(n):
                        if len(available) == 1:
                            # Only one element left
                            perm.append(available[0])
                        else:
                            # Suggest rank (which remaining element to pick)
                            rank = trial.suggest_int(
                                f"{param_name}_rank{i}", 0, len(available) - 1
                            )
                            selected = available[rank]
                            perm.append(selected)
                            available.remove(selected)
                    return perm

                case _:
                    # Fallback: use default
                    return fragment.default()

        flat_config: list[object] = []

        # Iterate through flat spec fragments
        for idx, fragment in enumerate(self.config_gen.flat_spec):
            # Use index-based parameter names
            param_base = f"p{idx}"

            match fragment:
                case ListOf(inner=inner_fragment, length=length):
                    # List of values: suggest each independently
                    values = [
                        suggest_fragment_value(inner_fragment, f"{param_base}_{i}")
                        for i in range(length)
                    ]
                    flat_config.append(values)

                case _:
                    # Use the helper function for all other fragment types
                    flat_config.append(suggest_fragment_value(fragment, param_base))

        # Convert flat config to Config
        return self.config_gen.unflatten(flat_config)

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object.

        Returns:
            Performance metric (higher is better for 'maximize', lower for 'minimize').

        Raises:
            optuna.TrialPruned: If the trial should be pruned.
        """
        # Suggest a configuration
        config = self._suggest_config(trial)

        try:
            # Benchmark the configuration
            _, perf = self.benchmark(config)

            # Update best performance tracking
            if perf > self.best_perf_so_far:
                self.best_perf_so_far = perf
                self.log(f"New best: {perf:.3f} GB/s")

            # Return performance (GB/s by default)
            # Optuna maximizes by default, which aligns with throughput
            return perf

        except Exception as e:
            # If catch_exceptions is enabled, report failure and prune
            if self.params.catch_exceptions:
                self.log(f"Trial failed: {e}")
                # Report a very poor performance to indicate failure
                # Use pruning to stop the trial
                raise optuna.TrialPruned from e
            # Re-raise the exception
            raise

    def _autotune(self) -> Config:
        """Run Optuna optimization.

        Returns:
            Best configuration found.
        """
        # Create sampler and pruner
        sampler = self._create_sampler()
        pruner = self._create_pruner()

        # Generate study name if not provided
        study_name = self.params.study_name
        if study_name is None:
            study_name = f"helion_{self.kernel.kernel.fn.__name__}"

        # Create or load study
        self.log(f"Creating Optuna study: {study_name}")
        self.log(f"  Sampler: {sampler.__class__.__name__}")
        self.log(f"  Pruner: {pruner.__class__.__name__ if pruner else 'None'}")
        self.log(f"  Storage: {self.params.storage or 'in-memory'}")
        self.log(f"  Trials: {self.params.n_trials}")

        self.study = optuna.create_study(
            study_name=study_name,
            storage=self.params.storage,
            sampler=sampler,
            pruner=pruner,
            direction=self.params.direction,
            load_if_exists=self.params.load_if_exists,
        )

        # Log if we're resuming an existing study
        if self.params.load_if_exists and len(self.study.trials) > 0:
            self.log(
                f"Resuming existing study with {len(self.study.trials)} completed trials"
            )

        # Optimize
        self.study.optimize(
            self._objective,
            n_trials=self.params.n_trials,
            timeout=self.params.timeout,
            n_jobs=self.params.n_jobs,
            show_progress_bar=self.params.show_progress_bar,
            catch=(Exception,) if self.params.catch_exceptions else (),
        )

        # Log best trial
        best_trial = self.study.best_trial
        self.log("\nOptimization complete!")
        self.log(f"  Best trial: {best_trial.number}")
        self.log(f"  Best value: {best_trial.value:.3f} GB/s")
        self.log(f"  Total trials: {len(self.study.trials)}")
        self.log(
            f"  Completed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
        )
        self.log(
            f"  Pruned trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
        )
        self.log(
            f"  Failed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])}"
        )

        # Reconstruct best config
        return self._suggest_config(best_trial)

    def get_study(self) -> optuna.Study | None:
        """Get the Optuna study object for further analysis.

        This can be used to access trial history, visualize optimization,
        or perform additional analysis using Optuna's visualization tools.

        Returns:
            The Optuna study object, or None if optimization hasn't run yet.

        Example:
            >>> autotuner = OptunaSearch(bound_kernel, args)
            >>> best_config = autotuner.autotune()
            >>> study = autotuner.get_study()
            >>> # Visualize optimization history
            >>> import optuna.visualization as vis
            >>> vis.plot_optimization_history(study).show()
        """
        return self.study


__all__ = ["OptunaSearch", "OptunaSearchParams"]
