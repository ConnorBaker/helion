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
from typing import Literal
from typing import Sequence

from . import exc
from .base_search import BaseSearch
from .config_fragment import BooleanFragment
from .config_fragment import ConfigSpecFragment
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


SamplerName = Literal["tpe", "cmaes", "gp", "random", "grid"]
PrunerName = Literal["median", "hyperband", "percentile", "threshold"]
Direction = Literal["maximize", "minimize"]


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
        batch_size: Number of trials to run in parallel per batch. Defaults to 10.
            Set to 1 to disable batching.
    """

    n_trials: int = 100
    timeout: float | None = None
    sampler: SamplerName | BaseSampler = "tpe"
    sampler_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    pruner: PrunerName | Any | None = "median"
    pruner_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    study_name: str | None = None
    storage: str | None = None
    load_if_exists: bool = True
    direction: Direction = "maximize"
    show_progress_bar: bool = True
    catch_exceptions: bool = True
    batch_size: int = 10


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

        # Override batch size from env if set
        if env_batch := os.environ.get("HELION_AUTOTUNE_OPTUNA_BATCH_SIZE"):
            self.params.batch_size = int(env_batch)

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

        kwargs = self.params.sampler_kwargs

        match self.params.sampler:
            case "tpe":
                return optuna.samplers.TPESampler(**kwargs)
            case "cmaes":
                return optuna.samplers.CmaEsSampler(**kwargs)
            case "gp":
                return optuna.samplers.GPSampler(**kwargs)
            case "random":
                return optuna.samplers.RandomSampler(**kwargs)
            case "grid":
                # GridSampler requires search_space parameter
                if "search_space" not in kwargs:
                    raise ValueError(
                        "GridSampler requires 'search_space' in sampler_kwargs"
                    )
                return optuna.samplers.GridSampler(**kwargs)

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

        kwargs = self.params.pruner_kwargs

        match self.params.pruner:
            case "median":
                return optuna.pruners.MedianPruner(**kwargs)
            case "hyperband":
                return optuna.pruners.HyperbandPruner(**kwargs)
            case "percentile":
                return optuna.pruners.PercentilePruner(**kwargs)
            case "threshold":
                return optuna.pruners.ThresholdPruner(**kwargs)

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
            fragment: ConfigSpecFragment,
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

    def _create_study(self) -> optuna.Study:
        """Create or load an Optuna study.

        Returns:
            Configured Optuna study instance.
        """
        sampler = self._create_sampler()
        pruner = self._create_pruner()

        study_name = (
            self.params.study_name or f"helion_{self.kernel.kernel.fn.__name__}"
        )

        # Log study configuration
        self.log(f"Creating Optuna study: {study_name}")
        self.log(f"  Sampler: {sampler.__class__.__name__}")
        self.log(f"  Pruner: {pruner.__class__.__name__ if pruner else 'None'}")
        self.log(f"  Storage: {self.params.storage or 'in-memory'}")
        self.log(f"  Trials: {self.params.n_trials}")
        self.log(f"  Batch size: {self.params.batch_size}")

        study = optuna.create_study(
            study_name=study_name,
            storage=self.params.storage,
            sampler=sampler,
            pruner=pruner,
            direction=self.params.direction,
            load_if_exists=self.params.load_if_exists,
        )

        # Log if resuming
        if self.params.load_if_exists and len(study.trials) > 0:
            self.log(
                f"Resuming existing study with {len(study.trials)} completed trials"
            )

        return study

    def _handle_trial_failure(self, trial: optuna.Trial, error: Exception) -> None:
        """Handle a failed trial.

        Args:
            trial: The Optuna trial that failed.
            error: The exception that caused the failure.

        Raises:
            Exception: Re-raises the error if catch_exceptions is False.
        """
        if self.params.catch_exceptions:
            self.log(f"Trial {trial.number} failed: {error}")
            self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            raise error

    def _report_result(self, trial: optuna.Trial, perf: float) -> None:
        """Report a successful trial result to Optuna.

        Args:
            trial: The Optuna trial.
            perf: Performance metric (GB/s).
        """
        if perf > self.best_perf_so_far:
            self.best_perf_so_far = perf
            self.log(f"New best: {perf:.3f} GB/s")

        self.study.tell(trial, perf)

    def _log_final_statistics(self) -> None:
        """Log final optimization statistics."""
        best_trial = self.study.best_trial
        trials = self.study.trials
        completed = sum(
            1 for t in trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        failed = sum(1 for t in trials if t.state == optuna.trial.TrialState.FAIL)

        self.log("\nOptimization complete!")
        self.log(f"  Best trial: {best_trial.number}")
        self.log(f"  Best value: {best_trial.value:.3f} GB/s")
        self.log(f"  Total trials: {len(trials)}")
        self.log(f"  Completed trials: {completed}")
        self.log(f"  Failed trials: {failed}")

    def _autotune(self) -> Config:
        """Run Optuna optimization with pipelined compilation and benchmarking.

        Uses Optuna's ask-and-tell interface with true pipelining to maximize
        GPU and CPU utilization simultaneously:

        - Compiles batch N+1 (CPU-bound, parallel in subprocesses)
        - While benchmarking batch N sequentially (GPU-bound)
        - Achieves overlap to keep both resources busy

        When autotune_precompile is disabled, falls back to simple batching.

        Returns:
            Best configuration found.

        Raises:
            InvalidAPIUsage: If pipelining is enabled but device is not a GPU.
        """
        from itertools import starmap
        import math
        import time

        # Pipelining only makes sense on GPU (CPU compilation overlapping GPU benchmarking)
        if self.settings.autotune_precompile and self.kernel.env.device.type != "cuda":
            raise exc.InvalidAPIUsage(
                "Pipelined compilation (autotune_precompile) requires CUDA device. "
                f"Current device: {self.kernel.env.device.type}. "
                "Disable autotune_precompile when benchmarking on CPU."
            )

        self.study = self._create_study()

        start_time = time.time()
        trials_completed = 0

        def prepare_batch(
            n: int,
        ) -> tuple[list[optuna.Trial], list[Config]] | None:
            """Prepare a batch of trials and configs."""
            if n <= 0:
                return None

            batch_trials = []
            batch_configs = []

            for _ in range(n):
                trial = self.study.ask()
                try:
                    batch_trials.append(trial)
                    batch_configs.append(self._suggest_config(trial))
                except Exception as e:
                    batch_trials.pop()
                    self._handle_trial_failure(trial, e)

            return (batch_trials, batch_configs) if batch_configs else None

        def is_timeout() -> bool:
            """Check if timeout has been reached."""
            return bool(
                self.params.timeout and time.time() - start_time >= self.params.timeout
            )

        def report_batch(trials: list[optuna.Trial], results: Sequence[Any]) -> None:
            """Report batch results to Optuna."""
            for trial, result in zip(trials, results, strict=True):
                match result.status:
                    case "ok":
                        self._report_result(trial, result.perf)
                    case _:
                        self._handle_trial_failure(
                            trial,
                            RuntimeError(f"Benchmark failed: {result.status}"),
                        )

        # If precompilation is disabled, use simple batching
        if not self.settings.autotune_precompile:
            current_batch = prepare_batch(
                min(self.params.batch_size, self.params.n_trials)
            )

            while current_batch is not None and not is_timeout():
                current_trials, current_configs = current_batch
                batch_num = trials_completed // self.params.batch_size + 1

                results = self.parallel_benchmark(
                    current_configs, desc=f"Batch {batch_num}"
                )
                report_batch(current_trials, results)

                trials_completed += len(current_configs)
                remaining = self.params.n_trials - trials_completed
                current_batch = (
                    prepare_batch(min(self.params.batch_size, remaining))
                    if remaining > 0
                    else None
                )

        else:
            # Pipelined mode: manually manage compilation and benchmarking
            from .base_search import BenchmarkResult
            from .base_search import PrecompileFuture

            @dataclasses.dataclass
            class PipelineBatch:
                """State for a batch in the compilation/benchmarking pipeline."""

                trials: list[optuna.Trial]
                configs: list[Config]
                fns: list[object]
                futures: list[PrecompileFuture]

            def start_compilation(
                trials: list[optuna.Trial], configs: list[Config]
            ) -> PipelineBatch:
                """Start compiling a batch of configs."""
                fns = [
                    self.kernel.compile_config(cfg, allow_print=False)
                    for cfg in configs
                ]
                futures = list(
                    starmap(
                        self.start_precompile_and_check_for_hangs,
                        zip(configs, fns, strict=True),
                    )
                )
                return PipelineBatch(trials, configs, fns, futures)

            def benchmark_batch(
                batch: PipelineBatch, is_working: list[bool]
            ) -> list[BenchmarkResult]:
                """Benchmark a compiled batch."""
                results = []
                for idx, (fn, ok, future) in enumerate(
                    zip(batch.fns, is_working, batch.futures, strict=True)
                ):
                    config = batch.configs[idx]
                    compile_time = (
                        future.elapsed
                        if future.process is not None and future.started
                        else None
                    )

                    if ok:
                        perf = self.benchmark_function(config, fn)
                        status = "ok" if math.isfinite(perf) else "error"
                    else:
                        perf = math.inf
                        status = (
                            "timeout" if future.failure_reason == "timeout" else "error"
                        )

                    results.append(
                        BenchmarkResult(
                            config=config,
                            fn=fn,
                            perf=perf,
                            status=status,
                            compile_time=compile_time,
                        )
                    )
                return results

            # Start pipeline with first batch
            first_batch = prepare_batch(
                min(self.params.batch_size, self.params.n_trials)
            )
            if not first_batch:
                self._log_final_statistics()
                return self._suggest_config(self.study.best_trial)

            pending = start_compilation(*first_batch)

            while not is_timeout():
                # Prepare next batch
                remaining = (
                    self.params.n_trials - trials_completed - len(pending.configs)
                )
                next_batch = (
                    prepare_batch(min(self.params.batch_size, remaining))
                    if remaining > 0
                    else None
                )

                # Wait for current batch compilation
                batch_num = trials_completed // self.params.batch_size + 1
                desc = (
                    f"Batch {batch_num} precompiling"
                    if self.settings.autotune_progress_bar
                    else None
                )
                is_working = PrecompileFuture.wait_for_all(pending.futures, desc=desc)

                # Start next batch compilation NOW (overlaps with benchmarking)
                next_pending = start_compilation(*next_batch) if next_batch else None

                # Benchmark current batch while next compiles
                results = benchmark_batch(pending, is_working)
                report_batch(pending.trials, results)

                trials_completed += len(pending.configs)

                if next_pending is None:
                    break
                pending = next_pending

        self._log_final_statistics()
        return self._suggest_config(self.study.best_trial)

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
