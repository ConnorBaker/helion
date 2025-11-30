"""Configuration for Optuna-based autotuning.

This module provides configuration dataclasses and environment variable
handling for the Optuna autotuner.
"""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

if TYPE_CHECKING:
    from .types import Direction
    from .types import PrunerName
    from .types import SamplerName


@dataclasses.dataclass
class OptunaSearchParams:
    """Parameters for OptunaSearch.

    OptunaSearch supports both single-study and multi-study (meta-study) optimization:
    - When n_studies=1: Runs a single optimization study with the specified sampler
    - When n_studies>1: Runs a meta-study where each meta-trial is an inner study with
      a different seed. The meta-study uses a meta-sampler and meta-pruner to adaptively
      allocate resources to the most promising studies.

    Attributes:
        n_trials: Number of trials per study. For multi-study mode, this is trials per inner study.
        n_studies: Number of independent studies to run with different seeds. Default 1.
            When > 1, enables meta-study mode with adaptive resource allocation.
        timeout: Stop study after this number of seconds. None means no timeout.
        sampler_name: Optuna sampler name for inner studies. Only string names supported.
            Supported: 'tpe' (default), 'cmaes', 'gp', 'random', 'grid'.
            For multi-study mode, each inner study gets its own sampler with a different seed.
        sampler_kwargs: Additional kwargs passed to inner study sampler constructor.
        meta_sampler_name: Optuna sampler name for meta-study (only used when n_studies > 1).
            Supported: 'tpe', 'cmaes', 'gp', 'random', 'grid'. Default: 'random'.
            The meta-sampler explores which seed configurations are most promising.
        meta_sampler_kwargs: Additional kwargs passed to meta-sampler constructor.
        meta_pruner_name: Pruner for meta-study (only used when n_studies > 1).
            Controls which inner studies to stop early. Only string names supported.
            Recommended: 'median' (stop if below median), 'hyperband' (adaptive).
            Default: 'median'.
        meta_pruner_kwargs: Additional kwargs passed to meta-pruner constructor.
        report_interval: Report best value to meta-study every N trials (only for n_studies > 1).
            Smaller values allow more responsive pruning but add overhead. Default: 1.
        study_name: Name of the Optuna study. Used for persistence and logging.
        storage: Storage specification for study persistence. Examples:
            - None (default): In-memory study (not persistent)
            - 'optuna_journal.log': File path for JournalStorage (recommended for parallel execution)
            - 'sqlite:///autotuner.db': SQLite database URL (for sequential execution only)
            - 'postgresql://user:pass@host/db': PostgreSQL database URL
            Note: File paths (without '://') use JournalStorage which is optimized for parallel trials.
            Database URLs use RDBStorage. SQLite with RDBStorage doesn't support parallel execution well.
            For multi-study mode (n_studies > 1), storage applies to the meta-study; inner studies
            use in-memory storage.
        load_if_exists: If True, load existing study from storage. If False, create new.
        direction: Optimization direction. 'minimize' for latency in ms (default), 'maximize' for throughput.
        show_progress_bar: Whether to show a rich progress bar during optimization.
        catch_exceptions: If True, catch exceptions during trials and mark as failed.
        max_configs_ahead: Maximum number of configs to generate ahead of benchmark completion. Defaults to 20.
            Controls the exploration vs exploitation tradeoff: higher values explore more configurations
            before learning from results, lower values learn from each result before generating new configs.
            This is independent of compilation parallelism (controlled by autotune_precompile_jobs setting).
            In single-study mode: used directly for stream processing.
            In multi-study mode: divided among parallel studies (each gets max_configs_ahead // n_studies).
        enable_deduplication: Whether to skip duplicate configs across parallel studies.
            Only applies in multi-study mode (n_studies > 1). When enabled, configs that have
            already been suggested by any study are detected and skipped BEFORE compilation/benchmarking,
            preventing wasted GPU time. Uses thread-safe check-and-mark at generation time.
            Default: True for multi-study, False for single-study.
    """

    n_trials: int = 100
    n_studies: int = 1
    timeout: float | None = None
    sampler_name: SamplerName = "tpe"
    sampler_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: cast("dict[str, Any]", {})
    )
    meta_sampler_name: SamplerName | None = "random"
    meta_sampler_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: cast("dict[str, Any]", {})
    )
    meta_pruner_name: PrunerName | None = "median"
    meta_pruner_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: cast("dict[str, Any]", {})
    )
    report_interval: int = 1
    study_name: str | None = None
    storage: str | None = None
    load_if_exists: bool = True
    direction: Direction = "minimize"
    show_progress_bar: bool = True
    catch_exceptions: bool = True
    max_configs_ahead: int = 20
    enable_deduplication: bool | None = None  # None means auto (True for multi-study, False otherwise)

    def __post_init__(self) -> None:
        """Validate parameters after initialization.

        Raises:
            ValueError: If meta-study parameters are set when n_studies=1, or if
                seed is specified in sampler_kwargs when n_studies>1.
        """
        # Set default deduplication behavior: enabled for multi-study, disabled otherwise
        if self.enable_deduplication is None:
            self.enable_deduplication = self.n_studies > 1

        if self.n_studies == 1:
            # For single-study mode, meta parameters should not be customized
            # Check if user explicitly set any meta parameters to non-default values
            has_custom_meta_sampler = (
                self.meta_sampler_name != "random" or len(self.meta_sampler_kwargs) > 0
            )
            has_custom_meta_pruner = (
                self.meta_pruner_name != "median" or len(self.meta_pruner_kwargs) > 0
            )

            if has_custom_meta_sampler or has_custom_meta_pruner:
                raise ValueError(
                    "Meta-study parameters (meta_sampler_name, meta_sampler_kwargs, "
                    "meta_pruner_name, meta_pruner_kwargs) should not be set when n_studies=1. "
                    "These parameters are only used for multi-study mode (n_studies > 1)."
                )
        else:
            # For multi-study mode, seed should not be set in sampler_kwargs
            # Seeds are automatically managed based on meta-trial number
            if "seed" in self.sampler_kwargs:
                raise ValueError(
                    "Cannot set 'seed' in sampler_kwargs when n_studies > 1. "
                    "In multi-study mode, seeds are automatically set for each inner study "
                    "based on the meta-trial number to ensure diverse exploration. "
                    f"Remove 'seed' from sampler_kwargs (currently set to {self.sampler_kwargs['seed']})."
                )


def apply_env_overrides(params: OptunaSearchParams) -> None:
    """Apply environment variable overrides to parameters in-place.

    Environment variables:
        HELION_AUTOTUNE_OPTUNA_TRIALS: Override n_trials
        HELION_AUTOTUNE_OPTUNA_STUDIES: Override n_studies
        HELION_AUTOTUNE_OPTUNA_TIMEOUT: Override timeout
        HELION_AUTOTUNE_OPTUNA_SAMPLER: Override sampler_name
        HELION_AUTOTUNE_OPTUNA_STORAGE: Override storage
        HELION_AUTOTUNE_OPTUNA_STUDY: Override study_name
        HELION_AUTOTUNE_OPTUNA_MAX_CONFIGS_AHEAD: Override max_configs_ahead

    Args:
        params: OptunaSearchParams to modify.
    """
    if env_trials := os.environ.get("HELION_AUTOTUNE_OPTUNA_TRIALS"):
        params.n_trials = int(env_trials)

    if env_studies := os.environ.get("HELION_AUTOTUNE_OPTUNA_STUDIES"):
        params.n_studies = int(env_studies)

    if env_timeout := os.environ.get("HELION_AUTOTUNE_OPTUNA_TIMEOUT"):
        params.timeout = float(env_timeout)

    if env_sampler := os.environ.get("HELION_AUTOTUNE_OPTUNA_SAMPLER"):
        params.sampler_name = env_sampler  # type: ignore[assignment]

    if env_storage := os.environ.get("HELION_AUTOTUNE_OPTUNA_STORAGE"):
        params.storage = env_storage

    if env_study := os.environ.get("HELION_AUTOTUNE_OPTUNA_STUDY"):
        params.study_name = env_study

    if env_configs_ahead := os.environ.get("HELION_AUTOTUNE_OPTUNA_MAX_CONFIGS_AHEAD"):
        params.max_configs_ahead = int(env_configs_ahead)


__all__ = ["OptunaSearchParams", "apply_env_overrides"]
