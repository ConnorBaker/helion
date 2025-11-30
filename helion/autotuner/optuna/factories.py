"""Factory functions for creating Optuna objects.

This module provides factory functions for creating Optuna samplers, pruners,
and studies with the appropriate configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence

if TYPE_CHECKING:
    import optuna
    from optuna import pruners
    from optuna.samplers import BaseSampler
    from optuna.storages import JournalStorage
    from optuna.storages import RDBStorage

    from .types import Direction
    from .types import PrunerName
    from .types import SamplerName


def create_sampler(
    sampler_name: SamplerName,
    sampler_kwargs: dict[str, Any],
    seed: int | None = None,
    constraints_func: Callable[[optuna.trial.FrozenTrial], Sequence[float]]
    | None = None,
) -> BaseSampler:
    """Create an Optuna sampler with the specified configuration.

    This factory function creates samplers for both inner studies and meta-studies,
    unifying the logic that was previously duplicated.

    Args:
        sampler_name: Name of the sampler to create.
        sampler_kwargs: Additional keyword arguments for the sampler.
        seed: Optional seed for the sampler. If provided, overrides any seed in kwargs.
        constraints_func: Optional constraint function for constraint-aware samplers.

    Returns:
        Configured Optuna sampler instance.

    Raises:
        ValueError: If sampler name is not recognized.
    """
    import optuna

    kwargs = dict(sampler_kwargs)

    # Add seed if provided (overrides any seed in kwargs)
    if seed is not None:
        kwargs["seed"] = seed

    # Add constraints function if provided
    if constraints_func is not None:
        kwargs["constraints_func"] = constraints_func

    match sampler_name:
        case "tpe":
            return optuna.samplers.TPESampler(**kwargs)
        case "cmaes":
            return optuna.samplers.CmaEsSampler(**kwargs)
        case "gp":
            return optuna.samplers.GPSampler(**kwargs)
        case "random":
            # RandomSampler doesn't support constraints_func
            kwargs.pop("constraints_func", None)
            return optuna.samplers.RandomSampler(**kwargs)
        case "grid":
            # GridSampler doesn't support constraints_func
            kwargs.pop("constraints_func", None)
            # GridSampler requires search_space parameter
            if "search_space" not in kwargs:
                raise ValueError(
                    "GridSampler requires 'search_space' in sampler_kwargs"
                )
            return optuna.samplers.GridSampler(**kwargs)
        case _:
            raise ValueError(f"Unknown sampler_name: {sampler_name}")


def create_pruner(
    pruner_name: PrunerName | None,
    pruner_kwargs: dict[str, Any],
) -> pruners.BasePruner | None:
    """Create an Optuna pruner with the specified configuration.

    Args:
        pruner_name: Name of the pruner to create, or None for no pruning.
        pruner_kwargs: Additional keyword arguments for the pruner.

    Returns:
        Configured Optuna pruner instance or None.

    Raises:
        ValueError: If pruner name is not recognized.
    """
    import optuna

    if pruner_name is None:
        return None

    match pruner_name:
        case "median":
            return optuna.pruners.MedianPruner(**pruner_kwargs)
        case "hyperband":
            return optuna.pruners.HyperbandPruner(**pruner_kwargs)
        case "percentile":
            return optuna.pruners.PercentilePruner(**pruner_kwargs)
        case "threshold":
            return optuna.pruners.ThresholdPruner(**pruner_kwargs)
        case _:
            raise ValueError(f"Unknown pruner_name: {pruner_name}")


def create_storage(
    storage_spec: str | None,
) -> RDBStorage | JournalStorage | None:
    """Create Optuna storage from specification.

    Args:
        storage_spec: Storage specification. Can be:
            - None: No persistence (in-memory)
            - File path: JournalStorage (recommended for parallel)
            - Database URL: RDBStorage

    Returns:
        Storage instance or None for in-memory.
    """
    if not storage_spec:
        return None

    import optuna

    # Detect if this is a database URL (contains '://') or a file path
    if "://" in storage_spec:
        # Database URL - use RDBStorage
        return optuna.storages.RDBStorage(
            url=storage_spec,
            engine_kwargs={"connect_args": {"check_same_thread": False}},
        )
    # File path - use JournalStorage (recommended for parallel execution)
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    return JournalStorage(JournalFileBackend(storage_spec))


def create_study(
    study_name: str,
    sampler: BaseSampler,
    direction: Direction,
    storage: RDBStorage | JournalStorage | None = None,
    pruner: pruners.BasePruner | None = None,
    load_if_exists: bool = True,
) -> optuna.Study:
    """Create or load an Optuna study.

    Args:
        study_name: Name of the study.
        sampler: Sampler instance to use.
        direction: Optimization direction.
        storage: Optional storage for persistence.
        pruner: Optional pruner for early stopping.
        load_if_exists: If True, load existing study from storage.

    Returns:
        Configured Optuna study instance.
    """
    import optuna

    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=load_if_exists,
    )


def get_storage_description(storage: RDBStorage | JournalStorage | None) -> str:
    """Get a human-readable description of the storage type.

    Args:
        storage: Storage instance.

    Returns:
        Description string for logging.
    """
    if storage is None:
        return "in-memory"

    # Check if this is RDBStorage
    if hasattr(storage, "engine"):
        url = str(storage.engine.url)
        return f"RDBStorage({url})"

    # Check if this is JournalStorage
    if hasattr(storage, "_backend"):
        # Try to get the file path
        backend = storage._backend
        if hasattr(backend, "_file_path"):
            return f"JournalStorage({backend._file_path})"
        return "JournalStorage"

    return str(type(storage).__name__)


__all__ = [
    "create_pruner",
    "create_sampler",
    "create_storage",
    "create_study",
    "get_storage_description",
]
