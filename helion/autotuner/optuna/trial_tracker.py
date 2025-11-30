"""Trial state tracking for Optuna optimization.

This module provides classes for tracking trial results, accuracy constraints,
and maintaining the best configuration found during optimization.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

    from ...runtime.config import Config
    from .types import Direction


class TrialTracker:
    """Tracks trial results and maintains best configuration.

    This class centralizes all state related to trial tracking:
    - Mapping trial numbers to configs
    - Tracking accuracy constraint violations
    - Maintaining the best config and performance
    - Global deduplication across studies (multi-study mode)

    This eliminates duplication between single and multi-study modes.
    """

    def __init__(
        self, direction: Direction, enable_deduplication: bool = False
    ) -> None:
        """Initialize the trial tracker.

        Args:
            direction: Optimization direction ('minimize' or 'maximize').
            enable_deduplication: If True, track all evaluated configs globally
                for cross-study deduplication (multi-study mode only).
        """
        import threading

        self.direction = direction
        self.enable_deduplication = enable_deduplication

        # Initialize best performance based on direction
        self.best_perf: float = math.inf if direction == "minimize" else -math.inf
        self.best_config: Config | None = None

        # Map trial number to config for reconstruction
        self._trial_configs: dict[int, Config] = {}

        # Map trial number to accuracy constraint value
        # <= 0 means accuracy satisfied, > 0 means violated
        self._trial_accuracy_errors: dict[int, float] = {}

        # Global set of evaluated config tuples for deduplication (multi-study mode)
        # Stores configs as hashable tuples of (param_name, param_value) sorted pairs
        # CRITICAL: Configs are added at GENERATION time (before compilation/benchmark)
        # to prevent multiple studies from working on the same config in parallel
        self._evaluated_config_hashes: set[tuple] = set()

        # Lock for thread-safe deduplication across parallel studies
        self._dedup_lock: threading.Lock = threading.Lock()

        # Statistics
        self._duplicate_count: int = 0

    def create_constraints_func(
        self,
        enable_dedup_constraint: bool = False,
    ) -> callable[[optuna.trial.FrozenTrial], list[float]]:
        """Create a constraint function for Optuna samplers.

        Args:
            enable_dedup_constraint: If True, add a constraint that penalizes
                duplicate configs (multi-study mode).

        Returns:
            Function that returns constraint values for trials.
        """

        def constraints_func(trial: optuna.trial.FrozenTrial) -> list[float]:
            """Constraint function for Optuna samplers.

            Returns constraint values where:
            - <= 0 means constraint satisfied
            - > 0 means constraint violated

            Constraints (in order):
            1. Accuracy constraint: accuracy error (if enabled)
            2. Deduplication constraint: 1.0 if duplicate, 0.0 if unique (if enabled)

            Args:
                trial: Completed trial to evaluate constraints for.

            Returns:
                List of constraint values.
            """
            constraints = []

            # Accuracy constraint
            # Look up stored accuracy error for this trial
            # Default to 0.0 (satisfied) if not found (e.g., for old trials in resumed studies)
            accuracy_error = self._trial_accuracy_errors.get(trial.number, 0.0)
            constraints.append(accuracy_error)

            # Deduplication constraint (if enabled)
            if enable_dedup_constraint and self.enable_deduplication:
                # Get the config for this trial
                config = self._trial_configs.get(trial.number)
                if config is not None:
                    # Check if this config was already evaluated by another study
                    # Note: We check against configs evaluated BEFORE this trial started
                    # The constraint is: 1.0 (violated) if duplicate, 0.0 (satisfied) if unique
                    is_dup = self.is_config_duplicate(config)
                    constraints.append(1.0 if is_dup else 0.0)
                else:
                    # Config not found - treat as non-duplicate
                    constraints.append(0.0)

            return constraints

        return constraints_func

    def store_accuracy_error(
        self, trial_number: int, accuracy_error: float | None
    ) -> None:
        """Store accuracy constraint error for a trial.

        Args:
            trial_number: Trial number.
            accuracy_error: Accuracy error value (None means not applicable).
        """
        if accuracy_error is not None:
            self._trial_accuracy_errors[trial_number] = accuracy_error

    def store_trial_config(self, trial_number: int, config: Config) -> None:
        """Store config for a trial number.

        Args:
            trial_number: Trial number.
            config: Configuration for this trial.
        """
        self._trial_configs[trial_number] = config

    def get_trial_config(self, trial_number: int) -> Config | None:
        """Get stored config for a trial number.

        Args:
            trial_number: Trial number.

        Returns:
            Config if found, None otherwise.
        """
        return self._trial_configs.get(trial_number)

    def update_best(self, perf: float, config: Config) -> bool:
        """Update best performance and config if this is an improvement.

        Args:
            perf: Performance value.
            config: Configuration that achieved this performance.

        Returns:
            True if this is a new best, False otherwise.
        """
        is_better = (self.direction == "minimize" and perf < self.best_perf) or (
            self.direction == "maximize" and perf > self.best_perf
        )

        if is_better:
            self.best_perf = perf
            self.best_config = config
            return True

        return False

    def _hash_config(self, config: Config) -> tuple:
        """Convert config to hashable tuple for deduplication.

        Args:
            config: Configuration to hash.

        Returns:
            Hashable tuple of sorted (param_name, param_value) pairs.
        """

        def make_hashable(value):
            """Recursively convert unhashable types to hashable equivalents."""
            if isinstance(value, list):
                return tuple(make_hashable(item) for item in value)
            elif isinstance(value, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
            elif isinstance(value, set):
                return tuple(sorted(make_hashable(item) for item in value))
            else:
                # Primitives (int, float, str, bool, None, tuple) are already hashable
                return value

        # Sort by parameter name for consistent hashing
        # Convert all values to hashable types
        return tuple(sorted((k, make_hashable(v)) for k, v in config.items()))

    def check_and_mark_config(self, config: Config) -> bool:
        """Atomically check if config is duplicate and mark it if not.

        CRITICAL: This must be called at config GENERATION time (before compilation)
        to prevent multiple studies from working on the same config in parallel.

        Args:
            config: Configuration to check and potentially mark.

        Returns:
            True if this is a NEW config (not a duplicate), False if duplicate.
        """
        if not self.enable_deduplication:
            return True  # Not a duplicate, proceed

        config_hash = self._hash_config(config)

        with self._dedup_lock:
            # Check-and-mark must be atomic
            if config_hash in self._evaluated_config_hashes:
                # Duplicate detected
                self._duplicate_count += 1
                return False
            # New config - mark it immediately before releasing lock
            self._evaluated_config_hashes.add(config_hash)
            return True

    def mark_config_evaluated(self, config: Config) -> None:
        """Mark a config as evaluated for deduplication.

        NOTE: This is now handled by check_and_mark_config() at generation time.
        Keeping this method for backward compatibility but it's a no-op.

        Args:
            config: Configuration that was evaluated (unused).
        """
        # No-op: configs are now marked at generation time
        del config  # Unused parameter

    def is_config_duplicate(self, config: Config) -> bool:
        """Check if a config has already been evaluated.

        NOTE: This is for constraint-based checking (after trial completion).
        For generation-time checking, use check_and_mark_config() instead.

        Args:
            config: Configuration to check.

        Returns:
            True if this config was already evaluated, False otherwise.
        """
        if not self.enable_deduplication:
            return False

        config_hash = self._hash_config(config)
        with self._dedup_lock:
            return config_hash in self._evaluated_config_hashes

    def get_deduplication_stats(self) -> dict[str, int]:
        """Get statistics about deduplication.

        Returns:
            Dictionary with deduplication statistics.
        """
        with self._dedup_lock:
            return {
                "unique_configs_evaluated": len(self._evaluated_config_hashes),
                "duplicate_configs_skipped": self._duplicate_count,
                "deduplication_enabled": self.enable_deduplication,
            }

    def get_performance_unit(self) -> str:
        """Get the unit string for performance metrics.

        Returns:
            'ms' for minimize (latency), 'GB/s' for maximize (throughput).
        """
        return "ms" if self.direction == "minimize" else "GB/s"

    def initialize_from_study(
        self,
        study: optuna.Study,
        config_reconstructor: callable[[optuna.trial.FrozenTrial], Config],
    ) -> int:
        """Initialize state from existing study trials.

        This is used when resuming a study from persistent storage.

        Args:
            study: Optuna study with existing trials.
            config_reconstructor: Function to reconstruct config from trial.

        Returns:
            Number of completed trials found.
        """
        import optuna

        # Find all completed trials with finite performance
        completed_trials = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and math.isfinite(t.value)
        ]

        if not completed_trials:
            return 0

        # Find the best trial based on optimization direction
        def get_trial_value(t: optuna.trial.FrozenTrial) -> float:
            assert t.value is not None  # Guaranteed by filter above
            return t.value

        if self.direction == "minimize":
            best_trial = min(completed_trials, key=get_trial_value)
        else:
            best_trial = max(completed_trials, key=get_trial_value)

        # Update best performance and config
        assert best_trial.value is not None  # Guaranteed by filter
        self.best_perf = best_trial.value
        self.best_config = config_reconstructor(best_trial)
        self._trial_configs[best_trial.number] = self.best_config

        return len(completed_trials)


__all__ = ["TrialTracker"]
