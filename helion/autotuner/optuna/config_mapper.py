"""Mapping between Helion config space and Optuna parameters.

This module handles the bidirectional conversion between Helion's configuration
space (defined by ConfigSpec) and Optuna's parameter space.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..config_fragment import BooleanFragment
from ..config_fragment import ConfigSpecFragment
from ..config_fragment import EnumFragment
from ..config_fragment import IntegerFragment
from ..config_fragment import ListOf
from ..config_fragment import PermutationFragment
from ..config_fragment import PowerOfTwoFragment
from ..config_generation import ConfigGeneration

if TYPE_CHECKING:
    import optuna

    from ...runtime.config import Config
    from ...runtime.kernel import BoundKernel


class OptunaConfigMapper:
    """Maps Helion config space to Optuna parameter space.

    This class handles the conversion between Helion's configuration
    representation and Optuna's trial suggestion API, making it easy
    to suggest new configs and reconstruct configs from trial parameters.
    """

    def __init__(self, bound_kernel: BoundKernel) -> None:  # type: ignore[reportMissingTypeArgument]
        """Initialize the config mapper.

        Args:
            bound_kernel: The kernel being optimized.
        """
        self.config_gen = ConfigGeneration(bound_kernel.config_spec)

    def suggest_config(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> Config:
        """Suggest a configuration using Optuna trial.

        This method maps Helion's configuration space to Optuna's suggestion API,
        creating appropriate suggestions for each parameter type.

        For FrozenTrial (e.g., when resuming from persistent storage), the suggest_*
        methods return the stored parameter values, making this work for both
        generating new configs and reconstructing existing ones.

        Args:
            trial: Optuna trial object (Trial or FrozenTrial).

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


__all__ = ["OptunaConfigMapper"]
