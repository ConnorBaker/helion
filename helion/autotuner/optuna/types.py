"""Type aliases and protocols for Optuna integration.

This module contains shared type definitions used across the Optuna autotuner
implementation, including sampler types, optimization directions, and protocols.
"""

from __future__ import annotations

from typing import Literal

# Sampler type names supported by Optuna
SamplerName = Literal["tpe", "cmaes", "gp", "random", "grid"]

# Pruner type names supported by Optuna
PrunerName = Literal["median", "hyperband", "percentile", "threshold"]

# Optimization direction
Direction = Literal["maximize", "minimize"]


__all__ = ["Direction", "PrunerName", "SamplerName"]
