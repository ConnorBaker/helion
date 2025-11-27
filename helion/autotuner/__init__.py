from __future__ import annotations

from .config_fragment import BooleanFragment as BooleanFragment
from .config_fragment import EnumFragment as EnumFragment
from .config_fragment import IntegerFragment as IntegerFragment
from .config_fragment import ListOf as ListOf
from .config_fragment import PowerOfTwoFragment as PowerOfTwoFragment
from .config_spec import ConfigSpec as ConfigSpec
from .de_surrogate_hybrid import DESurrogateHybrid as DESurrogateHybrid
from .differential_evolution import (
    DifferentialEvolutionSearch as DifferentialEvolutionSearch,
)
from .effort_profile import AutotuneEffortProfile as AutotuneEffortProfile
from .effort_profile import DifferentialEvolutionConfig as DifferentialEvolutionConfig
from .effort_profile import PatternSearchConfig as PatternSearchConfig
from .effort_profile import RandomSearchConfig as RandomSearchConfig
from .finite_search import FiniteSearch as FiniteSearch
from .local_cache import LocalAutotuneCache as LocalAutotuneCache
from .local_cache import StrictLocalAutotuneCache as StrictLocalAutotuneCache
from .optuna_search import OptunaSearch as OptunaSearch
from .optuna_search import OptunaSearchParams as OptunaSearchParams
from .pattern_search import PatternSearch as PatternSearch
from .pipeline import CompilationBatch as CompilationBatch
from .pipeline import PipelineCallbacks as PipelineCallbacks
from .pipeline import PipelinedBatchExecutor as PipelinedBatchExecutor
from .random_search import RandomSearch as RandomSearch
from .surrogate_pattern_search import LFBOPatternSearch

search_algorithms = {
    "DESurrogateHybrid": DESurrogateHybrid,
    "LFBOPatternSearch": LFBOPatternSearch,
    "DifferentialEvolutionSearch": DifferentialEvolutionSearch,
    "FiniteSearch": FiniteSearch,
    "OptunaSearch": OptunaSearch,
    "PatternSearch": PatternSearch,
    "RandomSearch": RandomSearch,
}

cache_classes = {
    "LocalAutotuneCache": LocalAutotuneCache,
    "StrictLocalAutotuneCache": StrictLocalAutotuneCache,
}
