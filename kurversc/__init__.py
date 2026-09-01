"""KurveRSC: a one-call GraphReduce configuration optimizer."""

from .core import fit, predict
from .logging import configure_logging
from .relbench import (
    RelBenchProblem,
    load_relbench_problem,
    relbench_problem_from_objects,
)
from .search import FittedModel, FitResult, GraphConfig, Trial, incremental_configs
from .specs import GraphLabels, Labels, Relationship, Table

__version__ = "0.1.1"

__all__ = [
    "FitResult",
    "FittedModel",
    "GraphConfig",
    "GraphLabels",
    "Labels",
    "Relationship",
    "RelBenchProblem",
    "Table",
    "Trial",
    "configure_logging",
    "fit",
    "predict",
    "incremental_configs",
    "load_relbench_problem",
    "relbench_problem_from_objects",
]
