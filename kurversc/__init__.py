"""KurveRSC: a one-call GraphReduce configuration optimizer."""

from .core import estimate_features, fit, predict
from .feature_audit import ConfigFeatureEstimate, FeaturePreflight, TableFeatureEstimate
from .logging import configure_logging
from .relbench import (
    RelBenchProblem,
    load_relbench_problem,
    relbench_problem_from_objects,
)
from .search import FittedModel, FitResult, GraphConfig, Trial, incremental_configs
from .snowflake import (
    SNOWFLAKE_LEARNERS,
    SnowflakeBackend,
    SnowflakeEvaluation,
    SnowflakeGraphMaterializer,
    SnowflakeMLRunner,
    SnowflakeModelReference,
    SnowflakeNativeMLRunner,
    fit_snowflake,
    snowflake_model_runner,
)
from .specs import GraphLabels, Labels, Relationship, Table

__version__ = "0.1.2"

__all__ = [
    "FitResult",
    "FittedModel",
    "ConfigFeatureEstimate",
    "FeaturePreflight",
    "GraphConfig",
    "GraphLabels",
    "Labels",
    "Relationship",
    "RelBenchProblem",
    "SnowflakeBackend",
    "SnowflakeEvaluation",
    "SnowflakeGraphMaterializer",
    "SnowflakeMLRunner",
    "SnowflakeModelReference",
    "SnowflakeNativeMLRunner",
    "SNOWFLAKE_LEARNERS",
    "Table",
    "TableFeatureEstimate",
    "Trial",
    "configure_logging",
    "estimate_features",
    "fit",
    "fit_snowflake",
    "snowflake_model_runner",
    "predict",
    "incremental_configs",
    "load_relbench_problem",
    "relbench_problem_from_objects",
]
