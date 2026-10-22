"""KurveRSC: a one-call GraphReduce configuration optimizer."""

from .core import estimate_features, fit, predict
from .feature_audit import ConfigFeatureEstimate, FeaturePreflight, TableFeatureEstimate
from .logging import configure_logging
from .learned_messages import LearnedMessageFamily
from .pooled_messages import PooledMessageFamily
from .peer_history import PeerHistoryFamily
from .peer_context import PeerContext
from .relbench import add_relbench_peer_context
from .relational_inputs import SemanticPayloadFamily
from .planner_collection import PlannerCollectionResult, collect_planner_search
from .planner_profiles import (
    PLANNER_PROFILE_VERSION,
    ColumnProfile,
    DatabaseProfile,
    EventFrequencyProfile,
    RelationshipProfile,
    RuntimeProfile,
    TableProfile,
    TaskProfile,
)
from .planner_traces import (
    PLANNER_TRACE_VERSION,
    CandidateObservation,
    PlannerDatasetSplit,
    PlannerEpisode,
    episodes_from_fit_result,
    read_episodes_jsonl,
    split_episodes_by_database,
    write_episodes_jsonl,
)
from .relbench import (
    RelBenchProblem,
    load_relbench_problem,
    relbench_problem_from_objects,
)
from .search import (
    COMPACT_MODERN_FAMILY_TEMPLATES,
    WIDE_TARGET_HISTORY_TEMPLATES,
    FittedModel,
    FitResult,
    GraphConfig,
    SearchStageBudgets,
    Trial,
    campaign_modern_configs,
    compact_modern_configs,
    diverse_collection_configs,
    diverse_fidelity_trials,
    incremental_configs,
    published_compatible_configs,
    staged_challenger_configs,
    staged_modern_seed_configs,
    state_focused_configs,
    wide_modern_configs,
)
from .specs import GraphLabels, Labels, Relationship, Table
from .task_context import PredictionTask, SourceTable, TaskContext
from .staged import run_staged_task, StagedTaskResult
from .temporal_policy import (
    SelectedTemporalEstimator,
    TemporalFitPolicy,
    TemporalHoldoutSnapshot,
    select_and_fit_temporal_policy,
)

__version__ = "0.2.0rc1"

__all__ = [
    "COMPACT_MODERN_FAMILY_TEMPLATES",
    "WIDE_TARGET_HISTORY_TEMPLATES",
    "FitResult",
    "FittedModel",
    "SelectedTemporalEstimator",
    "TemporalFitPolicy",
    "TemporalHoldoutSnapshot",
    "select_and_fit_temporal_policy",
    "PLANNER_PROFILE_VERSION",
    "PLANNER_TRACE_VERSION",
    "CandidateObservation",
    "ColumnProfile",
    "ConfigFeatureEstimate",
    "DatabaseProfile",
    "EventFrequencyProfile",
    "FeaturePreflight",
    "GraphConfig",
    "GraphLabels",
    "Labels",
    "LearnedMessageFamily",
    "PooledMessageFamily",
    "PeerHistoryFamily",
    "PeerContext",
    "PredictionTask",
    "SourceTable",
    "TaskContext",
    "run_staged_task",
    "StagedTaskResult",
    "SemanticPayloadFamily",
    "PlannerDatasetSplit",
    "PlannerCollectionResult",
    "PlannerEpisode",
    "Relationship",
    "RelationshipProfile",
    "RelBenchProblem",
    "Table",
    "TableProfile",
    "TableFeatureEstimate",
    "TaskProfile",
    "Trial",
    "RuntimeProfile",
    "SearchStageBudgets",
    "configure_logging",
    "add_relbench_peer_context",
    "collect_planner_search",
    "campaign_modern_configs",
    "compact_modern_configs",
    "estimate_features",
    "diverse_collection_configs",
    "diverse_fidelity_trials",
    "episodes_from_fit_result",
    "fit",
    "predict",
    "incremental_configs",
    "published_compatible_configs",
    "load_relbench_problem",
    "read_episodes_jsonl",
    "relbench_problem_from_objects",
    "split_episodes_by_database",
    "state_focused_configs",
    "staged_challenger_configs",
    "staged_modern_seed_configs",
    "wide_modern_configs",
    "write_episodes_jsonl",
]
