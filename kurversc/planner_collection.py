"""Full-fidelity teacher-data collection for KurvePlanner."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import fit
from .planner_profiles import DatabaseProfile, RuntimeProfile, TaskProfile
from .planner_traces import (
    PlannerEpisode,
    episodes_from_fit_result,
    write_episodes_jsonl,
)
from .search import FitResult
from .specs import GraphLabels, Labels, Source, Table


@dataclass(frozen=True)
class PlannerCollectionResult:
    """A teacher search plus the portable episodes emitted from it."""

    fit_result: FitResult
    episodes: tuple[PlannerEpisode, ...]
    written_records: int = 0


def collect_planner_search(
    parent_node: Table | Source,
    label_node: Labels | GraphLabels | Source,
    *,
    database_profile: DatabaseProfile,
    task_profile: TaskProfile,
    runtime_profile: RuntimeProfile,
    trace_path: str | Path | None = None,
    append: bool = True,
    episode_key: str | None = None,
    **fit_kwargs: Any,
) -> PlannerCollectionResult:
    """Evaluate a diverse plan set at full fidelity and optionally append it.

    Metadata is supplied explicitly; this function never profiles the database.
    The ordinary table inputs are used only by GraphReduce to evaluate teacher
    candidates and are not serialized into the planner corpus.
    """

    if "planner_collection" in fit_kwargs:
        raise ValueError("planner_collection is managed by collect_planner_search")
    configured_backend = fit_kwargs.setdefault(
        "model_backend", runtime_profile.downstream_model
    )
    if configured_backend != runtime_profile.downstream_model:
        raise ValueError("RuntimeProfile.downstream_model must match fit model_backend")
    if runtime_profile.max_features is not None:
        fit_kwargs.setdefault("search_max_features", runtime_profile.max_features)
    if runtime_profile.max_working_cells is not None:
        fit_kwargs.setdefault(
            "search_max_working_cells", runtime_profile.max_working_cells
        )
    design_identity = "/".join(
        (
            database_profile.database_id,
            task_profile.task_id,
            episode_key or database_profile.as_of,
        )
    )
    fit_kwargs.setdefault(
        "planner_collection_random_state",
        int(hashlib.sha256(design_identity.encode()).hexdigest()[:8], 16),
    )

    result = fit(
        parent_node,
        label_node,
        planner_collection=True,
        **fit_kwargs,
    )
    episodes = episodes_from_fit_result(
        result,
        database_profile=database_profile,
        task_profile=task_profile,
        runtime_profile=runtime_profile,
        episode_key=episode_key,
        include_stages=("collection",),
    )
    written = (
        write_episodes_jsonl(trace_path, episodes, append=append)
        if trace_path is not None
        else 0
    )
    return PlannerCollectionResult(
        fit_result=result,
        episodes=episodes,
        written_records=written,
    )
