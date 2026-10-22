"""Search-trace records used to train and evaluate KurvePlanner."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence

from .planner_profiles import DatabaseProfile, RuntimeProfile, Scalar, TaskProfile
from .search import FitResult, Trial


PLANNER_TRACE_VERSION = "1.0"
_STAGE_WEIGHTS = {
    "screening": 0.25,
    "confirmation": 0.6,
    "rerank": 1.0,
    "collection": 1.0,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CandidateObservation:
    """One candidate plan and its measured teacher outcome."""

    candidate_id: str
    plan: Mapping[str, Any]
    status: str
    supervision: str
    metric: str
    validation_score: float | None
    objective_score: float | None
    validation_standard_error: float | None = None
    feature_count: int | None = None
    estimated_feature_count: int | None = None
    train_rows: int | None = None
    validation_rows: int | None = None
    feature_seconds: float | None = None
    model_seconds: float | None = None
    peak_memory_bytes: int | None = None
    error_type: str | None = None
    is_search_best: bool = False
    is_search_recommended: bool = False
    cutoff_labels: tuple[str, ...] = ()
    cutoff_scores: tuple[float | None, ...] = ()
    extras: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.metric or not self.status:
            raise ValueError("candidate_id, metric, and status must not be empty")
        if self.supervision not in {"observed", "infeasible", "censored"}:
            raise ValueError(
                "CandidateObservation.supervision must be 'observed', "
                "'infeasible', or 'censored'"
            )
        object.__setattr__(self, "plan", _json_safe(dict(self.plan)))
        object.__setattr__(self, "cutoff_labels", tuple(self.cutoff_labels))
        object.__setattr__(self, "cutoff_scores", tuple(self.cutoff_scores))
        if len(self.cutoff_labels) != len(self.cutoff_scores):
            raise ValueError("cutoff_labels and cutoff_scores must have equal length")
        object.__setattr__(self, "extras", dict(self.extras))

    @property
    def completed(self) -> bool:
        return self.status == "completed" and self.objective_score is not None

    @property
    def supervised(self) -> bool:
        return self.supervision != "censored"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateObservation":
        return cls(**dict(value))

    @classmethod
    def from_trial(
        cls,
        trial: Trial,
        *,
        is_search_best: bool = False,
        is_search_recommended: bool = False,
        plan_overrides: Mapping[str, Any] | None = None,
    ) -> "CandidateObservation":
        plan = asdict(trial.config)
        plan.update(dict(plan_overrides or {}))
        execution_fingerprint = (
            _fingerprint(trial.execution_plan)
            if trial.execution_plan is not None
            else None
        )
        error_type = trial.error.split(":", 1)[0] if trial.error else None
        error = trial.error or ""
        if trial.status in {"completed", "failed"}:
            supervision = "observed"
        elif "exceeds search_max_" in error or "Resource superset" in error:
            supervision = "infeasible"
        else:
            supervision = "censored"
        objective = (
            float(trial.objective_score)
            if math.isfinite(trial.objective_score)
            else None
        )
        validation = (
            float(trial.validation_score)
            if math.isfinite(trial.validation_score)
            else None
        )
        return cls(
            candidate_id=_fingerprint(plan)[:20],
            plan=plan,
            status=trial.status,
            supervision=supervision,
            metric=trial.metric,
            validation_score=validation,
            objective_score=objective,
            validation_standard_error=trial.validation_standard_error,
            feature_count=trial.feature_count,
            estimated_feature_count=trial.estimated_feature_count,
            train_rows=trial.train_rows,
            validation_rows=trial.validation_rows,
            feature_seconds=trial.feature_seconds,
            model_seconds=trial.model_seconds,
            error_type=error_type,
            is_search_best=is_search_best,
            is_search_recommended=is_search_recommended,
            cutoff_labels=trial.cutoff_labels,
            cutoff_scores=trial.cutoff_scores,
            extras={"execution_plan_fingerprint": execution_fingerprint},
        )


@dataclass(frozen=True)
class PlannerEpisode:
    """A candidate-ranking episode for one database, task, cutoff, and fidelity."""

    episode_id: str
    database_profile: DatabaseProfile
    task_profile: TaskProfile
    runtime_profile: RuntimeProfile
    observations: tuple[CandidateObservation, ...]
    stage: str = "screening"
    stage_weight: float = 0.25
    trace_version: str = PLANNER_TRACE_VERSION
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("PlannerEpisode.episode_id must not be empty")
        if self.trace_version != PLANNER_TRACE_VERSION:
            raise ValueError(
                f"Unsupported trace version {self.trace_version!r}; "
                f"expected {PLANNER_TRACE_VERSION!r}"
            )
        if self.stage_weight <= 0:
            raise ValueError("PlannerEpisode.stage_weight must be positive")
        object.__setattr__(self, "observations", tuple(self.observations))
        if not self.observations:
            raise ValueError("PlannerEpisode requires at least one observation")
        candidate_ids = [item.candidate_id for item in self.observations]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("PlannerEpisode candidate identifiers must be unique")
        mismatched_metrics = {
            item.metric
            for item in self.observations
            if item.metric != self.task_profile.metric
        }
        if mismatched_metrics:
            raise ValueError(
                "Candidate metrics do not match TaskProfile.metric: "
                f"{sorted(mismatched_metrics)}"
            )
        object.__setattr__(self, "extras", dict(self.extras))

    @property
    def database_id(self) -> str:
        return self.database_profile.database_id

    @property
    def completed_observations(self) -> tuple[CandidateObservation, ...]:
        return tuple(item for item in self.observations if item.completed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_version": self.trace_version,
            "episode_id": self.episode_id,
            "database_profile": self.database_profile.to_dict(),
            "task_profile": self.task_profile.to_dict(),
            "runtime_profile": self.runtime_profile.to_dict(),
            "observations": [item.to_dict() for item in self.observations],
            "stage": self.stage,
            "stage_weight": self.stage_weight,
            "extras": _json_safe(self.extras),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlannerEpisode":
        payload = dict(value)
        payload["database_profile"] = DatabaseProfile.from_dict(
            payload["database_profile"]
        )
        payload["task_profile"] = TaskProfile.from_dict(payload["task_profile"])
        payload["runtime_profile"] = RuntimeProfile.from_dict(
            payload["runtime_profile"]
        )
        payload["observations"] = tuple(
            CandidateObservation.from_dict(item)
            for item in payload.get("observations", ())
        )
        return cls(**payload)


def _column_selection_trace(result: FitResult) -> dict[str, Any]:
    """Return pointer-head supervision from the feature-funnel audit.

    This stays in episode extras so the current candidate ranker does not turn
    schema-specific table or column names into fixed features. A future graph
    encoder can instead align these names with ``DatabaseProfile.columns`` and
    learn a within-node pointer/ranking objective.
    """

    audit = result.feature_audit
    if audit.empty:
        return {}
    fields = {
        "table",
        "column",
        "role",
        "eligible",
        "structural",
        "utility_score",
        "utility_rank",
        "configured_position",
        "selection_rank",
        "column_selection_strategy",
        "eligible_families",
        "family_ranks",
        "within_family_budget",
        "within_expanded_family_budget",
        "eligible_budget_tiers",
        "reason",
    }
    available = [column for column in audit.columns if column in fields]
    records = audit.loc[:, available].to_dict(orient="records")
    strategies = tuple(
        dict.fromkeys(
            str(value)
            for value in audit.get("column_selection_strategy", ())
            if value is not None
        )
    )
    return _json_safe(
        {
            "column_selection_strategy": (
                strategies[0] if len(strategies) == 1 else "mixed"
            ),
            "column_selection": records,
        }
    )


def episodes_from_fit_result(
    result: FitResult,
    *,
    database_profile: DatabaseProfile,
    task_profile: TaskProfile,
    runtime_profile: RuntimeProfile,
    episode_key: str | None = None,
    include_stages: Sequence[str] = (
        "collection",
        "screening",
        "confirmation",
        "rerank",
    ),
) -> tuple[PlannerEpisode, ...]:
    """Convert a real KurveRSC search result into stage-aware teacher traces."""

    if task_profile.root_table not in {
        table.table_id for table in database_profile.tables
    }:
        raise ValueError("TaskProfile.root_table is absent from DatabaseProfile")
    if result.metric and task_profile.metric != result.metric:
        raise ValueError(
            f"Task metric {task_profile.metric!r} does not match result "
            f"metric {result.metric!r}"
        )
    is_collection = bool(result.trials) and all(
        trial.stage == "collection" for trial in result.trials
    )
    stage_trials = {
        "collection": tuple(result.trials) if is_collection else (),
        "screening": () if is_collection else tuple(result.trials),
        "confirmation": tuple(result.confirmation_trials),
        "rerank": tuple(result.rerank_trials),
    }
    unknown = set(include_stages) - set(stage_trials)
    if unknown:
        raise ValueError(f"Unknown trace stages: {sorted(unknown)}")
    key = episode_key or database_profile.as_of
    column_selection = _column_selection_trace(result)
    episodes = []
    for stage in include_stages:
        trials = stage_trials[stage]
        if not trials:
            continue
        observations = tuple(
            CandidateObservation.from_trial(
                trial,
                is_search_best=trial.config == result.best_config,
                is_search_recommended=trial.config == result.recommended_config,
            )
            for trial in trials
        )
        episodes.append(
            PlannerEpisode(
                episode_id=(
                    f"{database_profile.database_id}/{task_profile.task_id}/"
                    f"{key}/{stage}"
                ),
                database_profile=database_profile,
                task_profile=task_profile,
                runtime_profile=runtime_profile,
                observations=observations,
                stage=stage,
                stage_weight=_STAGE_WEIGHTS[stage],
                extras=column_selection,
            )
        )
    return tuple(episodes)


def write_episodes_jsonl(
    path: str | Path,
    episodes: Iterable[PlannerEpisode],
    *,
    append: bool = False,
) -> int:
    """Persist planner traces as portable JSON Lines records."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = [
        json.dumps(episode.to_dict(), sort_keys=True, separators=(",", ":"))
        for episode in episodes
    ]
    if append:
        with destination.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(record + "\n")
        return len(records)

    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        for record in records:
            stream.write(record + "\n")
    temporary.replace(destination)
    return len(records)


def read_episodes_jsonl(
    paths: str | Path | Sequence[str | Path],
) -> tuple[PlannerEpisode, ...]:
    """Read and validate one or more planner trace files."""

    selected = (paths,) if isinstance(paths, (str, Path)) else tuple(paths)
    episodes = []
    seen = set()
    for path in selected:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    episode = PlannerEpisode.from_dict(json.loads(line))
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(
                        f"Invalid planner trace {path}:{line_number}"
                    ) from exc
                if episode.episode_id in seen:
                    raise ValueError(f"Duplicate planner episode: {episode.episode_id}")
                seen.add(episode.episode_id)
                episodes.append(episode)
    return tuple(episodes)


@dataclass(frozen=True)
class PlannerDatasetSplit:
    """Database-disjoint training, validation, and final-test episodes."""

    train: tuple[PlannerEpisode, ...]
    validation: tuple[PlannerEpisode, ...]
    test: tuple[PlannerEpisode, ...]
    train_databases: tuple[str, ...]
    validation_databases: tuple[str, ...]
    test_databases: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_databases": list(self.train_databases),
            "validation_databases": list(self.validation_databases),
            "test_databases": list(self.test_databases),
            "train_episodes": len(self.train),
            "validation_episodes": len(self.validation),
            "test_episodes": len(self.test),
        }


def split_episodes_by_database(
    episodes: Sequence[PlannerEpisode],
    *,
    train_fraction: float = 0.8,
    validation_size: int | None = None,
    test_size: int = 2,
    validation_databases: Sequence[str] | None = None,
    test_databases: Sequence[str] | None = None,
    required_test_task_types: Sequence[str] = (),
    random_state: int = 42,
) -> PlannerDatasetSplit:
    """Create a deterministic split without leaking any database across sets."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    database_ids = sorted({episode.database_id for episode in episodes})
    if len(database_ids) < 4:
        raise ValueError("At least four databases are required for grouped splitting")
    known = set(database_ids)
    explicit_test = tuple(dict.fromkeys(test_databases or ()))
    explicit_validation = tuple(dict.fromkeys(validation_databases or ()))
    unknown = (set(explicit_test) | set(explicit_validation)) - known
    if unknown:
        raise ValueError(f"Unknown databases in requested split: {sorted(unknown)}")
    overlap = set(explicit_test) & set(explicit_validation)
    if overlap:
        raise ValueError(f"Validation and test databases overlap: {sorted(overlap)}")

    rng = random.Random(random_state)
    available = [
        value
        for value in database_ids
        if value not in explicit_test and value not in explicit_validation
    ]
    rng.shuffle(available)
    required_types = set(required_test_task_types)

    def task_types(selected: Sequence[str]) -> set[str]:
        selected_set = set(selected)
        return {
            episode.task_profile.task_type
            for episode in episodes
            if episode.database_id in selected_set
        }

    if explicit_test:
        selected_test = explicit_test
        missing_types = required_types - task_types(selected_test)
        if missing_types:
            raise ValueError(
                "Final-test databases do not cover required task types: "
                f"{sorted(missing_types)}"
            )
    else:
        if test_size < 1:
            raise ValueError("test_size must be positive")
        if test_size > len(available):
            raise ValueError("test_size exceeds the available database count")
        selected_test = next(
            (
                tuple(candidate)
                for candidate in combinations(available, test_size)
                if required_types.issubset(task_types(candidate))
            ),
            None,
        )
        if selected_test is None:
            raise ValueError(
                "No final-test database combination covers required task types: "
                f"{sorted(required_types)}"
            )
        selected_test_set = set(selected_test)
        available = [value for value in available if value not in selected_test_set]

    if explicit_validation:
        selected_validation = explicit_validation
    else:
        if validation_size is None:
            target_train = int(len(database_ids) * train_fraction)
            validation_size = max(
                1, len(database_ids) - len(selected_test) - target_train
            )
        if validation_size < 1:
            raise ValueError("validation_size must be positive")
        selected_validation = tuple(available[:validation_size])
        available = available[validation_size:]

    reserved = set(selected_test) | set(selected_validation)
    selected_train = tuple(value for value in database_ids if value not in reserved)
    if not selected_train or not selected_validation or not selected_test:
        raise ValueError("Grouped split produced an empty partition")

    def select(database_set: set[str]) -> tuple[PlannerEpisode, ...]:
        return tuple(
            episode for episode in episodes if episode.database_id in database_set
        )

    return PlannerDatasetSplit(
        train=select(set(selected_train)),
        validation=select(set(selected_validation)),
        test=select(set(selected_test)),
        train_databases=tuple(sorted(selected_train)),
        validation_databases=tuple(sorted(selected_validation)),
        test_databases=tuple(sorted(selected_test)),
    )
