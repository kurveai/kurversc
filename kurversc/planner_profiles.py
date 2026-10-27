"""Versioned, cutoff-aware metadata contracts for KurvePlanner.

The planner consumes aggregate metadata produced by Kurve or another explicit
profiler.  It does not inspect raw database tables.  Keeping this contract in
the open-source package lets product profilers and research training pipelines
exchange the same privacy-conscious representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from math import isfinite, log1p
from typing import Any, Mapping, Sequence


PLANNER_PROFILE_VERSION = "1.0"
Scalar = str | int | float | bool | None


def _validate_fraction(value: float | None, name: str) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between zero and one")


def _validate_nonnegative(value: int | float | None, name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_iso_timestamp(value: str, name: str) -> None:
    if not value:
        raise ValueError(f"{name} must not be empty")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc


def _copy_scalars(values: Mapping[str, Scalar], name: str) -> dict[str, Scalar]:
    copied = {str(key): value for key, value in values.items()}
    invalid = [
        key
        for key, value in copied.items()
        if value is not None and not isinstance(value, (str, int, float, bool))
    ]
    if invalid:
        raise TypeError(f"{name} values must be JSON scalar values: {invalid[:3]}")
    return copied


def _finite(values: Sequence[int | float | None]) -> list[float]:
    return [
        float(value) for value in values if value is not None and isfinite(float(value))
    ]


def _summary(
    output: dict[str, Scalar], prefix: str, values: Sequence[int | float | None]
) -> None:
    clean = _finite(values)
    output[f"{prefix}_available"] = len(clean)
    if not clean:
        return
    output[f"{prefix}_mean"] = sum(clean) / len(clean)
    output[f"{prefix}_min"] = min(clean)
    output[f"{prefix}_max"] = max(clean)


@dataclass(frozen=True)
class EventFrequencyProfile:
    """Aggregate event-sequence statistics for one dated relationship."""

    num_dimensions_with_events: int | None = None
    avg_duration_seconds: float | None = None
    median_duration_seconds: float | None = None
    avg_num_events: float | None = None
    median_num_events: float | None = None
    avg_time_between_updates_seconds: float | None = None
    median_time_between_updates_seconds: float | None = None
    p10_time_between_updates_seconds: float | None = None
    p90_time_between_updates_seconds: float | None = None
    sampled_rows: int | None = None
    coverage_fraction: float | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "coverage_fraction":
                _validate_fraction(value, name)
            else:
                _validate_nonnegative(value, name)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EventFrequencyProfile":
        return cls(**dict(value))

    @classmethod
    def from_kurve_payload(cls, value: Mapping[str, Any]) -> "EventFrequencyProfile":
        """Read the aggregate fields from Kurve's persisted edge payload."""

        accepted = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in accepted})


@dataclass(frozen=True)
class ColumnProfile:
    """Aggregate, value-free description of a database column."""

    name: str
    physical_type: str
    semantic_type: str | None = None
    role: str | None = None
    non_null_count: int | None = None
    distinct_count: int | None = None
    missing_fraction: float | None = None
    uniqueness_fraction: float | None = None
    normalized_entropy: float | None = None
    mean: float | None = None
    standard_deviation: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.physical_type:
            raise ValueError("ColumnProfile name and physical_type must not be empty")
        _validate_nonnegative(self.non_null_count, "non_null_count")
        _validate_nonnegative(self.distinct_count, "distinct_count")
        _validate_nonnegative(self.standard_deviation, "standard_deviation")
        _validate_fraction(self.missing_fraction, "missing_fraction")
        _validate_fraction(self.uniqueness_fraction, "uniqueness_fraction")
        _validate_fraction(self.normalized_entropy, "normalized_entropy")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ColumnProfile":
        return cls(**dict(value))


@dataclass(frozen=True)
class TableProfile:
    """Aggregate metadata for one table in the relational schema."""

    table_id: str
    row_count: int
    columns: tuple[ColumnProfile, ...] = ()
    primary_key: tuple[str, ...] = ()
    date_key: str | None = None
    role: str | None = None
    time_min: str | None = None
    time_max: str | None = None
    time_span_seconds: float | None = None
    sampled_rows: int | None = None
    sample_fraction: float | None = None
    extras: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.table_id:
            raise ValueError("TableProfile.table_id must not be empty")
        _validate_nonnegative(self.row_count, "row_count")
        _validate_nonnegative(self.time_span_seconds, "time_span_seconds")
        _validate_nonnegative(self.sampled_rows, "sampled_rows")
        _validate_fraction(self.sample_fraction, "sample_fraction")
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "primary_key", tuple(self.primary_key))
        object.__setattr__(
            self, "extras", _copy_scalars(self.extras, "TableProfile.extras")
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TableProfile":
        payload = dict(value)
        payload["columns"] = tuple(
            ColumnProfile.from_dict(item) for item in payload.get("columns", ())
        )
        payload["primary_key"] = tuple(payload.get("primary_key", ()))
        return cls(**payload)


@dataclass(frozen=True)
class RelationshipProfile:
    """Aggregate metadata for one directed relational edge."""

    relationship_id: str
    parent_table: str
    child_table: str
    parent_key: tuple[str, ...]
    child_key: tuple[str, ...]
    cardinality: str | None = None
    reduce: bool = True
    foreign_key_coverage: float | None = None
    orphan_fraction: float | None = None
    avg_fanout: float | None = None
    median_fanout: float | None = None
    p90_fanout: float | None = None
    max_fanout: float | None = None
    join_expansion_ratio: float | None = None
    event_frequency: EventFrequencyProfile | None = None
    context_key_count: int = 0
    extras: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.relationship_id or not self.parent_table or not self.child_table:
            raise ValueError("relationship and table identifiers must not be empty")
        object.__setattr__(self, "parent_key", tuple(self.parent_key))
        object.__setattr__(self, "child_key", tuple(self.child_key))
        if not self.parent_key or len(self.parent_key) != len(self.child_key):
            raise ValueError("parent_key and child_key must have equal positive width")
        _validate_fraction(self.foreign_key_coverage, "foreign_key_coverage")
        _validate_fraction(self.orphan_fraction, "orphan_fraction")
        for name in (
            "avg_fanout",
            "median_fanout",
            "p90_fanout",
            "max_fanout",
            "join_expansion_ratio",
            "context_key_count",
        ):
            _validate_nonnegative(getattr(self, name), name)
        object.__setattr__(
            self,
            "extras",
            _copy_scalars(self.extras, "RelationshipProfile.extras"),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RelationshipProfile":
        payload = dict(value)
        payload["parent_key"] = tuple(payload.get("parent_key", ()))
        payload["child_key"] = tuple(payload.get("child_key", ()))
        event = payload.get("event_frequency")
        if event is not None and not isinstance(event, EventFrequencyProfile):
            payload["event_frequency"] = EventFrequencyProfile.from_dict(event)
        return cls(**payload)


@dataclass(frozen=True)
class DatabaseProfile:
    """Complete planner input describing a database as of one safe cutoff."""

    database_id: str
    as_of: str
    tables: tuple[TableProfile, ...]
    relationships: tuple[RelationshipProfile, ...]
    profile_version: str = PLANNER_PROFILE_VERSION
    generated_at: str | None = None
    extractor: str = "external"
    extras: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.database_id:
            raise ValueError("DatabaseProfile.database_id must not be empty")
        if self.profile_version != PLANNER_PROFILE_VERSION:
            raise ValueError(
                f"Unsupported profile version {self.profile_version!r}; "
                f"expected {PLANNER_PROFILE_VERSION!r}"
            )
        _validate_iso_timestamp(self.as_of, "DatabaseProfile.as_of")
        if self.generated_at is not None:
            _validate_iso_timestamp(self.generated_at, "DatabaseProfile.generated_at")
        object.__setattr__(self, "tables", tuple(self.tables))
        object.__setattr__(self, "relationships", tuple(self.relationships))
        table_ids = [table.table_id for table in self.tables]
        if len(set(table_ids)) != len(table_ids):
            raise ValueError("DatabaseProfile table identifiers must be unique")
        known = set(table_ids)
        for relationship in self.relationships:
            if (
                relationship.parent_table not in known
                or relationship.child_table not in known
            ):
                raise ValueError(
                    f"Relationship {relationship.relationship_id!r} references "
                    "an unknown table"
                )
        object.__setattr__(
            self, "extras", _copy_scalars(self.extras, "DatabaseProfile.extras")
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatabaseProfile":
        payload = dict(value)
        payload["tables"] = tuple(
            TableProfile.from_dict(item) for item in payload.get("tables", ())
        )
        payload["relationships"] = tuple(
            RelationshipProfile.from_dict(item)
            for item in payload.get("relationships", ())
        )
        return cls(**payload)

    def aggregate_features(self) -> dict[str, Scalar]:
        """Return a compact baseline encoding; graph models use full structure."""

        output: dict[str, Scalar] = {
            "schema_table_count": len(self.tables),
            "schema_relationship_count": len(self.relationships),
            "schema_total_rows_log1p": log1p(
                sum(table.row_count for table in self.tables)
            ),
            "schema_max_table_rows_log1p": log1p(
                max((table.row_count for table in self.tables), default=0)
            ),
            "schema_column_count": sum(len(table.columns) for table in self.tables),
            "schema_dated_table_count": sum(
                table.date_key is not None for table in self.tables
            ),
            "schema_self_relationship_count": sum(
                edge.parent_table == edge.child_table for edge in self.relationships
            ),
            "schema_context_edge_count": sum(
                edge.context_key_count > 0 for edge in self.relationships
            ),
            "schema_profile_extractor": self.extractor,
        }
        roles: dict[str, int] = {}
        physical_types: dict[str, int] = {}
        for table in self.tables:
            if table.role:
                roles[table.role] = roles.get(table.role, 0) + 1
            for column in table.columns:
                key = column.semantic_type or column.physical_type
                physical_types[key] = physical_types.get(key, 0) + 1
        for role, count in roles.items():
            output[f"schema_table_role__{role}"] = count
        for physical_type, count in physical_types.items():
            output[f"schema_column_type__{physical_type}"] = count

        cardinalities: dict[str, int] = {}
        for edge in self.relationships:
            key = edge.cardinality or "unknown"
            cardinalities[key] = cardinalities.get(key, 0) + 1
        for cardinality, count in cardinalities.items():
            output[f"schema_cardinality__{cardinality}"] = count

        _summary(
            output,
            "schema_time_span_seconds",
            [table.time_span_seconds for table in self.tables],
        )
        _summary(
            output,
            "schema_missing_fraction",
            [
                column.missing_fraction
                for table in self.tables
                for column in table.columns
            ],
        )
        _summary(
            output,
            "schema_distinct_count_log1p",
            [
                log1p(column.distinct_count)
                for table in self.tables
                for column in table.columns
                if column.distinct_count is not None
            ],
        )
        _summary(
            output,
            "schema_fanout",
            [edge.avg_fanout for edge in self.relationships],
        )
        frequencies = [
            edge.event_frequency
            for edge in self.relationships
            if edge.event_frequency is not None
        ]
        output["schema_event_frequency_edge_count"] = len(frequencies)
        _summary(
            output,
            "schema_avg_num_events",
            [frequency.avg_num_events for frequency in frequencies],
        )
        _summary(
            output,
            "schema_inter_event_seconds",
            [frequency.avg_time_between_updates_seconds for frequency in frequencies],
        )
        for key, value in self.extras.items():
            output[f"schema_extra__{key}"] = value
        return output


@dataclass(frozen=True)
class TaskProfile:
    """Label-head and support-set metadata visible to the planner."""

    task_id: str
    task_type: str
    metric: str
    root_table: str
    target_id: str
    support_rows: int
    horizon_seconds: float | None = None
    support_start: str | None = None
    support_end: str | None = None
    class_count: int | None = None
    positive_fraction: float | None = None
    target_mean: float | None = None
    target_standard_deviation: float | None = None
    repeated_observations_per_entity: float | None = None
    probes: Mapping[str, float] = field(default_factory=dict)
    extras: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("task_id", "task_type", "metric", "root_table", "target_id"):
            if not getattr(self, name):
                raise ValueError(f"TaskProfile.{name} must not be empty")
        _validate_nonnegative(self.support_rows, "support_rows")
        _validate_nonnegative(self.horizon_seconds, "horizon_seconds")
        _validate_nonnegative(self.class_count, "class_count")
        _validate_fraction(self.positive_fraction, "positive_fraction")
        _validate_nonnegative(
            self.target_standard_deviation, "target_standard_deviation"
        )
        _validate_nonnegative(
            self.repeated_observations_per_entity,
            "repeated_observations_per_entity",
        )
        if self.support_start is not None:
            _validate_iso_timestamp(self.support_start, "support_start")
        if self.support_end is not None:
            _validate_iso_timestamp(self.support_end, "support_end")
        object.__setattr__(
            self,
            "probes",
            {str(key): float(value) for key, value in self.probes.items()},
        )
        object.__setattr__(
            self, "extras", _copy_scalars(self.extras, "TaskProfile.extras")
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskProfile":
        return cls(**dict(value))

    def aggregate_features(self) -> dict[str, Scalar]:
        output: dict[str, Scalar] = {
            "task_type": self.task_type,
            "task_metric": self.metric,
            "task_support_rows_log1p": log1p(self.support_rows),
            "task_horizon_seconds": self.horizon_seconds,
            "task_class_count": self.class_count,
            "task_positive_fraction": self.positive_fraction,
            "task_target_mean": self.target_mean,
            "task_target_standard_deviation": self.target_standard_deviation,
            "task_repeated_observations_per_entity": (
                self.repeated_observations_per_entity
            ),
        }
        for key, value in self.probes.items():
            output[f"task_probe__{key}"] = value
        for key, value in self.extras.items():
            output[f"task_extra__{key}"] = value
        return output


@dataclass(frozen=True)
class RuntimeProfile:
    """Execution constraints that can change which plan is preferable."""

    downstream_model: str
    execution_backend: str = "duckdb"
    cpu_count: int | None = None
    memory_bytes: int | None = None
    max_features: int | None = None
    max_working_cells: int | None = None
    latency_budget_seconds: float | None = None
    extras: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.downstream_model or not self.execution_backend:
            raise ValueError("runtime model and backend must not be empty")
        for name in (
            "cpu_count",
            "memory_bytes",
            "max_features",
            "max_working_cells",
            "latency_budget_seconds",
        ):
            _validate_nonnegative(getattr(self, name), name)
        object.__setattr__(
            self, "extras", _copy_scalars(self.extras, "RuntimeProfile.extras")
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeProfile":
        return cls(**dict(value))

    def aggregate_features(self) -> dict[str, Scalar]:
        output: dict[str, Scalar] = {
            "runtime_downstream_model": self.downstream_model,
            "runtime_execution_backend": self.execution_backend,
            "runtime_cpu_count": self.cpu_count,
            "runtime_memory_bytes_log1p": (
                log1p(self.memory_bytes) if self.memory_bytes is not None else None
            ),
            "runtime_max_features": self.max_features,
            "runtime_max_working_cells_log1p": (
                log1p(self.max_working_cells)
                if self.max_working_cells is not None
                else None
            ),
            "runtime_latency_budget_seconds": self.latency_budget_seconds,
        }
        for key, value in self.extras.items():
            output[f"runtime_extra__{key}"] = value
        return output
