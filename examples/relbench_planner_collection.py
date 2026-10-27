"""Collect full-fidelity KurvePlanner teacher episodes from RelBench.

This is an explicit offline corpus-building utility.  It profiles public
RelBench frames before calling :func:`kurversc.collect_planner_search`; neither
``fit`` nor a future planner prediction call silently scans a database.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import kurversc


def _key_tuple(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _iso_timestamp(value: Any) -> str:
    return _utc_timestamp(value).isoformat()


def _dated_at_cutoff(
    frame: pd.DataFrame, date_key: str | None, cutoff: Any
) -> pd.DataFrame:
    if date_key is None or date_key not in frame:
        return frame
    dates = pd.to_datetime(frame[date_key], errors="coerce", utc=True)
    return frame.loc[dates <= _utc_timestamp(cutoff)]


def _sample_rows(
    frame: pd.DataFrame, limit: int | None, *, random_state: int
) -> pd.DataFrame:
    if limit is None or len(frame) <= limit:
        return frame
    return frame.sample(n=limit, random_state=random_state).sort_index()


def _hashable_values(series: pd.Series) -> pd.Series:
    try:
        series.nunique(dropna=True)
        return series
    except TypeError:
        return series.map(lambda value: repr(value) if value is not None else None)


def _physical_type(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_string_dtype(dtype):
        return "string"
    return str(dtype)


def _semantic_type(
    series: pd.Series,
    *,
    role: str | None,
    distinct_count: int,
    non_null_count: int,
) -> str | None:
    if role in {"primary_key", "foreign_key"}:
        return "identifier"
    if role == "date":
        return "timestamp"
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "timestamp"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        ratio = distinct_count / max(non_null_count, 1)
        return "categorical" if distinct_count <= 100 or ratio <= 0.05 else "text"
    return None


def _column_profile(
    name: str,
    series: pd.Series,
    *,
    role: str | None,
) -> kurversc.ColumnProfile:
    non_null = series.dropna()
    hashable = _hashable_values(non_null)
    distinct_count = int(hashable.nunique(dropna=True))
    non_null_count = int(len(non_null))
    counts = hashable.value_counts(normalize=True, dropna=True)
    entropy = None
    if distinct_count > 1 and not counts.empty:
        raw_entropy = float(-(counts * np.log(counts)).sum())
        entropy = min(1.0, max(0.0, raw_entropy / math.log(distinct_count)))
    numeric = (
        pd.to_numeric(non_null, errors="coerce").dropna()
        if pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        else pd.Series(dtype=float)
    )
    return kurversc.ColumnProfile(
        name=str(name),
        physical_type=_physical_type(series),
        semantic_type=_semantic_type(
            series,
            role=role,
            distinct_count=distinct_count,
            non_null_count=non_null_count,
        ),
        role=role,
        non_null_count=non_null_count,
        distinct_count=distinct_count,
        missing_fraction=(1.0 - non_null_count / len(series) if len(series) else None),
        uniqueness_fraction=(
            distinct_count / non_null_count if non_null_count else None
        ),
        normalized_entropy=entropy,
        mean=float(numeric.mean()) if not numeric.empty else None,
        standard_deviation=(float(numeric.std(ddof=0)) if not numeric.empty else None),
        minimum=float(numeric.min()) if not numeric.empty else None,
        maximum=float(numeric.max()) if not numeric.empty else None,
    )


def _table_profile(
    table_id: str,
    table: Any,
    *,
    cutoff: Any,
    profile_rows: int | None,
    random_state: int,
) -> kurversc.TableProfile:
    frame = _dated_at_cutoff(table.df, table.time_col, cutoff)
    sample = _sample_rows(frame, profile_rows, random_state=random_state)
    primary_key = _key_tuple(table.pkey_col)
    foreign_keys = set(table.fkey_col_to_pkey_table)
    columns = []
    for column in sample.columns:
        role = (
            "primary_key"
            if column in primary_key
            else "date"
            if column == table.time_col
            else "foreign_key"
            if column in foreign_keys
            else None
        )
        columns.append(_column_profile(str(column), sample[column], role=role))

    time_min = time_max = None
    time_span_seconds = None
    if table.time_col is not None and table.time_col in frame:
        dates = pd.to_datetime(
            frame[table.time_col], errors="coerce", utc=True
        ).dropna()
        if not dates.empty:
            minimum = dates.min()
            maximum = dates.max()
            time_min = minimum.isoformat()
            time_max = maximum.isoformat()
            time_span_seconds = float((maximum - minimum).total_seconds())

    role = "event" if table.time_col is not None else "entity"
    return kurversc.TableProfile(
        table_id=table_id,
        row_count=len(frame),
        columns=tuple(columns),
        primary_key=primary_key,
        date_key=table.time_col,
        role=role,
        time_min=time_min,
        time_max=time_max,
        time_span_seconds=time_span_seconds,
        sampled_rows=len(sample),
        sample_fraction=(len(sample) / len(frame) if len(frame) else None),
    )


def _keys_not_null(frame: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    return frame[list(keys)].notna().all(axis=1)


def _key_index(frame: pd.DataFrame, keys: Sequence[str]) -> pd.Index:
    if len(keys) == 1:
        return pd.Index(frame[keys[0]])
    return pd.MultiIndex.from_frame(frame[list(keys)])


def _sample_relationship_rows(
    frame: pd.DataFrame,
    keys: Sequence[str],
    limit: int | None,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, float | None]:
    valid = frame.loc[_keys_not_null(frame, keys)]
    if valid.empty:
        return valid, None
    if limit is None or len(valid) <= limit:
        return valid, 1.0
    sampled = valid.sample(n=limit, random_state=random_state).sort_index()
    return sampled, limit / len(valid)


def _event_frequency(
    frame: pd.DataFrame,
    *,
    keys: Sequence[str],
    date_key: str | None,
    sampled_rows: int,
    coverage_fraction: float | None,
) -> kurversc.EventFrequencyProfile | None:
    if date_key is None or date_key not in frame or frame.empty:
        return None
    dated = frame[list(keys) + [date_key]].copy()
    dated[date_key] = pd.to_datetime(dated[date_key], errors="coerce", utc=True)
    dated = dated.dropna(subset=[*keys, date_key]).sort_values([*keys, date_key])
    if dated.empty:
        return None
    grouped = dated.groupby(list(keys), dropna=False, sort=False)[date_key]
    counts = grouped.size().astype(float)
    durations = (grouped.max() - grouped.min()).dt.total_seconds().astype(float)
    intervals = grouped.diff().dt.total_seconds().dropna().astype(float)
    return kurversc.EventFrequencyProfile(
        num_dimensions_with_events=len(counts),
        avg_duration_seconds=float(durations.mean()),
        median_duration_seconds=float(durations.median()),
        avg_num_events=float(counts.mean()),
        median_num_events=float(counts.median()),
        avg_time_between_updates_seconds=(
            float(intervals.mean()) if not intervals.empty else None
        ),
        median_time_between_updates_seconds=(
            float(intervals.median()) if not intervals.empty else None
        ),
        p10_time_between_updates_seconds=(
            float(intervals.quantile(0.1)) if not intervals.empty else None
        ),
        p90_time_between_updates_seconds=(
            float(intervals.quantile(0.9)) if not intervals.empty else None
        ),
        sampled_rows=sampled_rows,
        coverage_fraction=coverage_fraction,
    )


def _relationship_profile(
    relationship_id: str,
    *,
    parent_name: str,
    parent: Any,
    parent_keys: Sequence[str],
    child_name: str,
    child: Any,
    child_keys: Sequence[str],
    cutoff: Any,
    profile_rows: int | None,
    random_state: int,
) -> kurversc.RelationshipProfile:
    parent_frame = _dated_at_cutoff(parent.df, parent.time_col, cutoff)
    child_frame = _dated_at_cutoff(child.df, child.time_col, cutoff)
    sampled_parent = _sample_rows(
        parent_frame,
        profile_rows,
        random_state=random_state + 1,
    )
    sampled_child_rows = _sample_rows(
        child_frame,
        profile_rows,
        random_state=random_state,
    )
    non_null_mask = _keys_not_null(sampled_child_rows, child_keys)
    non_null_child = sampled_child_rows.loc[non_null_mask]
    parent_values = _key_index(
        sampled_parent.loc[_keys_not_null(sampled_parent, parent_keys)], parent_keys
    ).unique()
    child_values = _key_index(non_null_child, child_keys)
    matched = child_values.isin(parent_values)
    parent_profile_is_complete = (
        profile_rows is None or len(parent_frame) <= profile_rows
    )
    orphan_fraction = (
        float(1.0 - matched.mean())
        if len(matched) and parent_profile_is_complete
        else None
    )
    sampled_child, row_coverage = _sample_relationship_rows(
        child_frame,
        child_keys,
        profile_rows,
        random_state=random_state,
    )
    fanout = sampled_child.groupby(list(child_keys), dropna=False).size().astype(float)
    child_is_unique = not non_null_child.duplicated(subset=list(child_keys)).any()
    parent_non_null = sampled_parent.loc[_keys_not_null(sampled_parent, parent_keys)]
    parent_is_unique = not parent_non_null.duplicated(subset=list(parent_keys)).any()
    cardinality = (
        "one_to_one"
        if parent_is_unique and child_is_unique
        else "one_to_many"
        if parent_is_unique
        else "many_to_one"
        if child_is_unique
        else "many_to_many"
    )
    return kurversc.RelationshipProfile(
        relationship_id=relationship_id,
        parent_table=parent_name,
        child_table=child_name,
        parent_key=tuple(parent_keys),
        child_key=tuple(child_keys),
        cardinality=cardinality,
        reduce=True,
        foreign_key_coverage=(
            float(non_null_mask.mean()) if len(non_null_mask) else None
        ),
        orphan_fraction=orphan_fraction,
        avg_fanout=float(fanout.mean()) if not fanout.empty else None,
        median_fanout=float(fanout.median()) if not fanout.empty else None,
        p90_fanout=float(fanout.quantile(0.9)) if not fanout.empty else None,
        max_fanout=float(fanout.max()) if not fanout.empty else None,
        join_expansion_ratio=None,
        event_frequency=_event_frequency(
            sampled_child,
            keys=child_keys,
            date_key=child.time_col,
            sampled_rows=len(sampled_child),
            coverage_fraction=row_coverage,
        ),
        context_key_count=max(0, len(child.fkey_col_to_pkey_table) - 1),
    )


def profile_relbench_database(
    database: Any,
    *,
    database_id: str,
    cutoff: Any,
    profile_rows: int | None = 250_000,
    random_state: int = 42,
) -> kurversc.DatabaseProfile:
    """Build a cutoff-safe, value-free profile from an explicit DB object."""

    table_profiles = tuple(
        _table_profile(
            table_name,
            table,
            cutoff=cutoff,
            profile_rows=profile_rows,
            random_state=random_state + index,
        )
        for index, (table_name, table) in enumerate(database.table_dict.items())
    )
    relationships = []
    for child_index, (child_name, child) in enumerate(database.table_dict.items()):
        for key_index, (child_key, parent_name) in enumerate(
            child.fkey_col_to_pkey_table.items()
        ):
            parent = database.table_dict[parent_name]
            parent_keys = _key_tuple(parent.pkey_col)
            if not parent_keys:
                continue
            child_keys = _key_tuple(child_key)
            relationships.append(
                _relationship_profile(
                    f"{parent_name}:{','.join(parent_keys)}<-"
                    f"{child_name}:{','.join(child_keys)}",
                    parent_name=parent_name,
                    parent=parent,
                    parent_keys=parent_keys,
                    child_name=child_name,
                    child=child,
                    child_keys=child_keys,
                    cutoff=cutoff,
                    profile_rows=profile_rows,
                    random_state=random_state + child_index * 101 + key_index,
                )
            )
    return kurversc.DatabaseProfile(
        database_id=database_id,
        as_of=_iso_timestamp(cutoff),
        extractor="relbench-explicit-profiler",
        tables=table_profiles,
        relationships=tuple(relationships),
    )


def profile_relbench_task(
    task: Any,
    *,
    task_id: str,
    cutoff: Any,
) -> kurversc.TaskProfile:
    """Build task-head metadata from the training support available at cutoff."""

    frame = task.get_table("train").df.copy()
    times = pd.to_datetime(frame[task.time_col], errors="coerce", utc=True)
    frame = frame.loc[times <= _utc_timestamp(cutoff)]
    target = frame[task.target_col].dropna()
    task_type_raw = str(task.task_type).lower()
    task_type = "classification" if "classification" in task_type_raw else "regression"
    metric = "roc_auc" if task_type == "classification" else "mae"
    class_count = int(target.nunique()) if task_type == "classification" else None
    positive_fraction = None
    if task_type == "classification" and class_count == 2:
        encoded = pd.Series(pd.factorize(target, sort=True)[0], index=target.index)
        positive_fraction = float(encoded.mean())
    repeated = None
    if not frame.empty:
        repeated = float(frame.groupby(task.entity_col, dropna=False).size().mean())
    horizon = getattr(task, "timedelta", None)
    return kurversc.TaskProfile(
        task_id=task_id,
        task_type=task_type,
        metric=metric,
        root_table=task.entity_table,
        target_id=task.target_col,
        support_rows=len(frame),
        horizon_seconds=(
            float(pd.Timedelta(horizon).total_seconds())
            if horizon is not None
            else None
        ),
        support_start=(
            _iso_timestamp(frame[task.time_col].min()) if not frame.empty else None
        ),
        support_end=(
            _iso_timestamp(frame[task.time_col].max()) if not frame.empty else None
        ),
        class_count=class_count,
        positive_fraction=positive_fraction,
        target_mean=(float(target.mean()) if task_type == "regression" else None),
        target_standard_deviation=(
            float(target.std(ddof=0)) if task_type == "regression" else None
        ),
        repeated_observations_per_entity=repeated,
    )


def _available_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _memory_limit_bytes(value: str) -> int:
    normalized = value.strip().upper().replace("IB", "B")
    units = {
        "KB": 1_000,
        "MB": 1_000_000,
        "GB": 1_000_000_000,
        "TB": 1_000_000_000_000,
        "B": 1,
    }
    for suffix in ("TB", "GB", "MB", "KB", "B"):
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)].strip()
            return int(float(number) * units[suffix])
    return int(float(normalized))


def _discover_tasks(dataset_name: str) -> list[str]:
    from relbench.base import EntityTask, TaskType
    from relbench.tasks import task_registry

    selected = []
    for task_name, entry in task_registry.get(dataset_name, {}).items():
        task_class = entry[0] if isinstance(entry, tuple) else entry
        kwargs = entry[2] if isinstance(entry, tuple) and len(entry) > 2 else {}
        task_type = getattr(task_class, "task_type", None) or kwargs.get("task_type")
        if (
            isinstance(task_class, type)
            and issubclass(task_class, EntityTask)
            and task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION}
        ):
            selected.append(task_name)
    return sorted(selected)


def _write_profile(path: Path, profile: kurversc.DatabaseProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect full-fidelity RelBench teacher traces for KurvePlanner"
    )
    parser.add_argument("dataset", help="RelBench dataset name, e.g. rel-f1")
    parser.add_argument(
        "tasks", nargs="*", help="Task names; default: all binary/regression tasks"
    )
    parser.add_argument("--output", default="planner-traces", help="Output directory")
    parser.add_argument("--profile-rows", type=int, default=250_000)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--training-frames", type=int, default=3)
    parser.add_argument("--column-budgets", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--max-train-timestamps", type=int, default=15)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--schema-depth", type=int, default=4)
    parser.add_argument("--max-base-table-occurrences", type=int, default=1)
    parser.add_argument("--duckdb-memory-limit", default="128GB")
    parser.add_argument("--duckdb-temp-limit", default="128GB")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.profile_rows < 1 or args.candidates < 2 or args.training_frames < 1:
        raise ValueError(
            "profile rows, candidates and training frames must be positive"
        )
    task_names = args.tasks or _discover_tasks(args.dataset)
    if not task_names:
        raise ValueError(
            f"No binary classification/regression tasks for {args.dataset}"
        )

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    kurversc.configure_logging(colors=False)
    configured_memory = _memory_limit_bytes(args.duckdb_memory_limit)
    available_memory = _available_memory_bytes()
    runtime = kurversc.RuntimeProfile(
        downstream_model="catboost",
        execution_backend="duckdb",
        cpu_count=os.cpu_count(),
        memory_bytes=(
            min(configured_memory, available_memory)
            if available_memory is not None
            else configured_memory
        ),
        max_features=8_000,
    )
    profile_cache: dict[str, kurversc.DatabaseProfile] = {}
    outputs = []

    for task_name in task_names:
        trace_path = destination / f"{args.dataset}__{task_name}.jsonl"
        if trace_path.exists() and not args.overwrite:
            print(f"skip existing {trace_path}", flush=True)
            continue
        problem = kurversc.load_relbench_problem(
            args.dataset,
            task_name,
            sample_rows=50_000,
            # Keep only the adapter's path-discovery sources bounded. Teacher
            # collection overrides fit to use every table's authoritative
            # ``source`` rather than these sampled ``search_source`` frames.
            search_full_data=False,
            search_training_frames=args.training_frames,
            max_train_timestamps=args.max_train_timestamps,
            schema_depth=args.schema_depth,
            max_base_table_occurrences=args.max_base_table_occurrences,
            random_state=args.random_state,
            download=not args.no_download,
        )
        cutoff = max(problem.train_timestamps)
        cutoff_key = _iso_timestamp(cutoff)
        profile = profile_cache.get(cutoff_key)
        if profile is None:
            profile = profile_relbench_database(
                problem.task.dataset.get_db(),
                database_id=args.dataset,
                cutoff=cutoff,
                profile_rows=args.profile_rows,
                random_state=args.random_state,
            )
            profile_cache[cutoff_key] = profile
            profile_name = cutoff_key.replace(":", "-")
            _write_profile(
                destination / "profiles" / f"{args.dataset}__{profile_name}.json",
                profile,
            )
        task_profile = profile_relbench_task(
            problem.task,
            task_id=task_name,
            cutoff=cutoff,
        )
        fit_kwargs = problem.fit_kwargs()
        parent_node = fit_kwargs.pop("parent_node")
        label_node = fit_kwargs.pop("label_node")
        print(
            f"collect {args.dataset}/{task_name} cutoff={cutoff_key} "
            f"candidates={args.candidates}",
            flush=True,
        )
        collected = kurversc.collect_planner_search(
            parent_node,
            label_node,
            database_profile=profile,
            task_profile=task_profile,
            runtime_profile=runtime,
            trace_path=trace_path,
            append=False,
            episode_key=cutoff_key,
            planner_collection_candidates=args.candidates,
            planner_collection_training_frames=args.training_frames,
            planner_collection_column_budgets=tuple(args.column_budgets),
            max_depth=args.max_depth,
            duckdb_memory_limit=args.duckdb_memory_limit,
            duckdb_max_temp_directory_size=args.duckdb_temp_limit,
            random_state=args.random_state,
            **fit_kwargs,
        )
        winner = collected.fit_result.best_trial
        summary = {
            "dataset": args.dataset,
            "task": task_name,
            "trace": str(trace_path),
            "candidates": len(collected.fit_result.trials),
            "winner": winner.as_record(),
        }
        (trace_path.with_suffix(".summary.json")).write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        outputs.append(summary)
        print(json.dumps(summary, sort_keys=True, default=str), flush=True)

    print(json.dumps({"completed": outputs}, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
