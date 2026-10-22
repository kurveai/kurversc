"""Official RelBench metadata adapter for KurveRSC."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Literal

import pandas as pd

from .specs import Labels, Relationship, Table


@dataclass(frozen=True)
class RelBenchProblem:
    """A production RelBench task translated into declarative KurveRSC inputs."""

    dataset_name: str
    task_name: str
    parent_node: Table
    label_node: Labels
    tables: tuple[Table, ...]
    relationships: tuple[Relationship, ...]
    timestamp_column: str
    train_timestamps: tuple[pd.Timestamp, ...]
    validation_timestamps: tuple[pd.Timestamp, ...]
    test_timestamps: tuple[pd.Timestamp, ...]
    task: Any
    peer_context: Any = None

    def fit_kwargs(self) -> dict[str, Any]:
        task_type = str(self.task.task_type).lower()
        return {
            "parent_node": self.parent_node,
            "label_node": self.label_node,
            "tables": self.tables,
            "relationships": self.relationships,
            "task": (
                "multiclass_classification"
                if "multiclass" in task_type
                else "classification"
                if "classification" in task_type
                else "regression"
            ),
            "class_labels": (
                tuple(self.task.target_encoder.categories_[0])
                if "multiclass" in task_type
                else None
            ),
        }


def add_relbench_peer_context(
    problem: RelBenchProblem,
    train_table: Any,
    *,
    additional_history_table: Any | None = None,
    mode: str = "both",
    max_history_rows: int = 50_000,
) -> RelBenchProblem:
    """Attach a reusable, leak-safe peer-label provider to a RelBench task.

    The inner arm passes train labels only. The outer arm may also pass the
    observed validation labels, which are available before masked test rows.
    Search and prediction use the same provider; no peer ID is a model column.
    """

    if problem.peer_context is not None or problem.label_node.peer_context is not None:
        raise ValueError("Peer context is already attached")
    from .core import _normalize_tables
    from .peer_context import PeerContext

    task = problem.task
    required = [task.entity_col, task.time_col, task.target_col]
    frames = [
        (value.df if hasattr(value, "df") else value)[required].copy()
        for value in (train_table, additional_history_table)
        if value is not None
    ]
    history = pd.concat(frames, ignore_index=True)
    history[task.time_col] = pd.to_datetime(history[task.time_col])
    availability = getattr(task, "label_available_at", None)
    if availability:
        # A portable task adapter may declare real observation times rather
        # than assuming that every label arrives exactly at horizon end.
        observed = pd.concat([(value.df if hasattr(value,"df") else value)[availability]
            for value in (train_table,additional_history_table) if value is not None],ignore_index=True)
        history["__peer_label_available_at"] = pd.to_datetime(observed).to_numpy()
    else:
        history["__peer_label_available_at"] = history[task.time_col] + task.timedelta
    root, tables = _normalize_tables(problem.parent_node, problem.tables)
    provider = PeerContext(mode=mode, max_history_rows=max_history_rows).fit(
        history,
        tables=tables,
        relationships=problem.relationships,
        root_name=root,
        entity_key=task.entity_col,
        timestamp=problem.label_node.timestamp,
        target=task.target_col,
        available_at="__peer_label_available_at",
    )
    return replace(
        problem,
        label_node=replace(problem.label_node, peer_context=provider),
        peer_context=provider,
    )


def _sample_timestamps(
    timestamps: list[pd.Timestamp], limit: int, *, seed: int
) -> list[pd.Timestamp]:
    """Mirror the Stack benchmark's stratified timestamp selection."""

    if limit < 1:
        raise ValueError("max_train_timestamps must be positive")
    if len(timestamps) <= limit:
        return timestamps
    if limit == 1:
        return [timestamps[-1]]
    rng = random.Random(seed)
    historical_count = len(timestamps) - 1
    sample_count = limit - 1
    selected = []
    for index in range(sample_count):
        start = index * historical_count // sample_count
        stop = (index + 1) * historical_count // sample_count
        selected.append(timestamps[rng.randrange(start, stop)])
    selected.append(timestamps[-1])
    return selected


def _recently_spaced_timestamps(
    timestamps: list[pd.Timestamp],
    limit: int,
    *,
    window_days: int,
) -> list[pd.Timestamp]:
    """Pick evenly spaced cutoffs from a recent window, always including latest."""
    if window_days < 1:
        raise ValueError("recent_train_window_days must be positive")
    if limit < 1:
        raise ValueError("max_train_timestamps must be positive")
    if len(timestamps) <= limit:
        return timestamps

    anchor = timestamps[-1]
    lower_bound = anchor - pd.Timedelta(days=window_days)
    candidates = [timestamp for timestamp in timestamps if timestamp >= lower_bound]
    if len(candidates) <= limit:
        return candidates
    if limit == 1:
        return [anchor]

    span = len(candidates) - 1
    intervals = limit - 1
    indices = [(index * span + intervals // 2) // intervals for index in range(limit)]
    return [candidates[index] for index in indices]


def _stratified_rows(
    frame: pd.DataFrame,
    *,
    target: str,
    limit: int,
    random_state: int,
) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.copy()
    from sklearn.model_selection import train_test_split

    target_counts = frame[target].value_counts(dropna=True)
    class_count = len(target_counts)
    can_stratify = (
        class_count > 1
        and target_counts.min() >= 2
        and limit >= class_count
        and len(frame) - limit >= class_count
    )
    selected, _ = train_test_split(
        frame,
        train_size=limit,
        random_state=random_state,
        stratify=frame[target] if can_stratify else None,
    )
    return selected.sort_index().reset_index(drop=True)


def _enrichment_columns(
    table: Any,
    limit: int | None,
    *,
    high_cardinality_text: Literal["skip", "fallback", "short"] = "skip",
) -> tuple[str, ...] | None:
    """Select a bounded, deterministic set of dimension attributes.

    Enrichment joins intentionally exclude keys from the budget because the
    graph builder adds required PK/FK/date columns separately. High-cardinality
    text is omitted by default. ``fallback`` retains short dictionary
    descriptors only for otherwise empty dimensions; ``short`` also admits
    them alongside compact attributes. Long narrative text remains excluded.
    """

    if high_cardinality_text not in {"skip", "fallback", "short"}:
        raise ValueError("high_cardinality_text must be skip, fallback, or short")
    if limit is None:
        return None
    frame = table.df
    reserved = {
        value
        for value in (
            table.pkey_col,
            table.time_col,
            getattr(table,"available_col",None),
            *table.fkey_col_to_pkey_table.keys(),
        )
        if value is not None
    }
    sample = frame.head(10_000)
    ranked: list[tuple[int, int, str]] = []
    descriptor_fallbacks: list[tuple[int, int, str]] = []
    for ordinal, column in enumerate(frame.columns):
        if column in reserved:
            continue
        values = sample[column].dropna()
        if values.empty:
            continue
        cardinality_values = values
        if pd.api.types.is_object_dtype(values.dtype):
            cardinality_values = values.map(_hashable_cardinality_value)
        cardinality = int(cardinality_values.nunique())
        if cardinality <= 1:
            continue
        ratio = cardinality / len(values)
        dtype = values.dtype
        if pd.api.types.is_bool_dtype(dtype):
            priority = 0
        elif pd.api.types.is_numeric_dtype(dtype):
            priority = 0 if cardinality <= 1_000 and ratio <= 0.5 else 2
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            priority = 1
        elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            lengths = values.astype(str).str.len()
            if lengths.mean() > 80:
                continue
            if cardinality > 1_000 and ratio > 0.25:
                if high_cardinality_text == "skip":
                    continue
                descriptor_fallbacks.append((2, ordinal, str(column)))
                continue
            else:
                priority = 0 if cardinality <= 1_000 else 1
        else:
            continue
        ranked.append((priority, ordinal, str(column)))
    if high_cardinality_text == "short" or (
        high_cardinality_text == "fallback" and not ranked
    ):
        ranked.extend(descriptor_fallbacks)
    ranked.sort()
    return tuple(column for _priority, _ordinal, column in ranked[:limit])


def _hashable_cardinality_value(value: Any) -> Any:
    """Normalize nested RelBench cells for deterministic cardinality checks."""
    try:
        hash(value)
        return value
    except TypeError:
        converted = value.tolist() if hasattr(value, "tolist") else value
        return repr(converted)


def _edge_instances(
    database: Any,
    *,
    root_table: str,
    root_frame: pd.DataFrame,
    search_root_frame: pd.DataFrame,
    fidelity_root_frame: pd.DataFrame | None = None,
    schema_depth: int,
    sample_rows: int | None,
    random_state: int,
    max_enrichment_columns: int | None = 8,
    high_cardinality_dimension_text: Literal["skip", "fallback", "short"] = "skip",
    max_base_table_occurrences: int = 1,
) -> tuple[Table, tuple[Table, ...], tuple[Relationship, ...]]:
    """Expand the schema into a bounded, path-unrolled feature graph.

    RelBench schemas contain both one-to-many edges (for example,
    ``customer -> transactions``) and association-table edges to referenced
    dimensions (``transactions -> article``). The latter are intentionally
    represented as feature edges in the direction of aggregation, even though
    the physical foreign key points in the opposite direction.

    Repeated base tables are represented by distinct path instances, preserving
    a DAG even when the relational walk revisits an entity type.  The occurrence
    cap bounds those unrolled cyclic/self-relational walks: one is the legacy
    acyclic behavior; two admits paths such as ``user -> friendship -> user``.
    """

    table_dict = database.table_dict
    root_meta = table_dict[root_table]
    root = Table(
        root_frame,
        search_source=search_root_frame,
        fidelity_source=fidelity_root_frame,
        name=root_table,
        key=root_meta.pkey_col,
        date=root_meta.time_col,
        timeless=root_meta.time_col is None,
        prefix=root_table[:10],
        path_base_tables=(root_table,),
        available_at=getattr(root_meta,"available_col",None),
        cutoff_boundary=getattr(root_meta,"cutoff_boundary",None),
        model_exclusions=tuple(root_meta.fkey_col_to_pkey_table)
            if getattr(database,"exclude_structural_aliases",False) else (),
    )
    tables: list[Table] = []
    relationships: list[Relationship] = []
    # Instance name, base table, cheap-search frame, fidelity frame,
    # base-table path, graph depth.
    queue = [
        (
            root_table,
            root_table,
            search_root_frame,
            fidelity_root_frame,
            (root_table,),
            0,
        )
    ]
    used_names = {root_table}
    while queue:
        (
            parent_instance,
            parent_base,
            parent_search_frame,
            parent_fidelity_frame,
            path,
            depth,
        ) = queue.pop(0)
        if depth >= schema_depth:
            continue
        parent_meta = table_dict[parent_base]
        parent_pk = parent_meta.pkey_col
        candidates: list[
            tuple[str, Any, str, str, str | None, bool, tuple[str, ...] | None]
        ] = []

        # Normal reverse-FK expansion: parent -> rows that reference parent.
        if parent_pk is not None:
            # RelBench populates table_dict from directory enumeration, whose
            # order can differ across hosts.  Path numbering and the sampling
            # seed below depend on this traversal order, so canonicalize it.
            for child_base, child_meta in sorted(table_dict.items()):
                for (
                    foreign_key,
                    referenced_table,
                ) in sorted(child_meta.fkey_col_to_pkey_table.items()):
                    if referenced_table == parent_base:
                        candidates.append(
                            (
                                child_base,
                                child_meta,
                                parent_pk,
                                foreign_key,
                                foreign_key,
                                True,
                                None,
                            )
                        )

        # Association/dimension expansion: a row's FK identifies a related
        # parent table.  Treat that related table as a feature child so its
        # attributes can be reduced back through the association table.
        for foreign_key, referenced_table in sorted(
            parent_meta.fkey_col_to_pkey_table.items()
        ):
            related_meta = table_dict[referenced_table]
            if related_meta.pkey_col is not None:
                candidates.append(
                    (
                        referenced_table,
                        related_meta,
                        foreign_key,
                        related_meta.pkey_col,
                        None,
                        False,
                        _enrichment_columns(
                            related_meta,
                            max_enrichment_columns,
                            high_cardinality_text=high_cardinality_dimension_text,
                        ),
                    )
                )

        for (
            child_base,
            child_meta,
            parent_key,
            child_key,
            relation_fk,
            reduce,
            columns,
        ) in candidates:
            if path.count(child_base) >= max_base_table_occurrences:
                continue
            # Every walk occurrence receives its own node instance, so the
            # unrolled relation graph remains a DAG even for cyclic schemas.
            base_name = f"{parent_instance}__{child_base}__{child_key}"
            instance = base_name
            suffix = 2
            while instance in used_names:
                instance = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(instance)
            parent_values = parent_search_frame[parent_key].dropna()
            child_search_frame = child_meta.df.loc[
                child_meta.df[child_key].isin(parent_values)
            ]
            if sample_rows is not None and len(child_search_frame) > sample_rows:
                child_search_frame = child_search_frame.sample(
                    n=sample_rows,
                    random_state=random_state + len(tables),
                ).sort_index()
            child_fidelity_frame = None
            if parent_fidelity_frame is not None:
                fidelity_parent_values = parent_fidelity_frame[parent_key].dropna()
                child_fidelity_frame = child_meta.df.loc[
                    child_meta.df[child_key].isin(fidelity_parent_values)
                ]
                if sample_rows is not None and len(child_fidelity_frame) > sample_rows:
                    child_fidelity_frame = child_fidelity_frame.sample(
                        n=sample_rows,
                        random_state=random_state + len(tables),
                    ).sort_index()
                child_fidelity_frame = child_fidelity_frame.reset_index(drop=True)
            child = Table(
                child_meta.df,
                search_source=child_search_frame.reset_index(drop=True),
                fidelity_source=child_fidelity_frame,
                name=instance,
                key=child_meta.pkey_col,
                date=child_meta.time_col,
                timeless=child_meta.time_col is None,
                prefix=f"r{len(tables):02d}",
                columns=columns,
                context_keys=tuple(
                    key
                    for key in sorted(child_meta.fkey_col_to_pkey_table)
                    if key != relation_fk
                ),
                path_base_tables=(*path, child_base),
                available_at=getattr(child_meta,"available_col",None),
                cutoff_boundary=getattr(child_meta,"cutoff_boundary",None),
                model_exclusions=tuple(c for c in (child_meta.pkey_col,*child_meta.fkey_col_to_pkey_table) if c)
                    if getattr(database,"exclude_structural_aliases",False) else (),
            )
            tables.append(child)
            relationships.append(
                Relationship(
                    parent=parent_instance,
                    child=instance,
                    parent_key=parent_key,
                    child_key=child_key,
                    reduce=reduce,
                )
            )
            queue.append(
                (
                    instance,
                    child_base,
                    child_search_frame,
                    child_fidelity_frame,
                    (*path, child_base),
                    depth + 1,
                )
            )
    return root, tuple(tables), tuple(relationships)


def relbench_problem_from_objects(
    task: Any,
    database: Any,
    train_table: Any,
    validation_table: Any,
    *,
    test_table: Any | None = None,
    dataset_name: str = "relbench",
    task_name: str | None = None,
    sample_rows: int = 100_000,
    search_full_data: bool = False,
    search_training_frames: int = 1,
    fidelity_training_frames: int | None = None,
    max_train_timestamps: int = 15,
    training_timestamp_policy: Literal["stratified", "recent"] = "stratified",
    recent_train_window_days: int | None = None,
    schema_depth: int = 4,
    max_enrichment_columns: int | None = 8,
    high_cardinality_dimension_text: Literal["skip", "fallback", "short"] = "skip",
    max_base_table_occurrences: int = 1,
    include_target_history: bool = True,
    random_state: int = 42,
) -> RelBenchProblem:
    """Translate already-censored RelBench objects into KurveRSC inputs.

    Full sources remain untouched for the production phase. Separate connected
    and stratified search sources are attached to the declarative table specs.
    """

    if sample_rows < 2:
        raise ValueError("sample_rows must be at least 2")
    if max_train_timestamps < 1:
        raise ValueError("max_train_timestamps must be positive")
    if search_training_frames < 1:
        raise ValueError("search_training_frames must be positive")
    if fidelity_training_frames is None:
        fidelity_training_frames = search_training_frames
    if fidelity_training_frames < 1:
        raise ValueError("fidelity_training_frames must be positive or None")
    if schema_depth < 1:
        raise ValueError("schema_depth must be at least 1")
    if max_base_table_occurrences < 1:
        raise ValueError("max_base_table_occurrences must be at least 1")
    if max_enrichment_columns is not None and max_enrichment_columns < 1:
        raise ValueError("max_enrichment_columns must be positive or None")

    def frame(value: Any) -> pd.DataFrame:
        source = value.df if hasattr(value, "df") else value
        return source.copy()

    train = frame(train_table)
    validation = frame(validation_table)
    test = frame(test_table) if test_table is not None else None
    # Salt autocomplete rows are timestamped independently, often with only a
    # few labels at any exact second. KurveRSC trains on cutoff frames, so use
    # the following midnight as a daily bucket. A start-of-day cutoff would
    # exclude entities created later that same day and empty every frame.
    timestamp_column = task.time_col
    if "multiclass" in str(task.task_type).lower():
        timestamp_column = "__kurversc_relbench_day"
    for split_frame in (train, validation, test):
        if split_frame is not None:
            split_frame[task.time_col] = pd.to_datetime(split_frame[task.time_col])
            if timestamp_column != task.time_col:
                split_frame[timestamp_column] = split_frame[
                    task.time_col
                ].dt.normalize() + pd.Timedelta(days=1)
    split_column = "__kurversc_relbench_split__"
    history_train = train.copy()
    history_validation = validation.copy()
    history_train[split_column] = "train"
    history_validation[split_column] = "validation"
    history_labels = pd.concat([history_train, history_validation], ignore_index=True)
    all_train_timestamps = sorted(
        pd.Timestamp(value) for value in train[timestamp_column].dropna().unique()
    )
    if training_timestamp_policy not in {"stratified", "recent"}:
        raise ValueError("training_timestamp_policy must be 'stratified' or 'recent'")
    if training_timestamp_policy == "recent" and recent_train_window_days is not None:
        raise ValueError("Use recent training timestamps or a recent window, not both")
    if training_timestamp_policy == "recent":
        selected_train_timestamps = all_train_timestamps[-max_train_timestamps:]
    elif recent_train_window_days is None:
        selected_train_timestamps = _sample_timestamps(
            all_train_timestamps, max_train_timestamps, seed=random_state
        )
    else:
        selected_train_timestamps = _recently_spaced_timestamps(
            all_train_timestamps,
            max_train_timestamps,
            window_days=recent_train_window_days,
        )
    train = train.loc[train[timestamp_column].isin(selected_train_timestamps)].copy()
    validation_timestamps = tuple(
        sorted(
            pd.Timestamp(value)
            for value in validation[timestamp_column].dropna().unique()
        )
    )
    test_timestamps = (
        tuple(
            sorted(
                pd.Timestamp(value)
                for value in test[timestamp_column].dropna().unique()
            )
        )
        if test is not None
        else ()
    )

    train_limit = max(1, sample_rows // 2)
    validation_limit = max(1, sample_rows - train_limit)

    def task_connected_labels(
        selected_timestamps: list[pd.Timestamp],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build one deterministic, cutoff-balanced label cohort."""

        train_source = train.loc[
            train[timestamp_column].isin(selected_timestamps)
        ].copy()
        if search_full_data:
            return train_source, validation.copy()
        per_cutoff_limit, extra_rows = divmod(train_limit, len(selected_timestamps))
        sampled_frames = []
        for index, cutoff in enumerate(selected_timestamps):
            cutoff_frame = train_source.loc[train_source[timestamp_column] == cutoff]
            sampled_frames.append(
                _stratified_rows(
                    cutoff_frame,
                    target=task.target_col,
                    limit=max(1, per_cutoff_limit + (index < extra_rows)),
                    random_state=random_state + index,
                )
            )
        return (
            pd.concat(sampled_frames, ignore_index=True),
            _stratified_rows(
                validation,
                target=task.target_col,
                limit=validation_limit,
                random_state=random_state + 1,
            ),
        )

    # The cheap source owns feature-plan discovery. The fidelity source exposes
    # additional cutoffs only after that plan is frozen, so increasing temporal
    # evidence cannot silently change the candidate being compared.
    selected_search_timestamps = _sample_timestamps(
        selected_train_timestamps,
        search_training_frames,
        seed=random_state,
    )
    selected_fidelity_timestamps = _sample_timestamps(
        selected_train_timestamps,
        fidelity_training_frames,
        seed=random_state,
    )
    search_train, search_validation = task_connected_labels(selected_search_timestamps)
    fidelity_train, fidelity_validation = task_connected_labels(
        selected_fidelity_timestamps
    )
    search_sample_rows = None if search_full_data else sample_rows
    train[split_column] = "train"
    validation[split_column] = "validation"
    search_train[split_column] = "train"
    search_validation[split_column] = "validation"
    fidelity_train[split_column] = "train"
    fidelity_validation[split_column] = "validation"
    full_parts = [train, validation]
    if test is not None:
        test[split_column] = "test"
        full_parts.append(test)
    labels = pd.concat(full_parts, ignore_index=True)
    search_labels = pd.concat([search_train, search_validation], ignore_index=True)
    fidelity_labels = pd.concat(
        [fidelity_train, fidelity_validation], ignore_index=True
    )

    root_meta = database.table_dict[task.entity_table]
    entity_values = search_labels[task.entity_col].dropna()
    search_root_frame = root_meta.df.loc[
        root_meta.df[root_meta.pkey_col].isin(entity_values)
    ]
    fidelity_entity_values = fidelity_labels[task.entity_col].dropna()
    fidelity_root_frame = root_meta.df.loc[
        root_meta.df[root_meta.pkey_col].isin(fidelity_entity_values)
    ]
    parent, tables, relationships = _edge_instances(
        database,
        root_table=task.entity_table,
        root_frame=root_meta.df,
        search_root_frame=search_root_frame,
        fidelity_root_frame=fidelity_root_frame,
        schema_depth=schema_depth,
        sample_rows=search_sample_rows,
        random_state=random_state,
        max_enrichment_columns=max_enrichment_columns,
        high_cardinality_dimension_text=high_cardinality_dimension_text,
        max_base_table_occurrences=max_base_table_occurrences,
    )
    label_spec = Labels(
        labels,
        search_source=search_labels,
        fidelity_source=fidelity_labels,
        history_source=history_labels,
        target_history=include_target_history,
        key=task.entity_col,
        target=task.target_col,
        timestamp=timestamp_column,
        split=split_column,
        available_at=getattr(task,"label_available_at",None),
    )
    return RelBenchProblem(
        dataset_name=dataset_name,
        task_name=task_name or type(task).__name__,
        parent_node=parent,
        label_node=label_spec,
        tables=tables,
        relationships=relationships,
        timestamp_column=timestamp_column,
        train_timestamps=tuple(selected_train_timestamps),
        validation_timestamps=validation_timestamps,
        test_timestamps=test_timestamps,
        task=task,
    )


def load_relbench_problem(
    dataset_name: str,
    task_name: str,
    *,
    sample_rows: int = 100_000,
    search_full_data: bool = False,
    search_training_frames: int = 1,
    fidelity_training_frames: int | None = None,
    max_train_timestamps: int = 15,
    training_timestamp_policy: Literal["stratified", "recent"] = "stratified",
    schema_depth: int = 4,
    max_enrichment_columns: int | None = 8,
    high_cardinality_dimension_text: Literal["skip", "fallback", "short"] = "skip",
    max_base_table_occurrences: int = 1,
    include_target_history: bool = True,
    random_state: int = 42,
    download: bool = True,
) -> RelBenchProblem:
    """Load an official RelBench task and derive its graph from DB metadata.

    Labels come only from RelBench's production task tables. Date keys,
    primary keys, and foreign keys come only from its production database
    metadata. No feature expressions or per-task feature policies are added.
    """

    if sample_rows < 2:
        raise ValueError("sample_rows must be at least 2")
    if search_training_frames < 1:
        raise ValueError("search_training_frames must be positive")
    if schema_depth < 1:
        raise ValueError("schema_depth must be at least 1")
    if max_base_table_occurrences < 1:
        raise ValueError("max_base_table_occurrences must be at least 1")
    if max_enrichment_columns is not None and max_enrichment_columns < 1:
        raise ValueError("max_enrichment_columns must be positive or None")
    try:
        from relbench.datasets import get_dataset
        from relbench.tasks import get_task
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "RelBench adapters require the optional dependency; install "
            "`kurversc[relbench]`"
        ) from exc

    dataset = get_dataset(dataset_name, download=download)
    database = dataset.get_db()
    task = get_task(dataset_name, task_name, download=download)
    return relbench_problem_from_objects(
        task,
        database,
        task.get_table("train"),
        task.get_table("val"),
        test_table=task.get_table("test"),
        dataset_name=dataset_name,
        task_name=task_name,
        sample_rows=sample_rows,
        search_full_data=search_full_data,
        search_training_frames=search_training_frames,
        fidelity_training_frames=fidelity_training_frames,
        max_train_timestamps=max_train_timestamps,
        training_timestamp_policy=training_timestamp_policy,
        schema_depth=schema_depth,
        max_enrichment_columns=max_enrichment_columns,
        high_cardinality_dimension_text=high_cardinality_dimension_text,
        max_base_table_occurrences=max_base_table_occurrences,
        include_target_history=include_target_history,
        random_state=random_state,
    )
