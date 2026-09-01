"""Official RelBench metadata adapter for KurveRSC."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

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
    train_timestamps: tuple[pd.Timestamp, ...]
    validation_timestamps: tuple[pd.Timestamp, ...]
    test_timestamps: tuple[pd.Timestamp, ...]
    task: Any

    def fit_kwargs(self) -> dict[str, Any]:
        return {
            "parent_node": self.parent_node,
            "label_node": self.label_node,
            "tables": self.tables,
            "relationships": self.relationships,
            "task": (
                "classification"
                if "classification" in str(self.task.task_type).lower()
                else "regression"
            ),
        }


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


def _enrichment_columns(table: Any, limit: int | None) -> tuple[str, ...] | None:
    """Select a bounded, deterministic set of dimension attributes.

    Enrichment joins intentionally exclude keys from the budget because the
    graph builder adds required PK/FK/date columns separately. High-cardinality
    free text is omitted by default; compact categorical and numeric attributes
    are preferred because they remain useful after reduction through an event
    or association table.
    """

    if limit is None:
        return None
    frame = table.df
    reserved = {
        value
        for value in (
            table.pkey_col,
            table.time_col,
            *table.fkey_col_to_pkey_table.keys(),
        )
        if value is not None
    }
    sample = frame.head(10_000)
    ranked: list[tuple[int, int, str]] = []
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
            if (cardinality > 1_000 and ratio > 0.25) or lengths.mean() > 80:
                continue
            priority = 0 if cardinality <= 1_000 else 1
        else:
            continue
        ranked.append((priority, ordinal, str(column)))
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
    schema_depth: int,
    sample_rows: int | None,
    random_state: int,
    max_enrichment_columns: int | None = 8,
) -> tuple[Table, tuple[Table, ...], tuple[Relationship, ...]]:
    """Expand the schema into an acyclic, path-specific feature graph.

    RelBench schemas contain both one-to-many edges (for example,
    ``customer -> transactions``) and association-table edges to referenced
    dimensions (``transactions -> article``).  The latter are intentionally
    represented as feature edges in the direction of aggregation, even though
    the physical foreign key points in the opposite direction.
    """

    table_dict = database.table_dict
    root_meta = table_dict[root_table]
    root = Table(
        root_frame,
        search_source=search_root_frame,
        name=root_table,
        key=root_meta.pkey_col,
        date=root_meta.time_col,
        timeless=root_meta.time_col is None,
        prefix=root_table[:10],
    )
    tables: list[Table] = []
    relationships: list[Relationship] = []
    # instance name, base table, sampled frame, base-table path, graph depth
    queue = [(root_table, root_table, search_root_frame, (root_table,), 0)]
    used_names = {root_table}
    while queue:
        parent_instance, parent_base, parent_frame, path, depth = queue.pop(0)
        if depth >= schema_depth:
            continue
        parent_meta = table_dict[parent_base]
        parent_pk = parent_meta.pkey_col
        candidates: list[
            tuple[str, Any, str, str, str | None, bool, tuple[str, ...] | None]
        ] = []

        # Normal reverse-FK expansion: parent -> rows that reference parent.
        if parent_pk is not None:
            for child_base, child_meta in table_dict.items():
                for (
                    foreign_key,
                    referenced_table,
                ) in child_meta.fkey_col_to_pkey_table.items():
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
        for foreign_key, referenced_table in parent_meta.fkey_col_to_pkey_table.items():
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
                        _enrichment_columns(related_meta, max_enrichment_columns),
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
            if child_base in path:
                continue
            # Self/cyclic paths cannot be represented safely by GraphReduce's
            # DiGraph. Other foreign-key paths are each materialized once.
            base_name = f"{parent_instance}__{child_base}__{child_key}"
            instance = base_name
            suffix = 2
            while instance in used_names:
                instance = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(instance)
            parent_values = parent_frame[parent_key].dropna()
            child_search_frame = child_meta.df.loc[
                child_meta.df[child_key].isin(parent_values)
            ]
            if sample_rows is not None and len(child_search_frame) > sample_rows:
                child_search_frame = child_search_frame.sample(
                    n=sample_rows,
                    random_state=random_state + len(tables),
                ).sort_index()
            child = Table(
                child_meta.df,
                search_source=child_search_frame.reset_index(drop=True),
                name=instance,
                key=child_meta.pkey_col,
                date=child_meta.time_col,
                timeless=child_meta.time_col is None,
                prefix=f"r{len(tables):02d}",
                columns=columns,
                context_keys=tuple(
                    key
                    for key in child_meta.fkey_col_to_pkey_table
                    if key != relation_fk
                ),
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
    max_train_timestamps: int = 15,
    schema_depth: int = 3,
    max_enrichment_columns: int | None = 8,
    random_state: int = 42,
) -> RelBenchProblem:
    """Translate already-censored RelBench objects into KurveRSC inputs.

    Full sources remain untouched for the production phase. Separate connected
    and stratified search sources are attached to the declarative table specs.
    """

    if sample_rows < 2:
        raise ValueError("sample_rows must be at least 2")
    if search_training_frames < 1:
        raise ValueError("search_training_frames must be positive")
    if schema_depth < 1:
        raise ValueError("schema_depth must be at least 1")
    if max_enrichment_columns is not None and max_enrichment_columns < 1:
        raise ValueError("max_enrichment_columns must be positive or None")

    def frame(value: Any) -> pd.DataFrame:
        source = value.df if hasattr(value, "df") else value
        return source.copy()

    train = frame(train_table)
    validation = frame(validation_table)
    test = frame(test_table) if test_table is not None else None
    for split_frame in (train, validation, test):
        if split_frame is not None:
            split_frame[task.time_col] = pd.to_datetime(split_frame[task.time_col])
    all_train_timestamps = sorted(
        pd.Timestamp(value) for value in train[task.time_col].dropna().unique()
    )
    selected_train_timestamps = _sample_timestamps(
        all_train_timestamps, max_train_timestamps, seed=random_state
    )
    train = train.loc[train[task.time_col].isin(selected_train_timestamps)].copy()
    validation_timestamps = tuple(
        sorted(
            pd.Timestamp(value) for value in validation[task.time_col].dropna().unique()
        )
    )
    test_timestamps = (
        tuple(
            sorted(
                pd.Timestamp(value) for value in test[task.time_col].dropna().unique()
            )
        )
        if test is not None
        else ()
    )

    train_limit = max(1, sample_rows // 2)
    validation_limit = max(1, sample_rows - train_limit)
    # Preserve the requested, evenly spaced training cutoffs in the search
    # label source. KurveRSC can then materialize and retain each cutoff frame
    # for a joint downstream fit. The default remains the latest cutoff only.
    selected_search_timestamps = _sample_timestamps(
        selected_train_timestamps,
        search_training_frames,
        seed=random_state,
    )
    search_train_source = train.loc[
        train[task.time_col].isin(selected_search_timestamps)
    ].copy()
    if search_full_data:
        search_train = search_train_source.copy()
        search_validation = validation.copy()
        search_sample_rows = None
    else:
        per_cutoff_limit, extra_rows = divmod(
            train_limit, len(selected_search_timestamps)
        )
        sampled_search_frames = []
        for index, cutoff in enumerate(selected_search_timestamps):
            cutoff_frame = search_train_source.loc[
                search_train_source[task.time_col] == cutoff
            ]
            sampled_search_frames.append(
                _stratified_rows(
                    cutoff_frame,
                    target=task.target_col,
                    limit=max(1, per_cutoff_limit + (index < extra_rows)),
                    random_state=random_state + index,
                )
            )
        search_train = pd.concat(sampled_search_frames, ignore_index=True)
        search_validation = _stratified_rows(
            validation,
            target=task.target_col,
            limit=validation_limit,
            random_state=random_state + 1,
        )
        search_sample_rows = sample_rows
    split_column = "__kurversc_relbench_split__"
    train[split_column] = "train"
    validation[split_column] = "validation"
    search_train[split_column] = "train"
    search_validation[split_column] = "validation"
    full_parts = [train, validation]
    if test is not None:
        test[split_column] = "test"
        full_parts.append(test)
    labels = pd.concat(full_parts, ignore_index=True)
    search_labels = pd.concat([search_train, search_validation], ignore_index=True)

    root_meta = database.table_dict[task.entity_table]
    entity_values = search_labels[task.entity_col].dropna()
    search_root_frame = root_meta.df.loc[
        root_meta.df[root_meta.pkey_col].isin(entity_values)
    ]
    parent, tables, relationships = _edge_instances(
        database,
        root_table=task.entity_table,
        root_frame=root_meta.df,
        search_root_frame=search_root_frame,
        schema_depth=schema_depth,
        sample_rows=search_sample_rows,
        random_state=random_state,
        max_enrichment_columns=max_enrichment_columns,
    )
    label_spec = Labels(
        labels,
        search_source=search_labels,
        key=task.entity_col,
        target=task.target_col,
        timestamp=task.time_col,
        split=split_column,
    )
    return RelBenchProblem(
        dataset_name=dataset_name,
        task_name=task_name or type(task).__name__,
        parent_node=parent,
        label_node=label_spec,
        tables=tables,
        relationships=relationships,
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
    max_train_timestamps: int = 15,
    schema_depth: int = 3,
    max_enrichment_columns: int | None = 8,
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
        max_train_timestamps=max_train_timestamps,
        schema_depth=schema_depth,
        max_enrichment_columns=max_enrichment_columns,
        random_state=random_state,
    )
