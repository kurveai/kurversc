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


def _edge_instances(
    database: Any,
    *,
    root_table: str,
    root_frame: pd.DataFrame,
    search_root_frame: pd.DataFrame,
    schema_depth: int,
    sample_rows: int,
    random_state: int,
) -> tuple[Table, tuple[Table, ...], tuple[Relationship, ...]]:
    """Expand reverse foreign keys into an acyclic, path-specific graph."""

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
        parent_pk = table_dict[parent_base].pkey_col
        # RelBench association/event tables commonly have no primary key. They
        # remain valid reducible leaves, but cannot be parents of a deeper
        # foreign-key traversal.
        if parent_pk is None:
            continue
        parent_values = parent_frame[parent_pk].dropna()
        for child_base, child_meta in table_dict.items():
            for (
                foreign_key,
                referenced_table,
            ) in child_meta.fkey_col_to_pkey_table.items():
                if referenced_table != parent_base:
                    continue
                # Self/cyclic paths cannot be represented safely by GraphReduce's
                # DiGraph. Other foreign-key paths are each materialized once.
                if child_base in path:
                    continue
                base_name = f"{parent_instance}__{child_base}__{foreign_key}"
                instance = base_name
                suffix = 2
                while instance in used_names:
                    instance = f"{base_name}_{suffix}"
                    suffix += 1
                used_names.add(instance)
                child_search_frame = child_meta.df.loc[
                    child_meta.df[foreign_key].isin(parent_values)
                ]
                if len(child_search_frame) > sample_rows:
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
                )
                tables.append(child)
                relationships.append(
                    Relationship(
                        parent=parent_instance,
                        child=instance,
                        parent_key=parent_pk,
                        child_key=foreign_key,
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
    max_train_timestamps: int = 15,
    schema_depth: int = 3,
    random_state: int = 42,
) -> RelBenchProblem:
    """Translate already-censored RelBench objects into KurveRSC inputs.

    Full sources remain untouched for the production phase. Separate connected
    and stratified search sources are attached to the declarative table specs.
    """

    if sample_rows < 2:
        raise ValueError("sample_rows must be at least 2")
    if schema_depth < 1:
        raise ValueError("schema_depth must be at least 1")

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
            pd.Timestamp(value)
            for value in validation[task.time_col].dropna().unique()
        )
    )
    test_timestamps = (
        tuple(
            sorted(
                pd.Timestamp(value)
                for value in test[task.time_col].dropna().unique()
            )
        )
        if test is not None
        else ()
    )

    train_limit = max(1, sample_rows // 2)
    validation_limit = max(1, sample_rows - train_limit)
    # Configuration search is deliberately a single-frame operation. Select
    # the latest eligible training cutoff *before* row sampling so the entire
    # train-side sample budget is useful to the fitted search model. Full
    # refitting still receives every user-requested training frame above.
    latest_search_cutoff = selected_train_timestamps[-1]
    latest_train = train.loc[
        train[task.time_col] == latest_search_cutoff
    ].copy()
    search_train = _stratified_rows(
        latest_train,
        target=task.target_col,
        limit=train_limit,
        random_state=random_state,
    )
    search_validation = _stratified_rows(
        validation,
        target=task.target_col,
        limit=validation_limit,
        random_state=random_state + 1,
    )
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
    search_labels = pd.concat(
        [search_train, search_validation], ignore_index=True
    )

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
        sample_rows=sample_rows,
        random_state=random_state,
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
    max_train_timestamps: int = 15,
    schema_depth: int = 3,
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
    if schema_depth < 1:
        raise ValueError("schema_depth must be at least 1")
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
        max_train_timestamps=max_train_timestamps,
        schema_depth=schema_depth,
        random_state=random_state,
    )
