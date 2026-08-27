"""Implementation of the one-call KurveRSC interface."""

from __future__ import annotations

import hashlib
import io
import re
import copy
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Iterable, Iterator, Mapping, Sequence

import duckdb
import pandas as pd
import structlog

from .modeling import (
    fit_catboost,
    fit_final_catboost,
    prepare_features,
    prepare_prediction_features,
)
from .search import (
    DEFAULT_FAMILY_STAGES,
    FittedModel,
    FitResult,
    GraphConfig,
    Trial,
    annotate_complexity,
    incremental_configs,
)
from .specs import (
    GraphLabels,
    Key,
    Labels,
    Relationship,
    Source,
    Table,
    coerce_labels,
    coerce_relationship,
    coerce_table,
)


_NAME = re.compile(r"[^A-Za-z0-9_]+")
logger = structlog.get_logger("kurversc")


@contextmanager
def _connection_scope(
    connection: duckdb.DuckDBPyConnection | None,
    *,
    max_temp_directory_size: str = "32GB",
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Own an in-memory DuckDB connection and its bounded spill directory."""

    if connection is not None:
        yield connection
        return
    with TemporaryDirectory(prefix="kurversc-duckdb-") as temporary_directory:
        con = duckdb.connect(":memory:")
        escaped = temporary_directory.replace("'", "''")
        con.sql(f"SET temp_directory='{escaped}'")
        # A sampled candidate must fail cleanly instead of filling the host
        # filesystem with an unbounded intermediate join. Callers supplying
        # their own connection retain complete control over DuckDB settings.
        escaped_limit = max_temp_directory_size.replace("'", "''")
        con.sql(f"SET max_temp_directory_size='{escaped_limit}'")
        try:
            yield con
        finally:
            con.close()


def _freeze_execution_plan(graph: Any, cut_date: datetime) -> dict[str, Any]:
    plan = graph.freeze_execution_plan()
    # GraphReduce's external-label cut date is shifted by one microsecond to
    # make relation history inclusive. Root filters deliberately use the exact
    # task cutoff, so retain that second date for correct replay as well.
    plan["kurversc_root_cut_date"] = pd.Timestamp(cut_date).to_pydatetime()
    return plan


def _prepare_execution_plan(
    plan: Mapping[str, Any], *, root_prefix: str, cut_date: datetime
) -> dict[str, Any]:
    prepared = copy.deepcopy(dict(plan))
    original = prepared.get("kurversc_root_cut_date")
    if original is None:
        return prepared
    old = str(pd.Timestamp(original).to_pydatetime())
    new = str(pd.Timestamp(cut_date).to_pydatetime())
    for record in prepared.get("records", []):
        if record.get("node_prefix") != root_prefix:
            continue
        for field in ("ops", "method_ops", "date_filter_ops"):
            for operation in record.get(field, []):
                operation.opval = operation.opval.replace(old, new)
    return prepared


def _execution_plan_fingerprint(plan: Mapping[str, Any]) -> str:
    records = []
    for record in plan.get("records", []):
        records.append(
            (
                record.get("node_prefix"),
                record.get("method_name"),
                record.get("reduce_key"),
                record.get("edge"),
                tuple(
                    (getattr(operation, "optype", None), operation.opval)
                    for operation in record.get("ops", [])
                ),
            )
        )
    return hashlib.sha256(repr(records).encode()).hexdigest()


def _slug(value: str, fallback: str) -> str:
    result = _NAME.sub("_", value).strip("_").lower()
    if not result:
        result = fallback
    if result[0].isdigit():
        result = f"t_{result}"
    return result


def _source_name(source: Source, fallback: str) -> str:
    if isinstance(source, (str, Path)):
        value = str(source)
        path = Path(value)
        return _slug(path.stem if path.suffix else value.split(".")[-1], fallback)
    return fallback


def _key_parts(key: Key) -> list[str]:
    return [key] if isinstance(key, str) else list(key)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _quote_table(identifier: str) -> str:
    return ".".join(_quote_identifier(part) for part in identifier.split("."))


class _Workspace:
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        sample_rows: int,
        random_state: int,
    ) -> None:
        self.connection = connection
        self.sample_rows = sample_rows
        self.random_state = random_state
        self.views: dict[str, str] = {}
        self.columns: dict[str, list[str]] = {}
        self._registered: list[str] = []
        self._auxiliary_views: list[str] = []

    def _source_query(self, source: Source, name: str) -> str:
        if isinstance(source, pd.DataFrame):
            registration = f"kurversc_frame_{name}_{len(self._registered)}"
            # DuckDB 1.2 predates pandas 3's dedicated ``str`` dtype. Keep the
            # public API compatible with both by presenting string columns as
            # the object dtype understood by that GraphReduce-pinned release.
            compatible = source.copy(deep=False)
            string_columns = [
                column
                for column in compatible.columns
                if pd.api.types.is_string_dtype(compatible[column].dtype)
                and not pd.api.types.is_object_dtype(compatible[column].dtype)
            ]
            if string_columns:
                compatible = compatible.copy()
                for column in string_columns:
                    compatible[column] = compatible[column].astype(object)
            self.connection.register(registration, compatible)
            self._registered.append(registration)
            return f"SELECT * FROM {_quote_identifier(registration)}"
        value = str(source)
        path = Path(value).expanduser()
        if path.is_file():
            escaped = str(path.resolve()).replace("'", "''")
            if path.suffix.lower() in {".parquet", ".pq"}:
                return f"SELECT * FROM read_parquet('{escaped}')"
            if path.suffix.lower() in {".csv", ".csv.gz"}:
                return f"SELECT * FROM read_csv_auto('{escaped}')"
            raise ValueError(f"Unsupported source file type: {path.suffix}")
        return f"SELECT * FROM {_quote_table(value)}"

    def add(self, name: str, source: Source, *, sample: bool = True) -> str:
        if name in self.views:
            raise ValueError(f"Duplicate table name: {name}")
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        view = f"kurversc_{_slug(name, 'table')}_{digest}"
        query = self._source_query(source, name)
        limit = f" LIMIT {self.sample_rows}" if sample else ""
        self.connection.sql(
            f"CREATE OR REPLACE TEMP VIEW {_quote_identifier(view)} AS "
            f"SELECT * FROM ({query}) AS kurversc_source{limit}"
        )
        self.views[name] = view
        description = self.connection.sql(
            f"DESCRIBE SELECT * FROM {_quote_identifier(view)}"
        ).to_df()
        self.columns[name] = description["column_name"].astype(str).tolist()
        return view

    def frame(self, name: str) -> pd.DataFrame:
        return self.connection.sql(
            f"SELECT * FROM {_quote_identifier(self.views[name])}"
        ).to_df()

    def entity_filtered_view(
        self,
        name: str,
        *,
        key_columns: Sequence[str],
        entity_keys: pd.DataFrame,
    ) -> str:
        """Restrict a root source to the entities labeled at one cutoff."""

        ordinal = len(self._registered)
        registration = f"kurversc_entity_keys_{ordinal}"
        view = f"kurversc_{_slug(name, 'root')}_entities_{ordinal}"
        keys = entity_keys.loc[:, list(key_columns)].drop_duplicates().copy()
        self.connection.register(registration, keys)
        self._registered.append(registration)
        predicates = " AND ".join(
            f"source.{_quote_identifier(column)} = "
            f"task_entities.{_quote_identifier(column)}"
            for column in key_columns
        )
        self.connection.sql(
            f"CREATE OR REPLACE TEMP VIEW {_quote_identifier(view)} AS "
            f"SELECT source.* FROM {_quote_identifier(self.views[name])} AS source "
            f"INNER JOIN {_quote_identifier(registration)} AS task_entities "
            f"ON {predicates}"
        )
        self._auxiliary_views.append(view)
        return view

    def close(self) -> None:
        for view in self._auxiliary_views:
            try:
                self.connection.sql(f"DROP VIEW IF EXISTS {_quote_identifier(view)}")
            except Exception:
                pass
        for view in self.views.values():
            try:
                self.connection.sql(f"DROP VIEW IF EXISTS {_quote_identifier(view)}")
            except Exception:
                pass
        for registration in self._registered:
            try:
                self.connection.unregister(registration)
            except Exception:
                pass


def _normalize_tables(
    parent: Table,
    tables: Sequence[Table | Source] | Mapping[str, Table | Source],
) -> tuple[str, dict[str, Table]]:
    root_name = parent.name or _source_name(parent.source, "parent")
    root = Table(
        parent.source,
        search_source=parent.search_source,
        key=parent.key,
        name=root_name,
        date=parent.date,
        timeless=parent.timeless,
        prefix=parent.prefix,
        columns=parent.columns,
    )
    normalized = {root_name: root}
    values: Iterable[tuple[str | None, Table | Source]]
    values = (
        tables.items()
        if isinstance(tables, Mapping)
        else ((None, item) for item in tables)
    )
    for supplied_name, value in values:
        table = value if isinstance(value, Table) else Table(value, name=supplied_name)
        name = (
            supplied_name
            or table.name
            or _source_name(table.source, f"table_{len(normalized)}")
        )
        if name in normalized:
            raise ValueError(f"Duplicate table name: {name}")
        normalized[name] = Table(
            table.source,
            search_source=table.search_source,
            key=table.key,
            name=name,
            date=table.date,
            timeless=table.timeless,
            prefix=table.prefix,
            columns=table.columns,
        )
    return root_name, normalized


def _split_labels(
    labels: pd.DataFrame,
    spec: Labels,
    *,
    task: str,
    validation_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if spec.split:
        train = labels.loc[labels[spec.split] == spec.train_value].copy()
        validation = labels.loc[labels[spec.split] == spec.validation_value].copy()
    elif spec.timestamp:
        ordered = labels.sort_values(spec.timestamp, kind="stable")
        boundary = max(
            1, min(len(ordered) - 1, round(len(ordered) * (1 - validation_fraction)))
        )
        train, validation = (
            ordered.iloc[:boundary].copy(),
            ordered.iloc[boundary:].copy(),
        )
    else:
        from sklearn.model_selection import train_test_split

        stratify = labels[spec.target] if task == "classification" else None
        train, validation = train_test_split(
            labels,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=stratify,
        )
    if train.empty or validation.empty:
        raise ValueError("The label split produced an empty train or validation set")
    return train, validation


def _infer_task(labels: pd.Series, requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        return "classification" if labels.nunique(dropna=True) == 2 else "regression"
    if requested not in {"classification", "regression"}:
        raise ValueError("task must be 'classification', 'regression', or 'auto'")
    return requested


def _select_training_cutoffs(
    values: Sequence[Any], limit: int | None
) -> tuple[pd.Timestamp, ...]:
    """Select deterministic, evenly spaced point-in-time training frames."""

    timestamps = tuple(sorted({pd.Timestamp(value) for value in values}))
    if limit is None or len(timestamps) <= limit:
        return timestamps
    if limit < 1:
        raise ValueError("training frame counts must be positive")
    if limit == 1:
        return (timestamps[-1],)
    span = len(timestamps) - 1
    intervals = limit - 1
    indices = tuple(
        (index * span + intervals // 2) // intervals for index in range(limit)
    )
    return tuple(timestamps[index] for index in indices)


def _resource_block_for(
    config: GraphConfig,
    blocks: Sequence[tuple[tuple[str, ...], bool, int, str]],
) -> tuple[tuple[str, ...], bool, int, str] | None:
    """Return the prior resource failure that makes a candidate a superset."""

    for block in blocks:
        families, annotations, minimum_depth, _reason = block
        if (
            config.auto_annotate_features == annotations
            and config.depth >= minimum_depth
            and config.feature_families[: len(families)] == families
        ):
            return block
    return None


def _build_graph(
    workspace: _Workspace,
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    root_name: str,
    config: GraphConfig,
    *,
    cut_date: datetime,
    compute_period_days: int,
    excluded_columns: set[str],
    root_entity_keys: pd.DataFrame | None = None,
    graph_labels: GraphLabels | None = None,
    execution_plan: Mapping[str, Any] | None = None,
    train: bool = True,
):
    from graphreduce.enum import ComputeLayerEnum, PeriodUnit
    from graphreduce.graph_reduce import GraphReduce
    from graphreduce.models import sqlop
    from graphreduce.node import DuckdbNode
    from graphreduce.enum import SQLOpType

    root_key_columns = _key_parts(tables[root_name].key)
    root_source = (
        workspace.entity_filtered_view(
            root_name,
            key_columns=root_key_columns,
            entity_keys=root_entity_keys,
        )
        if root_entity_keys is not None
        else workspace.views[root_name]
    )
    # External point-in-time labels (including RelBench) define feature history
    # inclusively. Native GraphReduce labels deliberately retain GraphReduce's
    # own strict feature/label window boundaries around the exact cut date.
    graph_cut_date = (
        pd.Timestamp(cut_date)
        if graph_labels is not None
        else pd.Timestamp(cut_date) + pd.Timedelta(microseconds=1)
    ).to_pydatetime()
    nodes = {}
    for index, (name, table) in enumerate(tables.items()):
        available = workspace.columns[name]
        selected = list(table.columns) if table.columns is not None else list(available)
        selected = [column for column in selected if column not in excluded_columns]
        for key in _key_parts(table.key) if table.key is not None else []:
            if key not in selected:
                selected.insert(0, key)
        if table.date is not None and table.date not in selected:
            selected.append(table.date)
        missing_columns = set(selected) - set(available)
        if missing_columns:
            raise ValueError(
                f"Table {name!r} is missing configured columns: "
                f"{sorted(missing_columns)}"
            )
        prefix = table.prefix or f"{_slug(name, f'n{index}')[:10]}{index}"
        # GraphReduce nodes currently represent this setting as an integer.
        # A very large slice bound preserves KurveRSC's public ``None`` = no
        # cap semantics without changing GraphReduce itself.
        uncapped_budget = 2_147_483_647
        node = DuckdbNode(
            fpath=root_source if name == root_name else workspace.views[name],
            prefix=prefix,
            pk=table.key,
            date_key=table.date,
            columns=selected,
            ts_periods=[7, 30, 90],
            categorical_cardinality_threshold=20,
            categorical_top_k=5,
            auto_text_features=False,
            auto_annotate_features=config.auto_annotate_features,
            auto_annotate_max_categorical_columns=10,
            auto_annotate_max_gated_numeric_cols=4,
            auto_annotate_gated_numeric_top_k=3,
            feature_families=config.feature_families,
            feature_family_max_columns=(
                config.feature_family_max_columns
                if config.feature_family_max_columns is not None
                else uncapped_budget
            ),
        )
        if name == root_name:
            # Filter the source during do_data, before schema inference,
            # annotation, or joins. Merely setting date_key is insufficient:
            # GraphReduce only applies automatic cutoff predicates while
            # reducing relation nodes.
            select_expressions = [
                f"{node.render_identifier(column)} as "
                f"{node.render_identifier(node.colabbr(column))}"
                for column in selected
            ]
            root_filters = [
                sqlop(
                    optype=SQLOpType.where,
                    opval=f"{node.render_identifier(key)} IS NOT NULL",
                )
                for key in _key_parts(table.key)
            ]
            if table.date is not None:
                cutoff = pd.Timestamp(cut_date).isoformat(sep=" ")
                root_filters.append(
                    sqlop(
                        optype=SQLOpType.where,
                        opval=(
                            f"{node.render_identifier(table.date)} "
                            f"<= TIMESTAMP '{cutoff}'"
                        ),
                    )
                )
            node.do_data_ops = [
                sqlop(
                    optype=SQLOpType.select,
                    opval=",".join(select_expressions),
                ),
                *root_filters,
            ]
            # Also push the invariant through GraphReduce's public filtering
            # stage. These expressions reference the aliases produced by
            # do_data, matching custom production GraphReduce nodes.
            node.do_filters_ops = [
                sqlop(
                    optype=SQLOpType.where,
                    opval=(f"{node.render_identifier(node.colabbr(key))} IS NOT NULL"),
                )
                for key in _key_parts(table.key)
            ]
            if table.date is not None:
                node.do_filters_ops.append(
                    sqlop(
                        optype=SQLOpType.where,
                        opval=(
                            f"{node.render_identifier(node.colabbr(table.date))} "
                            f"<= TIMESTAMP '{cutoff}'"
                        ),
                    )
                )
        nodes[name] = node
    if graph_labels is not None:
        if graph_labels.table not in nodes:
            raise ValueError(
                f"GraphLabels references unknown table {graph_labels.table!r}"
            )
        label_node = nodes[graph_labels.table]
        if graph_labels.field not in workspace.columns[graph_labels.table]:
            raise ValueError(
                f"GraphLabels field {graph_labels.field!r} is missing from "
                f"table {graph_labels.table!r}"
            )
    else:
        label_node = None
    graph = GraphReduce(
        name=f"kurversc-{config.depth}-{'-'.join(config.feature_families)}",
        parent_node=nodes[root_name],
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=workspace.connection,
        cut_date=graph_cut_date,
        compute_period_val=compute_period_days,
        compute_period_unit=PeriodUnit.day,
        date_filters_on_agg=True,
        label_node=label_node,
        label_field=graph_labels.field if graph_labels else None,
        label_operation=graph_labels.operation if graph_labels else None,
        label_period_val=graph_labels.period_days if graph_labels else None,
        label_period_unit=PeriodUnit.day if graph_labels else None,
        train=train,
        **config.graphreduce_kwargs(),
    )
    for node in nodes.values():
        graph.add_node(node)
    for relationship in relationships:
        if relationship.parent not in nodes or relationship.child not in nodes:
            raise ValueError(
                f"Unknown relationship table: {relationship.parent} -> {relationship.child}"
            )
        graph.add_entity_edge(
            nodes[relationship.parent],
            nodes[relationship.child],
            parent_key=relationship.parent_key,
            relation_key=relationship.child_key,
            reduce=relationship.reduce,
        )
    if execution_plan is not None:
        # Date-dependent replay needs hydrated node periods before GraphReduce
        # can rebind cut, lookback, time-series, and future-label literals.
        graph.hydrate_graph_attrs()
        graph.apply_execution_plan(
            _prepare_execution_plan(
                execution_plan,
                root_prefix=nodes[root_name].prefix,
                cut_date=cut_date,
            )
        )
    return graph


def _materialize_at_cutoff(
    workspace: _Workspace,
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    root_name: str,
    config: GraphConfig,
    labels: pd.DataFrame,
    label_spec: Labels,
    *,
    cut_date: datetime,
    compute_period_days: int,
    verbose: bool,
    execution_plan: Mapping[str, Any] | None = None,
    frozen_plan_sink: list[dict[str, Any]] | None = None,
    train: bool = True,
    planning_labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    excluded = {label_spec.target}
    if label_spec.split:
        excluded.add(label_spec.split)
    if label_spec.timestamp:
        excluded.add(label_spec.timestamp)
    root = tables[root_name]
    root_keys = _key_parts(root.key)
    label_keys = _key_parts(label_spec.key)
    entity_labels = planning_labels if planning_labels is not None else labels
    root_entity_keys = entity_labels.loc[:, label_keys].rename(
        columns=dict(zip(label_keys, root_keys, strict=True))
    )
    with ExitStack() as stack:
        if not verbose:
            sink = io.StringIO()
            stack.enter_context(redirect_stdout(sink))
            stack.enter_context(redirect_stderr(sink))
        graph = _build_graph(
            workspace,
            tables,
            relationships,
            root_name,
            config,
            cut_date=cut_date,
            compute_period_days=compute_period_days,
            excluded_columns=excluded,
            root_entity_keys=root_entity_keys,
            execution_plan=execution_plan,
            train=train,
        )
        try:
            graph.do_transformations_sql()
            if frozen_plan_sink is not None:
                frozen_plan_sink.append(_freeze_execution_plan(graph, cut_date))
            features = workspace.connection.sql(
                f"SELECT * FROM {_quote_identifier(graph.parent_node._cur_data_ref)}"
            ).to_df()
        finally:
            graph._clean_refs()
    # GraphReduce intentionally keeps primary-key names unprefixed, while
    # custom node implementations may prefix them. Accept either convention.
    feature_keys = [
        key if key in features.columns else f"{graph.parent_node.prefix}_{key}"
        for key in root_keys
    ]
    return features.merge(
        labels,
        how="inner",
        left_on=feature_keys,
        right_on=label_keys,
        validate="one_to_many",
    )


def _materialize_split(
    workspace: _Workspace,
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    root_name: str,
    config: GraphConfig,
    labels: pd.DataFrame,
    label_spec: Labels,
    *,
    compute_period_days: int,
    verbose: bool,
    split_marker: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frozen_plans: list[dict[str, Any]] = []
    if label_spec.timestamp:
        parts = []
        if split_marker is not None:
            train_labels = labels.loc[labels[split_marker] == "train"]
            validation_labels = labels.loc[labels[split_marker] == "validation"]
        else:
            train_labels = labels
            validation_labels = labels.iloc[0:0]
        train_groups = list(
            train_labels.groupby(label_spec.timestamp, sort=True, dropna=False)
        )
        if not train_groups:
            raise ValueError("At least one training cutoff is required")
        # Fit feature operations once on the latest training cutoff, then
        # replay those exact operations everywhere else.
        anchor_timestamp, anchor_labels = train_groups[-1]
        if pd.isna(anchor_timestamp):
            raise ValueError("Label timestamps must not be missing")
        parts.append(
            _materialize_at_cutoff(
                workspace,
                tables,
                relationships,
                root_name,
                config,
                anchor_labels,
                label_spec,
                cut_date=pd.Timestamp(anchor_timestamp).to_pydatetime(),
                compute_period_days=compute_period_days,
                verbose=verbose,
                frozen_plan_sink=frozen_plans,
                planning_labels=train_labels,
            )
        )
        replay_groups = [
            *train_groups[:-1],
            *list(
                validation_labels.groupby(
                    label_spec.timestamp, sort=True, dropna=False
                )
            ),
        ]
        for timestamp, cutoff_labels in replay_groups:
            if pd.isna(timestamp):
                raise ValueError("Label timestamps must not be missing")
            cutoff = pd.Timestamp(timestamp).to_pydatetime()
            parts.append(
                _materialize_at_cutoff(
                    workspace,
                    tables,
                    relationships,
                    root_name,
                    config,
                    cutoff_labels,
                    label_spec,
                    cut_date=cutoff,
                    compute_period_days=compute_period_days,
                    verbose=verbose,
                    execution_plan=frozen_plans[0],
                )
            )
        return pd.concat(parts, ignore_index=True), frozen_plans[0]
    if split_marker is None:
        train_labels = labels
        validation_labels = labels.iloc[0:0]
    else:
        train_labels = labels.loc[labels[split_marker] == "train"]
        validation_labels = labels.loc[labels[split_marker] == "validation"]
    anchor = _materialize_at_cutoff(
        workspace,
        tables,
        relationships,
        root_name,
        config,
        train_labels,
        label_spec,
        cut_date=datetime.now(),
        compute_period_days=compute_period_days,
        verbose=verbose,
        frozen_plan_sink=frozen_plans,
    )
    if validation_labels.empty:
        return anchor, frozen_plans[0]
    validation = _materialize_at_cutoff(
        workspace,
        tables,
        relationships,
        root_name,
        config,
        validation_labels,
        label_spec,
        cut_date=datetime.now(),
        compute_period_days=compute_period_days,
        verbose=verbose,
        execution_plan=frozen_plans[0],
    )
    return pd.concat([anchor, validation], ignore_index=True), frozen_plans[0]


def _materialize_graph_labels_at_cutoff(
    workspace: _Workspace,
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    root_name: str,
    config: GraphConfig,
    graph_labels: GraphLabels,
    *,
    cut_date: datetime,
    split_value: str,
    split_marker: str,
    cutoff_marker: str,
    compute_period_days: int,
    verbose: bool,
    execution_plan: Mapping[str, Any] | None = None,
    frozen_plan_sink: list[dict[str, Any]] | None = None,
    train: bool = True,
) -> pd.DataFrame:
    """Run one graph whose target is produced by GraphReduce itself."""

    with ExitStack() as stack:
        if not verbose:
            sink = io.StringIO()
            stack.enter_context(redirect_stdout(sink))
            stack.enter_context(redirect_stderr(sink))
        graph = _build_graph(
            workspace,
            tables,
            relationships,
            root_name,
            config,
            cut_date=cut_date,
            compute_period_days=compute_period_days,
            excluded_columns={graph_labels.target},
            graph_labels=graph_labels,
            execution_plan=execution_plan,
            train=train,
        )
        label_node = graph.label_node
        generated_target = f"{label_node.colabbr(graph_labels.field)}_label"
        try:
            graph.do_transformations_sql()
            if frozen_plan_sink is not None:
                frozen_plan_sink.append(_freeze_execution_plan(graph, cut_date))
            frame = workspace.connection.sql(
                f"SELECT * FROM {_quote_identifier(graph.parent_node._cur_data_ref)}"
            ).to_df()
        finally:
            graph._clean_refs()
    if generated_target not in frame.columns:
        if not train:
            frame[cutoff_marker] = pd.Timestamp(cut_date)
            frame[split_marker] = split_value
            return frame
        raise ValueError(
            "GraphReduce did not attach its generated target column "
            f"{generated_target!r}; verify that the label table is connected "
            "to the parent and the relationship is reducible"
        )
    frame = frame.rename(columns={generated_target: graph_labels.target})
    if graph_labels.operation.lower() in {"bool", "count", "sum"}:
        frame[graph_labels.target] = frame[graph_labels.target].fillna(0)
    if graph_labels.operation.lower() == "bool":
        frame[graph_labels.target] = (
            pd.to_numeric(frame[graph_labels.target], errors="raise") > 0
        ).astype("int8")
    frame[cutoff_marker] = pd.Timestamp(cut_date)
    frame[split_marker] = split_value
    return frame


def _materialize_graph_label_splits(
    workspace: _Workspace,
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    root_name: str,
    config: GraphConfig,
    graph_labels: GraphLabels,
    *,
    split_marker: str,
    cutoff_marker: str,
    compute_period_days: int,
    verbose: bool,
    train_cutoffs: Sequence[Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    frozen_plans: list[dict[str, Any]] = []
    selected_train_cutoffs = tuple(train_cutoffs or graph_labels.train_cutoffs)
    if not selected_train_cutoffs:
        raise ValueError("At least one GraphLabels training frame is required")
    anchor = max(selected_train_cutoffs)
    frames.append(
        _materialize_graph_labels_at_cutoff(
            workspace,
            tables,
            relationships,
            root_name,
            config,
            graph_labels,
            cut_date=pd.Timestamp(anchor).to_pydatetime(),
            split_value="train",
            split_marker=split_marker,
            cutoff_marker=cutoff_marker,
            compute_period_days=compute_period_days,
            verbose=verbose,
            frozen_plan_sink=frozen_plans,
        )
    )
    for split_value, cutoffs in (
        (
            "train",
            tuple(cutoff for cutoff in selected_train_cutoffs if cutoff != anchor),
        ),
        ("validation", graph_labels.validation_cutoffs),
    ):
        for cutoff in cutoffs:
            frames.append(
                _materialize_graph_labels_at_cutoff(
                    workspace,
                    tables,
                    relationships,
                    root_name,
                    config,
                    graph_labels,
                    cut_date=pd.Timestamp(cutoff).to_pydatetime(),
                    split_value=split_value,
                    split_marker=split_marker,
                    cutoff_marker=cutoff_marker,
                    compute_period_days=compute_period_days,
                    verbose=verbose,
                    execution_plan=frozen_plans[0],
                )
            )
    return pd.concat(frames, ignore_index=True), frozen_plans[0]


def _materialize_external_test(
    workspace: _Workspace,
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    root_name: str,
    config: GraphConfig,
    labels: pd.DataFrame,
    label_spec: Labels,
    *,
    execution_plan: Mapping[str, Any],
    compute_period_days: int,
    verbose: bool,
) -> pd.DataFrame:
    """Build external-label test features without learning any operations."""

    if labels.empty:
        return pd.DataFrame()
    if label_spec.timestamp:
        parts = []
        groups = labels.groupby(label_spec.timestamp, sort=True, dropna=False)
        for timestamp, cutoff_labels in groups:
            if pd.isna(timestamp):
                raise ValueError("Test label timestamps must not be missing")
            parts.append(
                _materialize_at_cutoff(
                    workspace,
                    tables,
                    relationships,
                    root_name,
                    config,
                    cutoff_labels,
                    label_spec,
                    cut_date=pd.Timestamp(timestamp).to_pydatetime(),
                    compute_period_days=compute_period_days,
                    verbose=verbose,
                    execution_plan=execution_plan,
                    train=False,
                )
            )
        return pd.concat(parts, ignore_index=True)
    return _materialize_at_cutoff(
        workspace,
        tables,
        relationships,
        root_name,
        config,
        labels,
        label_spec,
        cut_date=datetime.now(),
        compute_period_days=compute_period_days,
        verbose=verbose,
        execution_plan=execution_plan,
        train=False,
    )


def _materialize_graph_test(
    workspace: _Workspace,
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    root_name: str,
    config: GraphConfig,
    graph_labels: GraphLabels,
    *,
    execution_plan: Mapping[str, Any],
    split_marker: str,
    cutoff_marker: str,
    compute_period_days: int,
    verbose: bool,
) -> pd.DataFrame:
    frames = [
        _materialize_graph_labels_at_cutoff(
            workspace,
            tables,
            relationships,
            root_name,
            config,
            graph_labels,
            cut_date=pd.Timestamp(cutoff).to_pydatetime(),
            split_value="test",
            split_marker=split_marker,
            cutoff_marker=cutoff_marker,
            compute_period_days=compute_period_days,
            verbose=verbose,
            execution_plan=execution_plan,
            train=False,
        )
        for cutoff in graph_labels.test_cutoffs
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _model_exclusions(
    tables: Mapping[str, Table],
    root_name: str,
    labels_spec: Labels | None,
    graph_labels: GraphLabels | None,
    cutoff_marker: str,
) -> set[str]:
    root = tables[root_name]
    excluded = {*_key_parts(root.key)}
    if labels_spec is not None:
        excluded.update(_key_parts(labels_spec.key))
    root_prefix = root.prefix or f"{_slug(root_name, 'n0')[:10]}0"
    excluded.update(f"{root_prefix}_{key}" for key in _key_parts(root.key))
    if graph_labels is not None:
        excluded.add(cutoff_marker)
    elif labels_spec is not None:
        if labels_spec.timestamp:
            excluded.add(labels_spec.timestamp)
        if labels_spec.split:
            excluded.add(labels_spec.split)
    return excluded


def fit(
    parent_node: Table | Source,
    label_node: Labels | GraphLabels | Source,
    *,
    tables: Sequence[Table | Source] | Mapping[str, Table | Source] = (),
    relationships: Sequence[Relationship | Mapping[str, Any]] = (),
    parent_key: Key | Sequence[str] | None = None,
    label_key: Key | Sequence[str] | None = None,
    target: str | None = None,
    parent_date: str | None = None,
    parent_timeless: bool = False,
    label_timestamp: str | None = None,
    split_column: str | None = None,
    task: str = "auto",
    max_depth: int = 3,
    feature_family_stages: Sequence[Sequence[str]] = DEFAULT_FAMILY_STAGES,
    auto_annotate_options: Sequence[bool] = (True, False),
    feature_family_max_columns: int | None = None,
    sample_rows: int = 100_000,
    search_training_frames: int = 1,
    full_training_frames: int | None = None,
    validation_fraction: float = 0.2,
    compute_period_days: int = 3650,
    random_state: int = 42,
    model_params: Mapping[str, Any] | None = None,
    connection: duckdb.DuckDBPyConnection | None = None,
    continue_on_error: bool = True,
    verbose: bool = False,
    classification_negligible_gain: float = 0.002,
    regression_negligible_relative_gain: float = 0.005,
    drastic_feature_growth: float = 2.0,
    graph_configs: Sequence[GraphConfig] | None = None,
) -> FitResult:
    """Fit and select a GraphReduce configuration using validation performance.

    The exact best validation score remains ``result.best_config``. A simpler
    configuration within the negligible-gain tolerance is separately exposed
    as ``result.recommended_config``.
    """

    if sample_rows < 1:
        raise ValueError("sample_rows must be positive")
    if search_training_frames != 1:
        raise ValueError(
            "search_training_frames must be 1; configuration search always uses "
            "the latest eligible training frame"
        )
    if full_training_frames is not None and full_training_frames < 1:
        raise ValueError("full_training_frames must be positive or None")
    parent = coerce_table(
        parent_node,
        key=parent_key,
        date=parent_date,
        timeless=parent_timeless,
    )
    graph_labels = label_node if isinstance(label_node, GraphLabels) else None
    labels_spec = (
        None
        if graph_labels is not None
        else coerce_labels(
            label_node,
            target=target,
            key=label_key,
            timestamp=label_timestamp,
            split=split_column,
        )
    )
    if parent.key is None or (labels_spec is not None and labels_spec.key is None):
        raise ValueError(
            "parent_key and label_key are required (directly or in Table/Labels)"
        )
    if labels_spec is not None and labels_spec.target is None:
        raise ValueError("target is required (directly or in Labels)")
    if (
        (graph_labels is not None or labels_spec.timestamp is not None)
        and parent.date is None
        and not parent.timeless
    ):
        raise ValueError(
            "parent_date is required when labels have timestamps unless the "
            "root entity is explicitly declared timeless=True; KurveRSC will "
            "not silently run a temporal task without a point-in-time root "
            "cutoff"
        )
    root_name, normalized_tables = _normalize_tables(parent, tables)
    normalized_relationships = tuple(
        coerce_relationship(item) for item in relationships
    )
    if graph_labels is not None and graph_labels.table not in normalized_tables:
        raise ValueError(
            f"GraphLabels table {graph_labels.table!r} must appear in tables"
        )
    configs = (
        tuple(graph_configs)
        if graph_configs is not None
        else incremental_configs(
            max_depth=max_depth,
            feature_family_stages=feature_family_stages,
            auto_annotate_options=auto_annotate_options,
            feature_family_max_columns=feature_family_max_columns,
        )
    )
    if not configs:
        raise ValueError("graph_configs must contain at least one configuration")

    with _connection_scope(connection) as con:
        workspace = _Workspace(con, sample_rows=sample_rows, random_state=random_state)
        try:
            for name, table in normalized_tables.items():
                workspace.add(
                    name,
                    table.search_source
                    if table.search_source is not None
                    else table.source,
                )
            split_marker = "__kurversc_validation_split__"
            cutoff_marker = "__kurversc_cutoff__"
            if graph_labels is not None:
                resolved_task = (
                    "classification"
                    if task == "auto" and graph_labels.operation.lower() == "bool"
                    else "regression" if task == "auto" else _infer_task(pd.Series(), task)
                )
                target_column = graph_labels.target
                search_labels = None
                search_graph_cutoffs = _select_training_cutoffs(
                    graph_labels.train_cutoffs, search_training_frames
                )
            else:
                label_name = "labels"
                while label_name in normalized_tables:
                    label_name += "_"
                workspace.add(
                    label_name,
                    labels_spec.search_source
                    if labels_spec.search_source is not None
                    else labels_spec.source,
                )
                label_frame = workspace.frame(label_name)
                required = {
                    *_key_parts(labels_spec.key),
                    labels_spec.target,
                    *([labels_spec.timestamp] if labels_spec.timestamp else []),
                    *([labels_spec.split] if labels_spec.split else []),
                }
                missing = required - set(label_frame.columns)
                if missing:
                    raise ValueError(
                        f"Label source is missing columns: {sorted(missing)}"
                    )
                label_frame = label_frame.dropna(subset=[labels_spec.target])
                resolved_task = _infer_task(label_frame[labels_spec.target], task)
                train_labels, validation_labels = _split_labels(
                    label_frame,
                    labels_spec,
                    task=resolved_task,
                    validation_fraction=validation_fraction,
                    random_state=random_state,
                )
                if labels_spec.timestamp:
                    selected_search_cutoffs = _select_training_cutoffs(
                        train_labels[labels_spec.timestamp].dropna().tolist(),
                        search_training_frames,
                    )
                    train_labels = train_labels.loc[
                        train_labels[labels_spec.timestamp].isin(
                            selected_search_cutoffs
                        )
                    ]
                while split_marker in label_frame.columns:
                    split_marker += "_"
                train_labels = train_labels.copy()
                validation_labels = validation_labels.copy()
                train_labels[split_marker] = "train"
                validation_labels[split_marker] = "validation"
                search_labels = pd.concat(
                    [train_labels, validation_labels], ignore_index=True
                )
                target_column = labels_spec.target

            logger.info(
                "search_started",
                candidates=len(configs),
                task=resolved_task,
                target=target_column,
                sample_rows=sample_rows,
                training_frames=(
                    len(search_graph_cutoffs)
                    if graph_labels is not None
                    else (
                        search_labels.loc[
                            search_labels[split_marker] == "train",
                            labels_spec.timestamp,
                        ].nunique()
                        if labels_spec.timestamp
                        else 1
                    )
                ),
            )
            trials: list[Trial] = []
            resource_blocks: list[tuple[tuple[str, ...], bool, int, str]] = []
            for trial_number, config in enumerate(configs, start=1):
                resource_block = _resource_block_for(config, resource_blocks)
                if resource_block is not None:
                    families, _annotations, minimum_depth, reason = resource_block
                    trials.append(
                        Trial(
                            config=config,
                            metric=(
                                "roc_auc"
                                if resolved_task == "classification"
                                else "mae"
                            ),
                            validation_score=float("nan"),
                            objective_score=float("-inf"),
                            feature_count=0,
                            train_rows=0,
                            validation_rows=0,
                            feature_seconds=0.0,
                            model_seconds=0.0,
                            status="skipped",
                            error=(
                                "Resource superset of failed configuration "
                                f"families={families}, depth={minimum_depth}: {reason}"
                            ),
                        )
                    )
                    logger.warning(
                        "trial_skipped",
                        trial=f"{trial_number}/{len(configs)}",
                        feature_families=config.feature_families,
                        depth=config.depth,
                        auto_annotate_features=config.auto_annotate_features,
                        reason="resource_superset",
                        blocked_by_families=families,
                        blocked_by_depth=minimum_depth,
                    )
                    continue
                logger.info(
                    "trial_started",
                    trial=f"{trial_number}/{len(configs)}",
                    feature_families=config.feature_families,
                    depth=config.depth,
                    auto_annotate_features=config.auto_annotate_features,
                    feature_family_max_columns=config.feature_family_max_columns,
                )
                feature_started = perf_counter()
                try:
                    if graph_labels is not None:
                        materialized, execution_plan = _materialize_graph_label_splits(
                            workspace,
                            normalized_tables,
                            normalized_relationships,
                            root_name,
                            config,
                            graph_labels,
                            split_marker=split_marker,
                            cutoff_marker=cutoff_marker,
                            compute_period_days=compute_period_days,
                            verbose=verbose,
                            train_cutoffs=search_graph_cutoffs,
                        )
                    else:
                        materialized, execution_plan = _materialize_split(
                            workspace,
                            normalized_tables,
                            normalized_relationships,
                            root_name,
                            config,
                            search_labels,
                            labels_spec,
                            compute_period_days=compute_period_days,
                            verbose=verbose,
                            split_marker=split_marker,
                        )
                    train = materialized.loc[
                        materialized[split_marker] == "train"
                    ].drop(columns=split_marker)
                    validation = materialized.loc[
                        materialized[split_marker] == "validation"
                    ].drop(columns=split_marker)
                    feature_seconds = perf_counter() - feature_started
                    if train.empty or validation.empty:
                        raise ValueError(
                            "No labels matched the sampled parent rows; increase "
                            "sample_rows or check parent_key/label_key"
                        )
                    excluded = _model_exclusions(
                        normalized_tables,
                        root_name,
                        labels_spec,
                        graph_labels,
                        cutoff_marker,
                    )
                    train_x, train_y, validation_x, validation_y, categorical = (
                        prepare_features(
                            train,
                            validation,
                            target=target_column,
                            excluded=excluded,
                        )
                    )
                    datetime_columns = tuple(
                        column
                        for column in train_x.columns
                        if pd.api.types.is_datetime64_any_dtype(train[column].dtype)
                    )
                    model, metric, score, model_seconds = fit_catboost(
                        train_x,
                        train_y,
                        validation_x,
                        validation_y,
                        task=resolved_task,
                        categorical=categorical,
                        random_state=random_state,
                        model_params=model_params,
                    )
                    completed_trial = Trial(
                        config=config,
                        metric=metric,
                        validation_score=score,
                        objective_score=score if metric == "roc_auc" else -score,
                        feature_count=train_x.shape[1],
                        train_rows=len(train_x),
                        validation_rows=len(validation_x),
                        feature_seconds=feature_seconds,
                        model_seconds=model_seconds,
                        model=model,
                        feature_columns=tuple(train_x.columns),
                        categorical_columns=tuple(
                            train_x.columns[index] for index in categorical
                        ),
                        datetime_columns=datetime_columns,
                        execution_plan=execution_plan,
                    )
                    trials.append(completed_trial)
                    logger.info(
                        "trial_completed",
                        trial=f"{trial_number}/{len(configs)}",
                        metric=metric,
                        score=round(score, 6),
                        features=completed_trial.feature_count,
                        train_rows=completed_trial.train_rows,
                        validation_rows=completed_trial.validation_rows,
                        feature_seconds=round(completed_trial.feature_seconds, 3),
                        model_seconds=round(completed_trial.model_seconds, 3),
                    )
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    error_text = f"{type(exc).__name__}: {exc}"
                    if (
                        type(exc).__name__ == "OutOfMemoryException"
                        or "max_temp_directory_size" in str(exc)
                    ):
                        resource_blocks.append(
                            (
                                config.feature_families,
                                config.auto_annotate_features,
                                config.depth,
                                error_text,
                            )
                        )
                    trials.append(
                        Trial(
                            config=config,
                            metric=(
                                "roc_auc"
                                if resolved_task == "classification"
                                else "mae"
                            ),
                            validation_score=float("nan"),
                            objective_score=float("-inf"),
                            feature_count=0,
                            train_rows=0,
                            validation_rows=0,
                            feature_seconds=perf_counter() - feature_started,
                            model_seconds=0.0,
                            status="failed",
                            error=error_text,
                        )
                    )
                    logger.warning(
                        "trial_failed",
                        trial=f"{trial_number}/{len(configs)}",
                        feature_families=config.feature_families,
                        depth=config.depth,
                        auto_annotate_features=config.auto_annotate_features,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
            annotate_complexity(
                trials,
                classification_gain=classification_negligible_gain,
                regression_relative_gain=regression_negligible_relative_gain,
                feature_growth=drastic_feature_growth,
            )
            successful = [trial for trial in trials if trial.status == "completed"]
            if not successful:
                errors = "; ".join(
                    f"{trial.config}: {trial.error}" for trial in trials[:3]
                )
                raise RuntimeError(f"Every KurveRSC trial failed. {errors}")
            best = max(successful, key=lambda item: item.objective_score)
            if best.metric == "roc_auc":
                eligible = [
                    trial
                    for trial in successful
                    if best.validation_score - trial.validation_score
                    <= classification_negligible_gain
                ]
            else:
                eligible = [
                    trial
                    for trial in successful
                    if trial.validation_score
                    <= best.validation_score * (1 + regression_negligible_relative_gain)
                ]
            recommended = min(
                eligible,
                key=lambda item: (
                    item.feature_count,
                    item.config.complexity,
                    item.feature_seconds,
                ),
            )
            for trial in successful:
                if trial.note:
                    logger.info(
                        "trial_complexity",
                        feature_families=trial.config.feature_families,
                        depth=trial.config.depth,
                        auto_annotate_features=(
                            trial.config.auto_annotate_features
                        ),
                        note=trial.note,
                    )
            logger.info(
                "search_selected",
                metric=best.metric,
                score=round(best.validation_score, 6),
                feature_families=best.config.feature_families,
                depth=best.config.depth,
                auto_annotate_features=best.config.auto_annotate_features,
                features=best.feature_count,
            )
            if recommended is not best:
                logger.info(
                    "search_recommended",
                    metric=recommended.metric,
                    score=round(recommended.validation_score, 6),
                    feature_families=recommended.config.feature_families,
                    depth=recommended.config.depth,
                    auto_annotate_features=(
                        recommended.config.auto_annotate_features
                    ),
                    features=recommended.feature_count,
                )

            # The search winner is only a configuration choice. Rebuild that
            # configuration against uncapped sources, learn one production
            # operation plan from full training, and replay it everywhere
            # else. Neither validation nor test may rediscover operations.
            logger.info(
                "full_refit_started",
                feature_families=best.config.feature_families,
                depth=best.config.depth,
                auto_annotate_features=best.config.auto_annotate_features,
                requested_training_frames=full_training_frames,
            )
            if connection is None:
                # Production deliberately operates on uncapped sources. Give
                # that requested work more room than disposable search
                # candidates while still preventing host-disk exhaustion.
                con.sql("SET max_temp_directory_size='128GB'")
            workspace.close()
            workspace = _Workspace(
                con, sample_rows=sample_rows, random_state=random_state
            )
            for name, table in normalized_tables.items():
                workspace.add(name, table.source, sample=False)

            external_test_labels = pd.DataFrame()
            if graph_labels is not None:
                full_graph_cutoffs = _select_training_cutoffs(
                    graph_labels.train_cutoffs, full_training_frames
                )
                full_materialized, production_plan = (
                    _materialize_graph_label_splits(
                        workspace,
                        normalized_tables,
                        normalized_relationships,
                        root_name,
                        best.config,
                        graph_labels,
                        split_marker=split_marker,
                        cutoff_marker=cutoff_marker,
                        compute_period_days=compute_period_days,
                        verbose=verbose,
                        train_cutoffs=full_graph_cutoffs,
                    )
                )
                production_training_frames = len(full_graph_cutoffs)
            else:
                workspace.add(label_name, labels_spec.source, sample=False)
                full_label_frame = workspace.frame(label_name)
                required = {
                    *_key_parts(labels_spec.key),
                    labels_spec.target,
                    *([labels_spec.timestamp] if labels_spec.timestamp else []),
                    *([labels_spec.split] if labels_spec.split else []),
                }
                missing = required - set(full_label_frame.columns)
                if missing:
                    raise ValueError(
                        f"Label source is missing columns: {sorted(missing)}"
                    )
                if labels_spec.split:
                    external_test_labels = full_label_frame.loc[
                        full_label_frame[labels_spec.split] == labels_spec.test_value
                    ].copy()
                labeled = full_label_frame.dropna(subset=[labels_spec.target])
                full_train_labels, full_validation_labels = _split_labels(
                    labeled,
                    labels_spec,
                    task=resolved_task,
                    validation_fraction=validation_fraction,
                    random_state=random_state,
                )
                if labels_spec.timestamp:
                    full_cutoffs = _select_training_cutoffs(
                        full_train_labels[labels_spec.timestamp].dropna().tolist(),
                        full_training_frames,
                    )
                    full_train_labels = full_train_labels.loc[
                        full_train_labels[labels_spec.timestamp].isin(full_cutoffs)
                    ]
                    production_training_frames = len(full_cutoffs)
                else:
                    production_training_frames = 1
                full_train_labels = full_train_labels.copy()
                full_validation_labels = full_validation_labels.copy()
                full_train_labels[split_marker] = "train"
                full_validation_labels[split_marker] = "validation"
                full_labels = pd.concat(
                    [full_train_labels, full_validation_labels], ignore_index=True
                )
                full_materialized, production_plan = _materialize_split(
                    workspace,
                    normalized_tables,
                    normalized_relationships,
                    root_name,
                    best.config,
                    full_labels,
                    labels_spec,
                    compute_period_days=compute_period_days,
                    verbose=verbose,
                    split_marker=split_marker,
                )

            full_train = full_materialized.loc[
                full_materialized[split_marker] == "train"
            ].drop(columns=split_marker)
            full_validation = full_materialized.loc[
                full_materialized[split_marker] == "validation"
            ].drop(columns=split_marker)
            if full_train.empty or full_validation.empty:
                raise ValueError("Full-data refit produced an empty model split")
            plan_fingerprint = _execution_plan_fingerprint(production_plan)
            logger.info(
                "production_plan_frozen",
                training_frames=production_training_frames,
                operations=len(production_plan.get("records", [])),
                fingerprint=plan_fingerprint,
            )
            excluded = _model_exclusions(
                normalized_tables,
                root_name,
                labels_spec,
                graph_labels,
                cutoff_marker,
            )
            (
                full_train_x,
                full_train_y,
                full_validation_x,
                full_validation_y,
                full_categorical,
            ) = prepare_features(
                full_train,
                full_validation,
                target=target_column,
                excluded=excluded,
            )
            full_datetime_columns = tuple(
                column
                for column in full_train_x.columns
                if pd.api.types.is_datetime64_any_dtype(full_train[column].dtype)
            )
            validation_model, full_metric, full_validation_score, _ = fit_catboost(
                full_train_x,
                full_train_y,
                full_validation_x,
                full_validation_y,
                task=resolved_task,
                categorical=full_categorical,
                random_state=random_state,
                model_params=model_params,
            )
            combined_x = pd.concat(
                [full_train_x, full_validation_x], ignore_index=True
            )
            combined_y = pd.concat(
                [full_train_y, full_validation_y], ignore_index=True
            )
            final_model, final_model_seconds, target_classes = fit_final_catboost(
                combined_x,
                combined_y,
                task=resolved_task,
                categorical=full_categorical,
                random_state=random_state,
                model_params=model_params,
            )

            test_frame = (
                _materialize_graph_test(
                    workspace,
                    normalized_tables,
                    normalized_relationships,
                    root_name,
                    best.config,
                    graph_labels,
                    execution_plan=production_plan,
                    split_marker=split_marker,
                    cutoff_marker=cutoff_marker,
                    compute_period_days=compute_period_days,
                    verbose=verbose,
                )
                if graph_labels is not None
                else _materialize_external_test(
                    workspace,
                    normalized_tables,
                    normalized_relationships,
                    root_name,
                    best.config,
                    external_test_labels,
                    labels_spec,
                    execution_plan=production_plan,
                    compute_period_days=compute_period_days,
                    verbose=verbose,
                )
            )
            test_predictions = None
            test_score = None
            if not test_frame.empty:
                test_x = prepare_prediction_features(
                    test_frame,
                    columns=list(full_train_x.columns),
                    categorical_columns={
                        full_train_x.columns[index] for index in full_categorical
                    },
                    datetime_columns=set(full_datetime_columns),
                )
                predictions = (
                    final_model.predict_proba(test_x)[:, 1]
                    if resolved_task == "classification"
                    else final_model.predict(test_x)
                )
                root = normalized_tables[root_name]
                root_prefix = root.prefix or f"{_slug(root_name, 'n0')[:10]}0"
                test_predictions = pd.DataFrame(index=test_frame.index)
                for key in _key_parts(root.key):
                    source_key = (
                        key if key in test_frame else f"{root_prefix}_{key}"
                    )
                    if source_key in test_frame:
                        test_predictions[key] = test_frame[source_key]
                for column in (
                    cutoff_marker,
                    *((labels_spec.timestamp,) if labels_spec and labels_spec.timestamp else ()),
                ):
                    if column in test_frame:
                        test_predictions[column] = test_frame[column]
                test_predictions["prediction"] = predictions
                if target_column in test_frame and test_frame[target_column].notna().all():
                    if resolved_task == "classification":
                        from sklearn.metrics import roc_auc_score

                        class_map = {
                            value: index for index, value in enumerate(target_classes)
                        }
                        encoded_test = test_frame[target_column].map(class_map)
                        if encoded_test.nunique() == 2:
                            test_score = float(roc_auc_score(encoded_test, predictions))
                    else:
                        from sklearn.metrics import mean_absolute_error

                        test_score = float(
                            mean_absolute_error(
                                pd.to_numeric(test_frame[target_column]), predictions
                            )
                        )

            fitted_model = FittedModel(
                config=best.config,
                execution_plan=production_plan,
                plan_fingerprint=plan_fingerprint,
                estimator=final_model,
                validation_estimator=validation_model,
                feature_columns=tuple(full_train_x.columns),
                categorical_columns=tuple(
                    full_train_x.columns[index] for index in full_categorical
                ),
                datetime_columns=full_datetime_columns,
                target=target_column,
                task=resolved_task,
                metric=full_metric,
                validation_score=full_validation_score,
                train_rows=len(full_train_x),
                validation_rows=len(full_validation_x),
                training_frames=production_training_frames,
                target_classes=target_classes,
                test_predictions=test_predictions,
                test_score=test_score,
            )
            logger.info(
                "full_refit_completed",
                metric=full_metric,
                validation_score=round(full_validation_score, 6),
                training_frames=production_training_frames,
                train_rows=len(full_train_x),
                validation_rows=len(full_validation_x),
                features=len(full_train_x.columns),
                plan_operations=len(production_plan.get("records", [])),
                plan_fingerprint=plan_fingerprint,
                final_model_seconds=round(final_model_seconds, 3),
                test_rows=0 if test_predictions is None else len(test_predictions),
                test_score=test_score,
            )
            return FitResult(
                task=resolved_task,
                metric=best.metric,
                best_trial=best,
                recommended_trial=recommended,
                trials=tuple(trials),
                fitted_model=fitted_model,
            )
        finally:
            workspace.close()


def predict(
    fitted: FitResult,
    parent_node: Table | Source,
    prediction_node: Labels | Source,
    *,
    tables: Sequence[Table | Source] | Mapping[str, Table | Source] = (),
    relationships: Sequence[Relationship | Mapping[str, Any]] = (),
    parent_key: Key | Sequence[str] | None = None,
    label_key: Key | Sequence[str] | None = None,
    parent_date: str | None = None,
    parent_timeless: bool = False,
    label_timestamp: str | None = None,
    compute_period_days: int = 3650,
    connection: duckdb.DuckDBPyConnection | None = None,
    verbose: bool = False,
    use_validation_model: bool = False,
) -> pd.DataFrame:
    """Replay a fitted production plan for new point-in-time entity rows.

    No GraphReduce feature discovery runs here: the production execution plan
    and ordered model schema captured by :func:`fit` are mandatory.
    """

    artifact = fitted.fitted_model
    if artifact is None:
        raise ValueError("The FitResult does not contain a production fitted model")
    parent = coerce_table(
        parent_node,
        key=parent_key,
        date=parent_date,
        timeless=parent_timeless,
    )
    labels_spec = coerce_labels(
        prediction_node,
        target=artifact.target,
        key=label_key,
        timestamp=label_timestamp,
        split=None,
    )
    if parent.key is None or labels_spec.key is None:
        raise ValueError("parent_key and label_key are required for prediction")
    if labels_spec.timestamp and parent.date is None and not parent.timeless:
        raise ValueError(
            "parent_date is required for temporal prediction unless the root "
            "entity is explicitly timeless"
        )
    root_name, normalized_tables = _normalize_tables(parent, tables)
    normalized_relationships = tuple(
        coerce_relationship(item) for item in relationships
    )

    with _connection_scope(
        connection, max_temp_directory_size="128GB"
    ) as con:
        workspace = _Workspace(con, sample_rows=1, random_state=0)
        try:
            for name, table in normalized_tables.items():
                workspace.add(name, table.source, sample=False)
            label_name = "prediction_rows"
            while label_name in normalized_tables:
                label_name += "_"
            workspace.add(label_name, labels_spec.source, sample=False)
            rows = workspace.frame(label_name)
            required = {
                *_key_parts(labels_spec.key),
                *([labels_spec.timestamp] if labels_spec.timestamp else []),
            }
            missing = required - set(rows.columns)
            if missing:
                raise ValueError(
                    f"Prediction source is missing columns: {sorted(missing)}"
                )
            order_column = "__kurversc_prediction_order__"
            while order_column in rows:
                order_column += "_"
            rows = rows.copy()
            rows[order_column] = range(len(rows))
            frame = _materialize_external_test(
                workspace,
                normalized_tables,
                normalized_relationships,
                root_name,
                artifact.config,
                rows,
                labels_spec,
                execution_plan=artifact.execution_plan,
                compute_period_days=compute_period_days,
                verbose=verbose,
            )
            inputs = prepare_prediction_features(
                frame,
                columns=list(artifact.feature_columns),
                categorical_columns=set(artifact.categorical_columns),
                datetime_columns=set(artifact.datetime_columns),
            )
            estimator = (
                artifact.validation_estimator
                if use_validation_model
                else artifact.estimator
            )
            values = (
                estimator.predict_proba(inputs)[:, 1]
                if artifact.task == "classification"
                else estimator.predict(inputs)
            )
            frame = frame.assign(__kurversc_prediction__=values)
            if (
                len(frame) != len(rows)
                or frame[order_column].duplicated().any()
                or set(frame[order_column]) != set(rows[order_column])
            ):
                raise ValueError(
                    "Frozen graph prediction did not return exactly one row for "
                    "every requested entity/timestamp"
                )
            prediction_by_order = frame.set_index(order_column)[
                "__kurversc_prediction__"
            ]
            output = rows.sort_values(order_column, kind="stable").drop(
                columns=order_column
            )
            output = output.copy()
            output["prediction"] = prediction_by_order.reindex(
                rows.sort_values(order_column, kind="stable")[order_column]
            ).to_numpy()
            return output.reset_index(drop=True)
        finally:
            workspace.close()
