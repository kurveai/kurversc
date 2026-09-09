"""Snowflake-native GraphReduce and Snowflake ML joint optimization.

This module deliberately keeps Snowflake dependencies lazy. Importing
``kurversc`` therefore remains lightweight for users of the local DuckDB path.
"""

from __future__ import annotations

import copy
import re
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from .core import (
    _build_graph,
    _execution_plan_fingerprint,
    _freeze_execution_plan,
    _normalize_tables,
    _select_trials,
    _slug,
    logger,
)
from .feature_audit import free_text_columns
from .search import (
    DEFAULT_FAMILY_STAGES,
    FittedModel,
    FitResult,
    GraphConfig,
    Trial,
    adaptive_depth_candidate_allowed,
    annotate_complexity,
    forward_candidate_allowed,
    incremental_configs,
    resolve_feature_family_column_budgets,
)
from .specs import (
    GraphLabels,
    Key,
    Relationship,
    Source,
    Table,
    coerce_relationship,
    coerce_table,
)


_SIMPLE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_NUMERIC_TYPES = (
    "NUMBER",
    "DECIMAL",
    "NUMERIC",
    "INT",
    "INTEGER",
    "BIGINT",
    "SMALLINT",
    "FLOAT",
    "DOUBLE",
    "REAL",
    "BOOLEAN",
)


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _quote_relation(value: str) -> str:
    parts = str(value).split(".")
    if not parts or any(not _SIMPLE_IDENTIFIER.fullmatch(part) for part in parts):
        raise ValueError(
            f"Snowflake relation {value!r} must contain one to three simple "
            "unquoted identifiers"
        )
    if len(parts) > 3:
        raise ValueError(f"Snowflake relation {value!r} has more than three parts")
    # This API accepts unquoted Snowflake identifiers. Snowflake folds those
    # names to uppercase when GraphReduce creates its temporary relations, so
    # preserve those semantics when we subsequently delimit the names.
    return ".".join(_quote_identifier(part.upper()) for part in parts)


def _literal(value: Any) -> str:
    return str(value).replace("'", "''")


def _key_parts(key: Key) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else tuple(key)


def _description_name(item: Any) -> str:
    name = getattr(item, "name", None)
    return str(name if name is not None else item[0])


def _execute(connection: Any, sql: str) -> Any:
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        return cursor
    except Exception:
        cursor.close()
        raise


def _execute_no_result(connection: Any, sql: str) -> None:
    cursor = _execute(connection, sql)
    cursor.close()


def _query_columns(connection: Any, relation: str) -> tuple[str, ...]:
    cursor = _execute(connection, f"SELECT * FROM {_quote_relation(relation)} LIMIT 0")
    try:
        return tuple(_description_name(item) for item in (cursor.description or ()))
    finally:
        cursor.close()


def _query_scalar(connection: Any, sql: str) -> Any:
    cursor = _execute(connection, sql)
    try:
        row = cursor.fetchone()
        return None if row is None else row[0]
    finally:
        cursor.close()


def _query_values(connection: Any, sql: str) -> tuple[Any, ...]:
    cursor = _execute(connection, sql)
    try:
        return tuple(row[0] for row in cursor.fetchall())
    finally:
        cursor.close()


def _resolve_column(columns: Sequence[str], requested: str) -> str:
    exact = {column: column for column in columns}
    if requested in exact:
        return requested
    matches = [column for column in columns if column.lower() == requested.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Snowflake column {requested!r} does not exist")
    raise ValueError(f"Snowflake column {requested!r} is ambiguous")


def _resolve_key(columns: Sequence[str], key: Key) -> Key:
    resolved = tuple(_resolve_column(columns, part) for part in _key_parts(key))
    return resolved[0] if len(resolved) == 1 else resolved


@dataclass(frozen=True)
class SnowflakeBackend:
    """Connection and artifact namespace for Snowflake-native optimization."""

    connection: Any = field(repr=False, compare=False)
    database: str
    schema: str
    warehouse: str | None = None
    artifact_prefix: str = "KURVERSC"

    def __post_init__(self) -> None:
        for field_name in ("database", "schema", "artifact_prefix"):
            value = getattr(self, field_name)
            if len(str(value)) > 255 or not _SIMPLE_IDENTIFIER.fullmatch(str(value)):
                raise ValueError(
                    f"SnowflakeBackend.{field_name} must be a simple identifier"
                )
            object.__setattr__(self, field_name, str(value).upper())
        if self.warehouse is not None and (
            len(str(self.warehouse)) > 255
            or not _SIMPLE_IDENTIFIER.fullmatch(str(self.warehouse))
        ):
            raise ValueError("SnowflakeBackend.warehouse must be a simple identifier")
        if self.warehouse is not None:
            object.__setattr__(self, "warehouse", str(self.warehouse).upper())


@dataclass(frozen=True)
class SnowflakeModelReference:
    """Stable reference to the winning model object in Snowflake."""

    database: str
    schema: str
    name: str
    version: str
    task: str
    feature_columns: tuple[str, ...]
    learner: str = "snowpark_xgboost"
    model_type: str = "snowflake.ml.modeling.xgboost"
    persistence: str = "snowflake_model_registry"

    @property
    def fqn(self) -> str:
        return f"{self.database}.{self.schema}.{self.name}"


@dataclass(frozen=True)
class SnowflakeEvaluation:
    metric: str
    score: float
    model_seconds: float
    target_classes: tuple[Any, ...] = ()


@dataclass
class SnowflakeFeatureSet:
    """Temporary Snowflake relations for one GraphConfig candidate."""

    connection: Any = field(repr=False)
    train_relation: str
    validation_relation: str
    combined_relation: str
    feature_columns: tuple[str, ...]
    execution_plan: dict[str, Any]
    train_rows: int
    validation_rows: int
    temporary_relations: tuple[str, ...]

    def close(self) -> None:
        for relation in reversed(self.temporary_relations):
            try:
                _execute_no_result(
                    self.connection,
                    f"DROP TABLE IF EXISTS {_quote_relation(relation)}",
                )
            except Exception:
                pass


class SnowflakeGraphMaterializer:
    """Materialize frozen GraphReduce candidates entirely inside Snowflake."""

    split_marker = "__KURVERSC_SPLIT__"
    cutoff_marker = "__KURVERSC_CUTOFF__"

    def __init__(
        self,
        backend: SnowflakeBackend,
        tables: Mapping[str, Table],
        relationships: Sequence[Relationship],
        root_name: str,
        graph_labels: GraphLabels,
        *,
        compute_period_days: int,
        sample_rows: int | None,
        infer_ts_periods: bool,
        verbose: bool,
    ) -> None:
        self.backend = backend
        self.connection = backend.connection
        self.root_name = root_name
        self.graph_labels = graph_labels
        self.compute_period_days = compute_period_days
        self.sample_rows = sample_rows
        self.infer_ts_periods = infer_ts_periods
        self.verbose = verbose
        self.views: dict[str, str] = {}
        self.columns: dict[str, list[str]] = {}
        self._free_text_columns: dict[str, frozenset[str]] = {}
        self._source_relations: list[str] = []
        self._sequence = 0
        try:
            self._configure_session()
            self._add_sources(tables)
            self.tables, self.relationships, self.graph_labels = self._resolve_metadata(
                tables, relationships, graph_labels
            )
        except Exception:
            self.close()
            raise

    def _configure_session(self) -> None:
        if self.backend.warehouse:
            _execute_no_result(
                self.connection,
                f"USE WAREHOUSE {_quote_identifier(self.backend.warehouse)}",
            )
        _execute_no_result(
            self.connection,
            f"USE DATABASE {_quote_identifier(self.backend.database)}",
        )
        _execute_no_result(
            self.connection,
            f"USE SCHEMA {_quote_identifier(self.backend.schema)}",
        )

    def _temporary_name(self, kind: str) -> str:
        self._sequence += 1
        prefix = _slug(self.backend.artifact_prefix, "kurversc").upper()[:24]
        kind_token = _slug(kind, "artifact").upper()[:40]
        suffix = uuid.uuid4().hex[:10].upper()
        # Keep room for GraphReduce's node/method/namespace suffixes when this
        # name becomes the source portion of a generated temporary relation.
        return f"{prefix}_{kind_token}_{self._sequence}_{suffix}"[:120]

    def _add_sources(self, tables: Mapping[str, Table]) -> None:
        for logical_name, table in tables.items():
            if not isinstance(table.source, str):
                raise TypeError(
                    "fit_snowflake requires every Table.source to be a Snowflake "
                    "table or view name"
                )
            source = _quote_relation(table.source)
            alias = self._temporary_name(f"SRC_{_slug(logical_name, 'TABLE')}")
            limit = f" LIMIT {int(self.sample_rows)}" if self.sample_rows else ""
            _execute_no_result(
                self.connection,
                f"CREATE OR REPLACE TEMPORARY VIEW {_quote_identifier(alias)} AS "
                f"SELECT * FROM {source}{limit}",
            )
            self._source_relations.append(alias)
            self.views[logical_name] = alias
            self.columns[logical_name] = list(_query_columns(self.connection, alias))

    def _resolve_metadata(
        self,
        tables: Mapping[str, Table],
        relationships: Sequence[Relationship],
        graph_labels: GraphLabels,
    ) -> tuple[dict[str, Table], tuple[Relationship, ...], GraphLabels]:
        resolved_tables: dict[str, Table] = {}
        for name, table in tables.items():
            columns = self.columns[name]
            resolved_tables[name] = replace(
                table,
                key=_resolve_key(columns, table.key) if table.key is not None else None,
                date=(
                    _resolve_column(columns, table.date)
                    if table.date is not None
                    else None
                ),
                columns=(
                    tuple(_resolve_column(columns, column) for column in table.columns)
                    if table.columns is not None
                    else None
                ),
                context_keys=tuple(
                    _resolve_column(columns, column) for column in table.context_keys
                ),
            )
        resolved_relationships = []
        for relationship in relationships:
            self._validate_relationship_tables(relationship)
            resolved_relationships.append(
                replace(
                    relationship,
                    parent_key=_resolve_key(
                        self.columns[relationship.parent], relationship.parent_key
                    ),
                    child_key=_resolve_key(
                        self.columns[relationship.child], relationship.child_key
                    ),
                )
            )
        resolved_labels = replace(
            graph_labels,
            field=_resolve_column(self.columns[graph_labels.table], graph_labels.field),
        )
        return resolved_tables, tuple(resolved_relationships), resolved_labels

    def free_text_columns(self, name: str) -> frozenset[str]:
        """Classify free text once per sampled Snowflake source view."""
        if name in self._free_text_columns:
            return self._free_text_columns[name]
        cursor = _execute(
            self.connection,
            f"SELECT * FROM {_quote_relation(self.views[name])} LIMIT 500",
        )
        try:
            columns = [_description_name(item) for item in (cursor.description or ())]
            sample = pd.DataFrame(cursor.fetchall(), columns=columns)
        finally:
            cursor.close()
        detected = free_text_columns(sample)
        self._free_text_columns[name] = detected
        return detected

    def _validate_relationship_tables(self, relationship: Relationship) -> None:
        missing = [
            name
            for name in (relationship.parent, relationship.child)
            if name not in self.columns
        ]
        if missing:
            raise ValueError(
                "Unknown Snowflake relationship table(s): " + ", ".join(missing)
            )

    def entity_filtered_view(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        raise NotImplementedError(
            "Snowflake GraphLabels do not require a client-side entity filter"
        )

    def _cleanup_graph(self, graph: Any) -> None:
        refs: list[str] = []
        for node in graph.nodes():
            refs.extend(str(ref) for ref in getattr(node, "_all_refs", ()))
        for relation in reversed(tuple(dict.fromkeys(refs))):
            try:
                _execute_no_result(
                    self.connection,
                    f"DROP TABLE IF EXISTS {_quote_relation(relation)}",
                )
            except Exception:
                pass

    def _materialize_cutoff(
        self,
        config: GraphConfig,
        *,
        cut_date: pd.Timestamp,
        split_value: str,
        execution_plan: Mapping[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        from graphreduce.enum import ComputeLayerEnum
        from graphreduce.node import SnowflakeNode

        graph = _build_graph(
            self,
            self.tables,
            self.relationships,
            self.root_name,
            config,
            cut_date=cut_date.to_pydatetime(),
            compute_period_days=self.compute_period_days,
            excluded_columns={self.graph_labels.target},
            graph_labels=self.graph_labels,
            execution_plan=execution_plan,
            train=True,
            infer_ts_periods=self.infer_ts_periods and execution_plan is None,
            compute_layer=ComputeLayerEnum.snowflake,
            node_class=SnowflakeNode,
        )
        output_relation = self._temporary_name(f"FRAME_{split_value}")
        try:
            graph.do_transformations_sql()
            selected_plan = (
                copy.deepcopy(dict(execution_plan))
                if execution_plan is not None
                else _freeze_execution_plan(graph, cut_date.to_pydatetime())
            )
            source_relation = str(graph.parent_node._cur_data_ref)
            source_columns = _query_columns(self.connection, source_relation)
            generated_target = (
                f"{graph.label_node.colabbr(self.graph_labels.field)}_label"
            )
            generated_target = _resolve_column(source_columns, generated_target)
            projection: list[str] = []
            for column in source_columns:
                if column.lower() == generated_target.lower():
                    target_expression = _quote_identifier(column)
                    if self.graph_labels.operation.lower() in {"bool", "count", "sum"}:
                        target_expression = f"COALESCE({target_expression}, 0)"
                    projection.append(
                        f"{target_expression} AS {_quote_identifier(self.graph_labels.target)}"
                    )
                elif column.lower() != self.graph_labels.target.lower():
                    projection.append(_quote_identifier(column))
            projection.extend(
                [
                    f"'{_literal(split_value)}' AS {_quote_identifier(self.split_marker)}",
                    "TO_TIMESTAMP_NTZ("
                    f"'{_literal(cut_date.isoformat())}') AS "
                    f"{_quote_identifier(self.cutoff_marker)}",
                ]
            )
            _execute_no_result(
                self.connection,
                f"CREATE OR REPLACE TEMPORARY TABLE {_quote_identifier(output_relation)} AS "
                f"SELECT {', '.join(projection)} FROM {_quote_relation(source_relation)}",
            )
            return output_relation, selected_plan
        finally:
            self._cleanup_graph(graph)

    def _combine(self, relations: Sequence[str], kind: str) -> str:
        if not relations:
            raise ValueError(f"Snowflake {kind} relation list must not be empty")
        expected = _query_columns(self.connection, relations[0])
        selects = []
        for relation in relations:
            columns = _query_columns(self.connection, relation)
            if tuple(column.lower() for column in columns) != tuple(
                column.lower() for column in expected
            ):
                raise ValueError(
                    "Frozen GraphReduce plan produced inconsistent Snowflake schemas"
                )
            selects.append(f"SELECT * FROM {_quote_relation(relation)}")
        combined = self._temporary_name(kind)
        _execute_no_result(
            self.connection,
            f"CREATE OR REPLACE TEMPORARY TABLE {_quote_identifier(combined)} AS "
            + " UNION ALL ".join(selects),
        )
        return combined

    def _column_types(self, relation: str) -> dict[str, str]:
        cursor = _execute(self.connection, f"DESC TABLE {_quote_relation(relation)}")
        try:
            return {str(row[0]): str(row[1]).upper() for row in cursor.fetchall()}
        finally:
            cursor.close()

    def materialize(
        self,
        config: GraphConfig,
        *,
        execution_plan: Mapping[str, Any] | None = None,
        include_combined: bool = False,
    ) -> SnowflakeFeatureSet:
        temporary_relations: list[str] = []
        selected_plan = copy.deepcopy(dict(execution_plan)) if execution_plan else None
        try:
            train_relations: list[str] = []
            train_cutoffs = tuple(sorted(self.graph_labels.train_cutoffs))
            anchor = train_cutoffs[-1]
            anchor_relation, selected_plan = self._materialize_cutoff(
                config,
                cut_date=pd.Timestamp(anchor),
                split_value="train",
                execution_plan=selected_plan,
            )
            train_relations.append(anchor_relation)
            temporary_relations.append(anchor_relation)
            for cutoff in train_cutoffs[:-1]:
                relation, _ = self._materialize_cutoff(
                    config,
                    cut_date=pd.Timestamp(cutoff),
                    split_value="train",
                    execution_plan=selected_plan,
                )
                train_relations.append(relation)
                temporary_relations.append(relation)

            validation_relations: list[str] = []
            for cutoff in self.graph_labels.validation_cutoffs:
                relation, _ = self._materialize_cutoff(
                    config,
                    cut_date=pd.Timestamp(cutoff),
                    split_value="validation",
                    execution_plan=selected_plan,
                )
                validation_relations.append(relation)
                temporary_relations.append(relation)

            train_relation = self._combine(train_relations, "TRAIN")
            temporary_relations.append(train_relation)
            validation_relation = self._combine(validation_relations, "VALIDATION")
            temporary_relations.append(validation_relation)
            combined_relation = ""
            if include_combined:
                combined_relation = self._combine(
                    (train_relation, validation_relation), "TRAIN_ALL"
                )
                temporary_relations.append(combined_relation)
            column_types = self._column_types(train_relation)
            exclusions = {
                self.graph_labels.target.lower(),
                self.split_marker.lower(),
                self.cutoff_marker.lower(),
            }
            root_prefix = self.tables[self.root_name].prefix or (
                f"{_slug(self.root_name, 'n0')[:10]}0"
            )
            for key in _key_parts(self.tables[self.root_name].key):
                exclusions.add(key.lower())
                exclusions.add(f"{root_prefix}_{key}".lower())
            feature_columns = tuple(
                column
                for column, data_type in column_types.items()
                if column.lower() not in exclusions
                and data_type.startswith(_NUMERIC_TYPES)
            )
            if not feature_columns:
                raise ValueError(
                    "GraphReduce produced no numeric Snowflake ML feature columns"
                )
            train_rows = int(
                _query_scalar(
                    self.connection,
                    f"SELECT COUNT(*) FROM {_quote_relation(train_relation)}",
                )
                or 0
            )
            validation_rows = int(
                _query_scalar(
                    self.connection,
                    f"SELECT COUNT(*) FROM {_quote_relation(validation_relation)}",
                )
                or 0
            )
            return SnowflakeFeatureSet(
                connection=self.connection,
                train_relation=train_relation,
                validation_relation=validation_relation,
                combined_relation=combined_relation,
                feature_columns=feature_columns,
                execution_plan=selected_plan or {},
                train_rows=train_rows,
                validation_rows=validation_rows,
                temporary_relations=tuple(temporary_relations),
            )
        except Exception:
            SnowflakeFeatureSet(
                self.connection, "", "", "", (), {}, 0, 0, tuple(temporary_relations)
            ).close()
            raise

    def close(self) -> None:
        for relation in reversed(self._source_relations):
            try:
                _execute_no_result(
                    self.connection,
                    f"DROP VIEW IF EXISTS {_quote_relation(relation)}",
                )
            except Exception:
                pass


class SnowflakeMLRunner:
    """Fit, score, and register Snowpark ML XGBoost models."""

    learner = "snowpark_xgboost"
    model_backend = "snowflake_ml_xgboost"
    persistence = "snowflake_model_registry"

    def __init__(self, backend: SnowflakeBackend) -> None:
        self.backend = backend
        self._session: Any = None

    def _imports(self) -> tuple[Any, Any, Any, Any, Any]:
        try:
            from snowflake.ml.modeling import metrics
            from snowflake.ml.modeling.xgboost import XGBClassifier, XGBRegressor
            from snowflake.ml.registry import Registry
            from snowflake.snowpark import Session
        except ImportError as exc:
            missing = getattr(exc, "name", None) or "snowflake.ml"
            raise RuntimeError(
                "Snowflake joint optimization requires the 'snowflake' extra "
                f"(failed import: {missing}). Install kurversc[snowflake]."
            ) from exc
        return Session, XGBClassifier, XGBRegressor, Registry, metrics

    def validate_runtime(self) -> None:
        """Fail before feature search when the optional ML runtime is absent."""

        self._imports()

    @property
    def session(self) -> Any:
        if self._session is None:
            Session, *_ = self._imports()
            self._session = Session.builder.configs(
                {"connection": self.backend.connection}
            ).create()
        return self._session

    @staticmethod
    def _model_params(task: str, values: Mapping[str, Any] | None) -> dict[str, Any]:
        supplied = dict(values or {})
        reserved = {
            "input_cols",
            "label_cols",
            "output_cols",
            "passthrough_cols",
            "drop_input_cols",
        }.intersection(supplied)
        if reserved:
            raise ValueError(
                "Snowflake model_params cannot override KurveRSC-managed fields: "
                f"{sorted(reserved)}"
            )
        defaults: dict[str, Any] = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "random_state": 42,
        }
        defaults.update(supplied)
        if task == "classification":
            defaults.setdefault("eval_metric", "auc")
        return defaults

    def _fit_model(
        self,
        relation: str,
        *,
        target: str,
        feature_columns: Sequence[str],
        task: str,
        model_params: Mapping[str, Any] | None,
    ) -> Any:
        _, XGBClassifier, XGBRegressor, _, _ = self._imports()
        frame = self.session.table(relation).select(*feature_columns, target)
        frame = frame.filter(frame[target].is_not_null()).fillna(
            0, subset=list(feature_columns)
        )
        estimator_class = XGBClassifier if task == "classification" else XGBRegressor
        model = estimator_class(
            input_cols=list(feature_columns),
            label_cols=[target],
            output_cols=["PREDICTION"],
            drop_input_cols=False,
            **self._model_params(task, model_params),
        )
        model.fit(frame)
        return model

    def evaluate(
        self,
        feature_set: SnowflakeFeatureSet,
        *,
        target: str,
        task: str,
        model_params: Mapping[str, Any] | None,
    ) -> SnowflakeEvaluation:
        *_, metrics = self._imports()
        if feature_set.train_rows < 2 or feature_set.validation_rows < 1:
            raise ValueError("Snowflake training and validation relations are empty")
        if task == "classification":
            target_classes = _query_values(
                self.backend.connection,
                f"SELECT DISTINCT {_quote_identifier(target)} "
                f"FROM {_quote_relation(feature_set.train_relation)} "
                f"WHERE {_quote_identifier(target)} IS NOT NULL "
                f"ORDER BY {_quote_identifier(target)}",
            )
            if len(target_classes) != 2:
                raise ValueError(
                    "Snowflake KurveRSC currently requires exactly two training classes"
                )
        started = perf_counter()
        model = self._fit_model(
            feature_set.train_relation,
            target=target,
            feature_columns=feature_set.feature_columns,
            task=task,
            model_params=model_params,
        )
        validation = self.session.table(feature_set.validation_relation).select(
            *feature_set.feature_columns, target
        )
        validation = validation.filter(validation[target].is_not_null()).fillna(
            0, subset=list(feature_set.feature_columns)
        )
        if task == "classification":
            predictions = model.predict_proba(
                validation, output_cols_prefix="KURVERSC_PROBABILITY_"
            )
            original = {column.lower() for column in validation.columns}
            probability_columns = [
                column
                for column in predictions.columns
                if column.lower() not in original
            ]
            if len(probability_columns) < 2:
                raise RuntimeError(
                    "Snowflake ML did not return binary probability columns"
                )
            score = float(
                metrics.roc_auc_score(
                    df=predictions,
                    y_true_col_names=target,
                    y_score_col_names=probability_columns[-1],
                )
            )
            metric = "roc_auc"
        else:
            predictions = model.predict(validation)
            score = float(
                metrics.mean_absolute_error(
                    df=predictions,
                    y_true_col_names=target,
                    y_pred_col_names="PREDICTION",
                )
            )
            metric = "mae"
            target_classes = ()
        return SnowflakeEvaluation(
            metric=metric,
            score=score,
            model_seconds=perf_counter() - started,
            target_classes=target_classes,
        )

    def fit_final(
        self,
        feature_set: SnowflakeFeatureSet,
        *,
        target: str,
        task: str,
        model_name: str,
        validation_score: float,
        metric: str,
        model_params: Mapping[str, Any] | None,
    ) -> SnowflakeModelReference:
        _, _, _, Registry, _ = self._imports()
        model = self._fit_model(
            feature_set.combined_relation,
            target=target,
            feature_columns=feature_set.feature_columns,
            task=task,
            model_params=model_params,
        )
        version = datetime.utcnow().strftime("V%Y%m%d%H%M%S%f")
        registry = Registry(
            session=self.session,
            database_name=self.backend.database,
            schema_name=self.backend.schema,
        )
        registry.log_model(
            model,
            model_name=model_name,
            version_name=version,
            metrics={metric: validation_score},
            comment="KurveRSC Snowflake-native jointly selected model",
        )
        registry.get_model(model_name).default = version
        return SnowflakeModelReference(
            database=self.backend.database,
            schema=self.backend.schema,
            name=model_name,
            version=version,
            task=task,
            feature_columns=tuple(feature_set.feature_columns),
            learner=self.learner,
            model_type=(
                "snowflake.ml.modeling.xgboost.XGBClassifier"
                if task == "classification"
                else "snowflake.ml.modeling.xgboost.XGBRegressor"
            ),
            persistence=self.persistence,
        )


class SnowflakeNativeMLRunner:
    """Fit and score Snowflake's native SQL classification model objects.

    Snowflake does not currently expose a general-purpose tabular regression
    ML Function equivalent to ``SNOWFLAKE.ML.CLASSIFICATION``. Regression is
    therefore intentionally left to another selectable learner instead of
    silently substituting the time-series ``FORECAST`` class.
    """

    learner = "snowflake_native"
    model_backend = "snowflake_ml_classification"
    persistence = "snowflake_classification_object"

    def __init__(self, backend: SnowflakeBackend) -> None:
        self.backend = backend

    def validate_runtime(self) -> None:
        """The native learner uses only the supplied connector connection."""

    @staticmethod
    def _validate_task(task: str) -> None:
        if task != "classification":
            raise ValueError(
                "The Snowflake native SQL learner supports general tabular "
                "classification only. Select learner='snowpark_xgboost' for "
                "regression; Snowflake FORECAST is a time-series learner and "
                "is not interchangeable with tabular regression."
            )

    @staticmethod
    def _validate_model_params(model_params: Mapping[str, Any] | None) -> None:
        if model_params:
            raise ValueError(
                "SNOWFLAKE.ML.CLASSIFICATION selects and tunes its algorithm "
                "internally and does not accept XGBoost model_params"
            )

    def _object_name(self, suffix: str) -> str:
        prefix = _slug(self.backend.artifact_prefix, "kurversc").upper()[:40]
        token = uuid.uuid4().hex[:16].upper()
        return f"{prefix}_{suffix}_{token}"[:250]

    def _fqn(self, name: str) -> str:
        return f"{self.backend.database}.{self.backend.schema}.{name}"

    @staticmethod
    def _model_input(feature_columns: Sequence[str]) -> str:
        return ", ".join(
            f"'{_literal(column)}': {_quote_identifier(column)}"
            for column in feature_columns
        )

    def _create_model(
        self,
        relation: str,
        *,
        target: str,
        feature_columns: Sequence[str],
        model_name: str,
        view_name: str,
    ) -> None:
        columns = ", ".join(
            _quote_identifier(column) for column in (target, *feature_columns)
        )
        view_fqn = self._fqn(view_name)
        model_fqn = self._fqn(model_name)
        _execute_no_result(
            self.backend.connection,
            f"CREATE OR REPLACE VIEW {_quote_relation(view_fqn)} AS "
            f"SELECT {columns} FROM {_quote_relation(relation)} "
            f"WHERE {_quote_identifier(target)} IS NOT NULL",
        )
        _execute_no_result(
            self.backend.connection,
            "CREATE OR REPLACE SNOWFLAKE.ML.CLASSIFICATION "
            f"{_quote_relation(model_fqn)}("
            "INPUT_DATA => SYSTEM$REFERENCE('VIEW', "
            f"'{_literal(_quote_relation(view_fqn))}'), "
            f"TARGET_COLNAME => '{_literal(target)}', "
            "CONFIG_OBJECT => {'evaluate': FALSE})",
        )

    def evaluate(
        self,
        feature_set: SnowflakeFeatureSet,
        *,
        target: str,
        task: str,
        model_params: Mapping[str, Any] | None,
    ) -> SnowflakeEvaluation:
        self._validate_task(task)
        self._validate_model_params(model_params)
        if feature_set.train_rows < 2 or feature_set.validation_rows < 1:
            raise ValueError("Snowflake training and validation relations are empty")
        target_classes = _query_values(
            self.backend.connection,
            f"SELECT DISTINCT TO_VARCHAR({_quote_identifier(target)}) "
            f"FROM {_quote_relation(feature_set.train_relation)} "
            f"WHERE {_quote_identifier(target)} IS NOT NULL "
            f"ORDER BY TO_VARCHAR({_quote_identifier(target)})",
        )
        if len(target_classes) != 2:
            raise ValueError(
                "Snowflake KurveRSC currently requires exactly two training classes"
            )

        model_name = self._object_name("CANDIDATE_MODEL")
        view_name = self._object_name("CANDIDATE_TRAIN_V")
        prediction_name = self._object_name("CANDIDATE_PRED")
        model_fqn = self._fqn(model_name)
        view_fqn = self._fqn(view_name)
        prediction_fqn = self._fqn(prediction_name)
        positive_class = str(target_classes[-1])
        started = perf_counter()
        try:
            self._create_model(
                feature_set.train_relation,
                target=target,
                feature_columns=feature_set.feature_columns,
                model_name=model_name,
                view_name=view_name,
            )
            _execute_no_result(
                self.backend.connection,
                f"CREATE OR REPLACE TEMPORARY TABLE {_quote_relation(prediction_fqn)} AS "
                f"SELECT {_quote_identifier(target)}, "
                f"{_quote_relation(model_fqn)}!PREDICT(INPUT_DATA => "
                f"{{{self._model_input(feature_set.feature_columns)}}}) AS PREDICTION "
                f"FROM {_quote_relation(feature_set.validation_relation)} "
                f"WHERE {_quote_identifier(target)} IS NOT NULL",
            )
            score = _query_scalar(
                self.backend.connection,
                "WITH SCORED AS ("
                f"SELECT IFF(TO_VARCHAR({_quote_identifier(target)}) = "
                f"'{_literal(positive_class)}', 1, 0) AS Y, "
                "GET_IGNORE_CASE(GET_IGNORE_CASE(PREDICTION, 'probability'), "
                f"'{_literal(positive_class)}')::FLOAT AS SCORE "
                f"FROM {_quote_relation(prediction_fqn)}), "
                "RANKED AS (SELECT Y, SCORE, "
                "RANK() OVER (ORDER BY SCORE) + "
                "(COUNT(*) OVER (PARTITION BY SCORE) - 1) / 2.0 AS AVG_RANK "
                "FROM SCORED WHERE SCORE IS NOT NULL), "
                "STATS AS (SELECT SUM(IFF(Y = 1, AVG_RANK, 0)) AS POS_RANKS, "
                "SUM(Y) AS POSITIVES, COUNT(*) - SUM(Y) AS NEGATIVES FROM RANKED) "
                "SELECT (POS_RANKS - POSITIVES * (POSITIVES + 1) / 2.0) / "
                "NULLIF(POSITIVES * NEGATIVES, 0) FROM STATS",
            )
            if score is None:
                raise ValueError(
                    "Snowflake native classification could not calculate ROC AUC "
                    "for the validation relation"
                )
            return SnowflakeEvaluation(
                metric="roc_auc",
                score=float(score),
                model_seconds=perf_counter() - started,
                target_classes=target_classes,
            )
        finally:
            for statement in (
                f"DROP TABLE IF EXISTS {_quote_relation(prediction_fqn)}",
                "DROP SNOWFLAKE.ML.CLASSIFICATION IF EXISTS "
                f"{_quote_relation(model_fqn)}",
                f"DROP VIEW IF EXISTS {_quote_relation(view_fqn)}",
            ):
                try:
                    _execute_no_result(self.backend.connection, statement)
                except Exception:
                    pass

    def fit_final(
        self,
        feature_set: SnowflakeFeatureSet,
        *,
        target: str,
        task: str,
        model_name: str,
        validation_score: float,
        metric: str,
        model_params: Mapping[str, Any] | None,
    ) -> SnowflakeModelReference:
        del validation_score, metric
        self._validate_task(task)
        self._validate_model_params(model_params)
        view_name = self._object_name("FINAL_TRAIN_V")
        view_fqn = self._fqn(view_name)
        try:
            self._create_model(
                feature_set.combined_relation,
                target=target,
                feature_columns=feature_set.feature_columns,
                model_name=model_name,
                view_name=view_name,
            )
        finally:
            try:
                _execute_no_result(
                    self.backend.connection,
                    f"DROP VIEW IF EXISTS {_quote_relation(view_fqn)}",
                )
            except Exception:
                pass
        return SnowflakeModelReference(
            database=self.backend.database,
            schema=self.backend.schema,
            name=model_name,
            version="",
            task=task,
            feature_columns=tuple(feature_set.feature_columns),
            learner=self.learner,
            model_type="SNOWFLAKE.ML.CLASSIFICATION",
            persistence=self.persistence,
        )


SNOWFLAKE_LEARNERS = ("snowflake_native", "snowpark_xgboost")


def snowflake_model_runner(
    backend: SnowflakeBackend, learner: str = "snowpark_xgboost"
) -> SnowflakeNativeMLRunner | SnowflakeMLRunner:
    """Create a Snowflake model runner from a stable public learner name."""

    aliases = {
        "native": "snowflake_native",
        "snowflake_sql": "snowflake_native",
        "snowflake_native": "snowflake_native",
        "xgboost": "snowpark_xgboost",
        "snowpark": "snowpark_xgboost",
        "snowpark_xgboost": "snowpark_xgboost",
    }
    normalized = aliases.get(str(learner).strip().lower())
    if normalized is None:
        raise ValueError(
            f"Unknown Snowflake learner {learner!r}; choose one of "
            f"{', '.join(SNOWFLAKE_LEARNERS)}"
        )
    if normalized == "snowflake_native":
        return SnowflakeNativeMLRunner(backend)
    return SnowflakeMLRunner(backend)


def _default_model_name(backend: SnowflakeBackend, root_name: str) -> str:
    prefix = _slug(backend.artifact_prefix, "kurversc").upper()[:40]
    root = _slug(root_name, "root").upper()[:80]
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{root}_{timestamp}_MODEL"[:250]


def fit_snowflake(
    parent_node: Table | Source,
    label_node: GraphLabels,
    *,
    backend: SnowflakeBackend,
    tables: Sequence[Table | Source] | Mapping[str, Table | Source] = (),
    relationships: Sequence[Relationship | Mapping[str, Any]] = (),
    parent_key: Key | Sequence[str] | None = None,
    parent_date: str | None = None,
    parent_timeless: bool = False,
    task: str = "auto",
    max_depth: int = 3,
    feature_family_stages: Sequence[Sequence[str]] = DEFAULT_FAMILY_STAGES,
    auto_annotate_options: Sequence[bool] = (True, False),
    feature_family_max_columns: int | None = 4,
    feature_family_max_column_options: Sequence[int | None] | None = None,
    feature_family_max_features_per_column: int | None = 32,
    feature_propagation_max_functions_per_column: int | None = 1,
    forward_search_beam_width: int = 2,
    compute_period_days: int = 3650,
    sample_rows: int | None = 100_000,
    graph_configs: Sequence[GraphConfig] | None = None,
    infer_ts_periods: bool = False,
    auto_text_features: bool = False,
    auto_annotate_max_text_columns: int | None = None,
    model_params: Mapping[str, Any] | None = None,
    model_name: str | None = None,
    learner: str = "snowpark_xgboost",
    continue_on_error: bool = True,
    verbose: bool = False,
    classification_negligible_gain: float = 0.002,
    regression_negligible_relative_gain: float = 0.005,
    drastic_feature_growth: float = 2.0,
    complexity_uncertainty_multiplier: float = 1.0,
    adaptive_depth_promotion: bool = True,
    model_runner: SnowflakeNativeMLRunner | SnowflakeMLRunner | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> FitResult:
    """Jointly select a GraphReduce plan and Snowflake ML model in Snowflake.

    Candidate feature relations are temporary session objects. Every candidate
    is trained and validated by the selected learner in the warehouse; after
    selection, the recommended plan is replayed and its refit model is
    persisted as the learner's native Snowflake object. The current Snowflake
    path accepts :class:`GraphLabels`, which preserves point-in-time label
    generation without transferring label frames locally.
    """

    if not isinstance(label_node, GraphLabels):
        raise TypeError("fit_snowflake currently requires label_node=GraphLabels(...)")
    if sample_rows is not None and sample_rows < 1:
        raise ValueError("sample_rows must be positive or None")
    if forward_search_beam_width < 1:
        raise ValueError("forward_search_beam_width must be positive")
    if compute_period_days < 1:
        raise ValueError("compute_period_days must be positive")

    parent = coerce_table(
        parent_node,
        key=parent_key,
        date=parent_date,
        timeless=parent_timeless,
    )
    if parent.key is None:
        raise ValueError("parent_key is required (directly or in Table)")
    if parent.date is None and not parent.timeless:
        raise ValueError(
            "parent_date is required for GraphLabels unless the root is timeless"
        )
    root_name, normalized_tables = _normalize_tables(parent, tables)
    normalized_relationships = tuple(
        coerce_relationship(item) for item in relationships
    )
    if label_node.table not in normalized_tables:
        raise ValueError(
            f"GraphLabels table {label_node.table!r} must appear in tables"
        )

    resolved_task = (
        "classification"
        if task == "auto" and label_node.operation.lower() == "bool"
        else "regression"
        if task == "auto"
        else task.lower()
    )
    if resolved_task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification', 'regression', or 'auto'")
    column_budgets = (
        tuple(
            dict.fromkeys(config.feature_family_max_columns for config in graph_configs)
        )
        if graph_configs is not None
        else resolve_feature_family_column_budgets(
            feature_family_max_columns,
            feature_family_max_column_options,
        )
    )
    configs = (
        tuple(graph_configs)
        if graph_configs is not None
        else incremental_configs(
            max_depth=max_depth,
            feature_family_stages=feature_family_stages,
            auto_annotate_options=auto_annotate_options,
            feature_family_max_columns=feature_family_max_columns,
            feature_family_max_column_options=feature_family_max_column_options,
            feature_family_max_features_per_column=(
                feature_family_max_features_per_column
            ),
            feature_propagation_max_functions_per_column=(
                feature_propagation_max_functions_per_column
            ),
        )
    )
    if not configs:
        raise ValueError("graph_configs must contain at least one configuration")
    if auto_text_features or auto_annotate_max_text_columns is not None:
        configs = tuple(
            replace(
                config,
                auto_text_features=auto_text_features,
                auto_annotate_max_text_columns=auto_annotate_max_text_columns,
            )
            for config in configs
        )

    def report_progress(event: str, **details: Any) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback({"event": event, **details})
        except Exception as exc:
            # Progress reporting is observational. A transient status-store
            # failure must not discard a warehouse optimization that can
            # otherwise complete successfully.
            logger.warning(
                "snowflake_progress_callback_failed",
                event=event,
                error=f"{type(exc).__name__}: {exc}",
            )

    runner = model_runner or snowflake_model_runner(backend, learner)
    if model_runner is None:
        runner.validate_runtime()
    materializer = SnowflakeGraphMaterializer(
        backend,
        normalized_tables,
        normalized_relationships,
        root_name,
        label_node,
        compute_period_days=compute_period_days,
        sample_rows=sample_rows,
        infer_ts_periods=infer_ts_periods,
        verbose=verbose,
    )
    trials: list[Trial] = []
    target_classes_by_config: dict[GraphConfig, tuple[Any, ...]] = {}
    logger.info(
        "snowflake_search_started",
        candidates=len(configs),
        task=resolved_task,
        database=backend.database,
        schema=backend.schema,
        warehouse=backend.warehouse,
        learner=getattr(runner, "learner", learner),
        sample_rows=sample_rows,
    )
    report_progress(
        "search_started",
        total_candidates=len(configs),
        task=resolved_task,
        metric="roc_auc" if resolved_task == "classification" else "mae",
        learner=getattr(runner, "learner", learner),
    )
    try:
        for trial_number, config in enumerate(configs, start=1):
            if graph_configs is None and not forward_candidate_allowed(
                config,
                trials,
                beam_width=forward_search_beam_width,
                feature_family_max_column_options=column_budgets,
            ):
                continue
            if (
                graph_configs is None
                and adaptive_depth_promotion
                and not adaptive_depth_candidate_allowed(
                    config,
                    trials,
                    classification_gain=classification_negligible_gain,
                    regression_relative_gain=regression_negligible_relative_gain,
                    uncertainty_multiplier=complexity_uncertainty_multiplier,
                    beam_width=1,
                )
            ):
                continue
            feature_set: SnowflakeFeatureSet | None = None
            started = perf_counter()
            logger.info(
                "snowflake_trial_started",
                trial=f"{trial_number}/{len(configs)}",
                feature_families=config.feature_families,
                depth=config.depth,
                auto_annotate_features=config.auto_annotate_features,
            )
            report_progress(
                "trial_started",
                trial_number=trial_number,
                total_candidates=len(configs),
                feature_families=list(config.feature_families),
                depth=config.depth,
                auto_annotate_features=config.auto_annotate_features,
            )
            try:
                feature_set = materializer.materialize(config)
                feature_seconds = perf_counter() - started
                evaluation = runner.evaluate(
                    feature_set,
                    target=label_node.target,
                    task=resolved_task,
                    model_params=model_params,
                )
                target_classes_by_config[config] = evaluation.target_classes
                trials.append(
                    Trial(
                        config=config,
                        metric=evaluation.metric,
                        validation_score=evaluation.score,
                        objective_score=(
                            evaluation.score
                            if evaluation.metric == "roc_auc"
                            else -evaluation.score
                        ),
                        feature_count=len(feature_set.feature_columns),
                        train_rows=feature_set.train_rows,
                        validation_rows=feature_set.validation_rows,
                        feature_seconds=feature_seconds,
                        model_seconds=evaluation.model_seconds,
                        feature_columns=feature_set.feature_columns,
                        execution_plan=copy.deepcopy(feature_set.execution_plan),
                    )
                )
                logger.info(
                    "snowflake_trial_completed",
                    trial=f"{trial_number}/{len(configs)}",
                    metric=evaluation.metric,
                    score=evaluation.score,
                    features=len(feature_set.feature_columns),
                    train_rows=feature_set.train_rows,
                    validation_rows=feature_set.validation_rows,
                )
                report_progress(
                    "trial_completed",
                    trial_number=trial_number,
                    total_candidates=len(configs),
                    status="completed",
                    metric=evaluation.metric,
                    score=evaluation.score,
                    feature_count=len(feature_set.feature_columns),
                    train_rows=feature_set.train_rows,
                    validation_rows=feature_set.validation_rows,
                )
            except Exception as exc:
                trials.append(
                    Trial(
                        config=config,
                        metric="roc_auc"
                        if resolved_task == "classification"
                        else "mae",
                        validation_score=float("nan"),
                        objective_score=float("-inf"),
                        feature_count=0,
                        train_rows=0,
                        validation_rows=0,
                        feature_seconds=perf_counter() - started,
                        model_seconds=0.0,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                logger.warning(
                    "snowflake_trial_failed",
                    trial=f"{trial_number}/{len(configs)}",
                    error=f"{type(exc).__name__}: {exc}",
                )
                report_progress(
                    "trial_completed",
                    trial_number=trial_number,
                    total_candidates=len(configs),
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                if not continue_on_error:
                    raise
            finally:
                if feature_set is not None:
                    feature_set.close()

        successful = [trial for trial in trials if trial.status == "completed"]
        if not successful:
            details = "; ".join(trial.error or "unknown error" for trial in trials[:3])
            raise RuntimeError(f"Every Snowflake KurveRSC trial failed. {details}")
        annotate_complexity(
            trials,
            classification_gain=classification_negligible_gain,
            regression_relative_gain=regression_negligible_relative_gain,
            feature_growth=drastic_feature_growth,
            uncertainty_multiplier=complexity_uncertainty_multiplier,
        )
        best, recommended = _select_trials(
            successful,
            classification_negligible_gain=classification_negligible_gain,
            regression_negligible_relative_gain=regression_negligible_relative_gain,
            complexity_uncertainty_multiplier=complexity_uncertainty_multiplier,
        )
        logger.info(
            "snowflake_search_selected",
            best_config=best.config,
            recommended_config=recommended.config,
            metric=recommended.metric,
            score=recommended.validation_score,
        )
        report_progress(
            "plan_selected",
            completed_trials=len(successful),
            attempted_trials=len(trials),
            metric=recommended.metric,
            score=recommended.validation_score,
            feature_families=list(recommended.config.feature_families),
            depth=recommended.config.depth,
            feature_count=recommended.feature_count,
        )

        # Search aliases may be sampled. Reopen the sources without a LIMIT so
        # the selected frozen plan and final Snowflake model use full data.
        materializer.close()
        final_materializer = SnowflakeGraphMaterializer(
            backend,
            normalized_tables,
            normalized_relationships,
            root_name,
            label_node,
            compute_period_days=compute_period_days,
            sample_rows=None,
            infer_ts_periods=False,
            verbose=verbose,
        )
        try:
            final_features = final_materializer.materialize(
                recommended.config,
                execution_plan=recommended.execution_plan,
                include_combined=True,
            )
            try:
                selected_model_name = model_name or _default_model_name(
                    backend, root_name
                )
                if len(selected_model_name) > 255 or not _SIMPLE_IDENTIFIER.fullmatch(
                    selected_model_name
                ):
                    raise ValueError("model_name must be a simple Snowflake identifier")
                selected_model_name = selected_model_name.upper()
                report_progress(
                    "final_refit_started",
                    model_name=selected_model_name,
                    feature_count=len(final_features.feature_columns),
                )
                model_reference = runner.fit_final(
                    final_features,
                    target=label_node.target,
                    task=resolved_task,
                    model_name=selected_model_name,
                    validation_score=recommended.validation_score,
                    metric=recommended.metric,
                    model_params=model_params,
                )
                plan = copy.deepcopy(final_features.execution_plan)
                plan["kurversc_feature_schema"] = {
                    "columns": tuple(final_features.feature_columns),
                    "categorical_columns": (),
                    "datetime_columns": (),
                }
                plan["kurversc_snowflake"] = {
                    "execution_backend": "snowflake",
                    "learner": getattr(model_reference, "learner", "snowpark_xgboost"),
                    "model_backend": getattr(
                        runner, "model_backend", "snowflake_ml_xgboost"
                    ),
                    "model_type": getattr(model_reference, "model_type", ""),
                    "model_persistence": getattr(
                        model_reference, "persistence", "snowflake_model_registry"
                    ),
                    "database": backend.database,
                    "schema": backend.schema,
                    "warehouse": backend.warehouse,
                    "model_name": model_reference.name,
                    "model_version": model_reference.version,
                    "model_fqn": model_reference.fqn,
                }
                fitted = FittedModel(
                    config=recommended.config,
                    execution_plan=plan,
                    plan_fingerprint=_execution_plan_fingerprint(plan),
                    estimator=model_reference,
                    validation_estimator=None,
                    feature_columns=tuple(final_features.feature_columns),
                    categorical_columns=(),
                    datetime_columns=(),
                    target=label_node.target,
                    task=resolved_task,
                    metric=recommended.metric,
                    validation_score=recommended.validation_score,
                    train_rows=(
                        final_features.train_rows + final_features.validation_rows
                    ),
                    validation_rows=recommended.validation_rows,
                    training_frames=(
                        len(label_node.train_cutoffs)
                        + len(label_node.validation_cutoffs)
                    ),
                    model_backend="snowflake_ml",
                    target_classes=(
                        target_classes_by_config.get(recommended.config, ())
                    ),
                )
                logger.info(
                    "snowflake_model_registered",
                    model_fqn=model_reference.fqn,
                    model_version=model_reference.version,
                    features=len(final_features.feature_columns),
                )
                report_progress(
                    "completed",
                    model_fqn=model_reference.fqn,
                    model_version=model_reference.version,
                    metric=recommended.metric,
                    score=recommended.validation_score,
                    feature_count=len(final_features.feature_columns),
                    learner=getattr(model_reference, "learner", ""),
                    model_type=getattr(model_reference, "model_type", ""),
                )
            finally:
                final_features.close()
        finally:
            final_materializer.close()
        return FitResult(
            task=resolved_task,
            metric=recommended.metric,
            best_trial=best,
            recommended_trial=recommended,
            trials=tuple(trials),
            fitted_model=fitted,
        )
    finally:
        materializer.close()
