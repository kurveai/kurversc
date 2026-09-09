from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

import kurversc
from kurversc import snowflake as snowflake_module


class _Cursor:
    def __init__(self, statements: list[str]) -> None:
        self.statements = statements

    def execute(self, sql: str):
        self.statements.append(sql)
        return self

    def close(self) -> None:
        return None


class _Connection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self.statements)


class _TextSampleCursor(_Cursor):
    description = (("POST_ID",), ("POST_STATUS",), ("POSTS_BODY",))

    def fetchall(self):
        return [
            (
                1,
                "question",
                "<p>This is a long post body and must not become a category.</p>",
            ),
            (
                2,
                "answer",
                "<p>This is another long post body that should be treated as text.</p>",
            ),
        ]


class _TextSampleConnection(_Connection):
    def cursor(self) -> _TextSampleCursor:
        return _TextSampleCursor(self.statements)


@dataclass
class _FakeMaterializer:
    backend: kurversc.SnowflakeBackend
    tables: object
    relationships: object
    root_name: str
    graph_labels: kurversc.GraphLabels
    compute_period_days: int
    sample_rows: int | None
    infer_ts_periods: bool
    verbose: bool

    instances: ClassVar[list["_FakeMaterializer"]] = []

    def __post_init__(self) -> None:
        self.materializations = []
        self.closed = False
        self.instances.append(self)

    def materialize(self, config, *, execution_plan=None, include_combined=False):
        self.materializations.append((config, execution_plan, include_combined))
        feature_count = 4 if "temporal" in config.feature_families else 2
        plan = dict(
            execution_plan or {"records": [], "candidate": config.feature_families}
        )
        return snowflake_module.SnowflakeFeatureSet(
            connection=self.backend.connection,
            train_relation="TRAIN_FRAME",
            validation_relation="VALIDATION_FRAME",
            combined_relation="TRAIN_ALL_FRAME",
            feature_columns=tuple(f"FEATURE_{index}" for index in range(feature_count)),
            execution_plan=plan,
            train_rows=100,
            validation_rows=25,
            temporary_relations=(),
        )

    def close(self) -> None:
        self.closed = True


class _FakeRunner:
    def __init__(self, backend: kurversc.SnowflakeBackend) -> None:
        self.backend = backend
        self.evaluations = []
        self.final_calls = []

    def evaluate(self, feature_set, *, target, task, model_params):
        self.evaluations.append((feature_set, target, task, model_params))
        score = 0.78 if len(feature_set.feature_columns) == 4 else 0.70
        return kurversc.SnowflakeEvaluation(
            metric="roc_auc",
            score=score,
            model_seconds=1.25,
            target_classes=(0, 1),
        )

    def fit_final(
        self,
        feature_set,
        *,
        target,
        task,
        model_name,
        validation_score,
        metric,
        model_params,
    ):
        self.final_calls.append(
            (
                feature_set,
                target,
                task,
                model_name,
                validation_score,
                metric,
                model_params,
            )
        )
        return kurversc.SnowflakeModelReference(
            database=self.backend.database,
            schema=self.backend.schema,
            name=model_name,
            version="V1",
            task=task,
            feature_columns=feature_set.feature_columns,
        )


def _problem():
    parent = kurversc.Table(
        "RAW.CUSTOMERS",
        name="customers",
        key="CUSTOMER_ID",
        timeless=True,
    )
    events = kurversc.Table(
        "RAW.EVENTS",
        name="events",
        key="EVENT_ID",
        date="EVENT_AT",
    )
    labels = kurversc.GraphLabels(
        table="events",
        field="EVENT_ID",
        operation="bool",
        train_cutoffs=("2026-01-01",),
        validation_cutoffs=("2026-02-01",),
        target="WILL_EVENT",
    )
    relationship = kurversc.Relationship(
        parent="customers",
        child="events",
        parent_key="CUSTOMER_ID",
        child_key="CUSTOMER_ID",
    )
    return parent, events, labels, relationship


def test_snowflake_materializer_detects_free_text_from_bounded_sample() -> None:
    materializer = object.__new__(snowflake_module.SnowflakeGraphMaterializer)
    materializer.connection = _TextSampleConnection()
    materializer.views = {"posts": "KURVERSC_POSTS_SOURCE"}
    materializer._free_text_columns = {}

    assert materializer.free_text_columns("posts") == frozenset({"POSTS_BODY"})
    assert materializer.free_text_columns("posts") == frozenset({"POSTS_BODY"})
    assert materializer.connection.statements == [
        'SELECT * FROM "KURVERSC_POSTS_SOURCE" LIMIT 500'
    ]


def test_fit_snowflake_jointly_selects_plan_and_registered_model(monkeypatch) -> None:
    _FakeMaterializer.instances.clear()
    monkeypatch.setattr(
        snowflake_module, "SnowflakeGraphMaterializer", _FakeMaterializer
    )
    connection = _Connection()
    backend = kurversc.SnowflakeBackend(
        connection=connection,
        database="KURVE",
        schema="OUTPUT",
        warehouse="SNOWPARK_WH",
    )
    runner = _FakeRunner(backend)
    parent, events, labels, relationship = _problem()
    configs = (
        kurversc.GraphConfig(
            feature_families=("base",), depth=1, auto_annotate_features=False
        ),
        kurversc.GraphConfig(
            feature_families=("base", "temporal"),
            depth=1,
            auto_annotate_features=False,
        ),
    )
    progress_events = []

    result = kurversc.fit_snowflake(
        parent,
        labels,
        backend=backend,
        tables=[events],
        relationships=[relationship],
        graph_configs=configs,
        model_name="CUSTOMER_EVENT_MODEL",
        model_runner=runner,
        progress_callback=progress_events.append,
    )

    assert result.best_config == configs[1]
    assert result.recommended_config == configs[1]
    assert result.full_validation_score == pytest.approx(0.78)
    assert result.fitted_model.model_backend == "snowflake_ml"
    assert result.fitted_model.target_classes == (0, 1)
    assert result.fitted_model.train_rows == 125
    assert result.fitted_model.estimator.fqn == "KURVE.OUTPUT.CUSTOMER_EVENT_MODEL"
    assert result.execution_plan["kurversc_snowflake"] == {
        "execution_backend": "snowflake",
        "learner": "snowpark_xgboost",
        "model_backend": "snowflake_ml_xgboost",
        "model_type": "snowflake.ml.modeling.xgboost",
        "model_persistence": "snowflake_model_registry",
        "database": "KURVE",
        "schema": "OUTPUT",
        "warehouse": "SNOWPARK_WH",
        "model_name": "CUSTOMER_EVENT_MODEL",
        "model_version": "V1",
        "model_fqn": "KURVE.OUTPUT.CUSTOMER_EVENT_MODEL",
    }
    assert len(runner.evaluations) == 2
    assert len(runner.final_calls) == 1
    search_materializer, final_materializer = _FakeMaterializer.instances
    assert search_materializer.sample_rows == 100_000
    assert all(not call[2] for call in search_materializer.materializations)
    assert search_materializer.closed is True
    assert final_materializer.sample_rows is None
    assert final_materializer.materializations[-1][1] == (
        result.recommended_trial.execution_plan
    )
    assert final_materializer.materializations[-1][2] is True
    assert final_materializer.closed is True
    assert progress_events[0] == {
        "event": "search_started",
        "total_candidates": 2,
        "task": "classification",
        "metric": "roc_auc",
        "learner": "snowpark_xgboost",
    }
    assert [event["event"] for event in progress_events].count("trial_started") == 2
    assert [event["event"] for event in progress_events].count("trial_completed") == 2
    assert progress_events[-2]["event"] == "final_refit_started"
    assert progress_events[-1] == {
        "event": "completed",
        "model_fqn": "KURVE.OUTPUT.CUSTOMER_EVENT_MODEL",
        "model_version": "V1",
        "metric": "roc_auc",
        "score": pytest.approx(0.78),
        "feature_count": 4,
        "learner": "snowpark_xgboost",
        "model_type": "snowflake.ml.modeling.xgboost",
    }


def test_fit_snowflake_requires_graph_labels() -> None:
    backend = kurversc.SnowflakeBackend(
        connection=_Connection(), database="KURVE", schema="OUTPUT"
    )
    with pytest.raises(TypeError, match="GraphLabels"):
        kurversc.fit_snowflake(
            kurversc.Table("RAW.CUSTOMERS", name="customers", key="ID", timeless=True),
            kurversc.Labels("RAW.LABELS", target="LABEL", key="ID"),
            backend=backend,
        )


@pytest.mark.parametrize("field", ["database", "schema", "warehouse"])
def test_snowflake_backend_rejects_unsafe_identifiers(field) -> None:
    values = {
        "connection": _Connection(),
        "database": "KURVE",
        "schema": "OUTPUT",
        "warehouse": "SNOWPARK_WH",
    }
    values[field] = "bad; drop table x"
    with pytest.raises(ValueError, match=field):
        kurversc.SnowflakeBackend(**values)


def test_snowflake_dependencies_are_lazy() -> None:
    assert kurversc.SnowflakeMLRunner is not None


def test_snowflake_backend_uses_unquoted_identifier_case() -> None:
    backend = kurversc.SnowflakeBackend(
        connection=_Connection(),
        database="kurve",
        schema="output",
        warehouse="snowpark_wh",
    )
    assert (backend.database, backend.schema, backend.warehouse) == (
        "KURVE",
        "OUTPUT",
        "SNOWPARK_WH",
    )


def test_snowflake_relation_quoting_preserves_unquoted_case_semantics() -> None:
    assert snowflake_module._quote_relation("raw.customer_events") == (
        '"RAW"."CUSTOMER_EVENTS"'
    )


def test_snowflake_model_params_protect_managed_columns() -> None:
    with pytest.raises(ValueError, match="input_cols"):
        kurversc.SnowflakeMLRunner._model_params(
            "regression", {"input_cols": ["UNSAFE"]}
        )


def test_snowflake_model_runner_selects_named_learners() -> None:
    backend = kurversc.SnowflakeBackend(
        connection=_Connection(), database="KURVE", schema="OUTPUT"
    )
    assert isinstance(
        kurversc.snowflake_model_runner(backend, "snowflake_native"),
        kurversc.SnowflakeNativeMLRunner,
    )
    assert isinstance(
        kurversc.snowflake_model_runner(backend, "snowpark_xgboost"),
        kurversc.SnowflakeMLRunner,
    )
    with pytest.raises(ValueError, match="Unknown Snowflake learner"):
        kurversc.snowflake_model_runner(backend, "mystery")


def test_native_snowflake_learner_scores_explicit_validation_relation(
    monkeypatch,
) -> None:
    connection = _Connection()
    backend = kurversc.SnowflakeBackend(
        connection=connection, database="KURVE", schema="OUTPUT"
    )
    runner = kurversc.SnowflakeNativeMLRunner(backend)
    feature_set = snowflake_module.SnowflakeFeatureSet(
        connection=connection,
        train_relation="TRAIN_FRAME",
        validation_relation="VALIDATION_FRAME",
        combined_relation="COMBINED_FRAME",
        feature_columns=("FEATURE_A", "FEATURE_B"),
        execution_plan={},
        train_rows=100,
        validation_rows=20,
        temporary_relations=(),
    )
    monkeypatch.setattr(
        snowflake_module, "_query_values", lambda connection, sql: ("0", "1")
    )
    monkeypatch.setattr(snowflake_module, "_query_scalar", lambda connection, sql: 0.82)

    evaluation = runner.evaluate(
        feature_set,
        target="TARGET",
        task="classification",
        model_params=None,
    )

    assert evaluation.metric == "roc_auc"
    assert evaluation.score == pytest.approx(0.82)
    statements = "\n".join(connection.statements)
    assert "CREATE OR REPLACE SNOWFLAKE.ML.CLASSIFICATION" in statements
    assert "CONFIG_OBJECT => {'evaluate': FALSE}" in statements
    assert "!PREDICT(INPUT_DATA =>" in statements
    assert "DROP SNOWFLAKE.ML.CLASSIFICATION IF EXISTS" in statements


def test_native_snowflake_learner_rejects_general_regression() -> None:
    backend = kurversc.SnowflakeBackend(
        connection=_Connection(), database="KURVE", schema="OUTPUT"
    )
    runner = kurversc.SnowflakeNativeMLRunner(backend)
    feature_set = snowflake_module.SnowflakeFeatureSet(
        connection=backend.connection,
        train_relation="TRAIN_FRAME",
        validation_relation="VALIDATION_FRAME",
        combined_relation="COMBINED_FRAME",
        feature_columns=("FEATURE",),
        execution_plan={},
        train_rows=10,
        validation_rows=2,
        temporary_relations=(),
    )
    with pytest.raises(ValueError, match="tabular classification only"):
        runner.evaluate(
            feature_set,
            target="TARGET",
            task="regression",
            model_params=None,
        )


def test_snowflake_feature_set_drops_temporary_relations() -> None:
    connection = _Connection()
    feature_set = snowflake_module.SnowflakeFeatureSet(
        connection=connection,
        train_relation="TEMP_TRAIN",
        validation_relation="TEMP_VALIDATION",
        combined_relation="",
        feature_columns=("FEATURE",),
        execution_plan={},
        train_rows=1,
        validation_rows=1,
        temporary_relations=("temp_train", "temp_validation"),
    )
    feature_set.close()
    assert connection.statements == [
        'DROP TABLE IF EXISTS "TEMP_VALIDATION"',
        'DROP TABLE IF EXISTS "TEMP_TRAIN"',
    ]


def test_snowflake_temporary_names_leave_room_for_graphreduce_suffixes() -> None:
    materializer = object.__new__(kurversc.SnowflakeGraphMaterializer)
    materializer.backend = kurversc.SnowflakeBackend(
        connection=_Connection(),
        database="KURVE",
        schema="OUTPUT",
        artifact_prefix="KURVERSC_LONG_ARTIFACT_NAMESPACE",
    )
    materializer._sequence = 0
    first = materializer._temporary_name("SRC_" + "VERY_LONG_TABLE_NAME_" * 20)
    second = materializer._temporary_name("SRC_" + "VERY_LONG_TABLE_NAME_" * 20)
    assert len(first) <= 120
    assert first != second
