from datetime import datetime, timedelta
from types import SimpleNamespace

import duckdb
import pytest

from graphreduce.enum import SQLOpType
from graphreduce.models import sqlop

from kurversc.plan_dates import atomic_date_replay, atomic_rebind, replace_dates_once
from kurversc.plan_dates import apply_graph_plan, atomic_execution_plan


def test_atomic_replacements_do_not_cascade():
    assert (
        replace_dates_once("old recent", {"old": "recent", "recent": "ancient"})
        == "recent ancient"
    )


def test_replay_when_current_cutoff_equals_old_lookback_boundary():
    original = datetime(2019, 1, 1)
    current = original - timedelta(days=3650)
    node = SimpleNamespace(
        prefix="root",
        compute_period_minutes=lambda: 3650 * 1440,
        ts_periods=[],
        base_predicate_windows=[],
    )
    graph = SimpleNamespace(cut_date=current)
    expression = f"date <= '{original}' AND date >= '{current}'"
    ops = [sqlop(optype=SQLOpType.where, opval=expression)]
    result = atomic_rebind(graph, node, ops, original)
    assert (
        result[0].opval
        == f"date <= '{current}' AND date >= '{current - timedelta(days=3650)}'"
    )
    assert ops[0].opval == expression


def test_forward_replay_cannot_move_feature_cutoff_into_a_future_label_window():
    original = datetime(2024, 1, 1)
    current = original + timedelta(days=1)
    node = SimpleNamespace(
        prefix="root",
        label_period_val=1,
        label_period_minutes=lambda: 1440,
        ts_periods=[],
        base_predicate_windows=[],
    )
    graph = SimpleNamespace(cut_date=current)
    operations = [
        sqlop(optype=SQLOpType.where, opval=f"date < '{original}'"),
        sqlop(
            optype=SQLOpType.where, opval=f"date >= '{original}' AND date < '{current}'"
        ),
    ]
    rebound = atomic_rebind(graph, node, operations, original)
    assert rebound[0].opval == f"date < '{current}'"
    assert (
        rebound[1].opval
        == f"date >= '{current}' AND date < '{current + timedelta(days=1)}'"
    )


def test_exact_root_and_shifted_graph_cutoffs_share_one_pass():
    exact = datetime(2019, 1, 1)
    shifted = exact + timedelta(microseconds=1)
    current = exact - timedelta(days=30)
    node = SimpleNamespace(prefix="root", ts_periods=[30], base_predicate_windows=[])
    graph = SimpleNamespace(
        cut_date=current + timedelta(microseconds=1),
        _kurversc_atomic_root_dates=("root", exact, current),
    )
    ops = [
        sqlop(
            optype=SQLOpType.where,
            opval=f"root <= '{exact}' AND child < '{shifted}' AND child > '{shifted - timedelta(days=30)}'",
        )
    ]
    result = atomic_rebind(graph, node, ops, shifted)
    assert (
        result[0].opval
        == f"root <= '{current}' AND child < '{current + timedelta(microseconds=1)}' AND child > '{current + timedelta(microseconds=1) - timedelta(days=30)}'"
    )


def test_patch_restores_original_methods():
    import kurversc.core as core
    from graphreduce.graph_reduce import GraphReduce

    original = (
        core._prepare_execution_plan,
        GraphReduce.apply_execution_plan,
        GraphReduce._rebind_sqlop_dates_for_node,
    )
    with atomic_date_replay():
        assert GraphReduce._rebind_sqlop_dates_for_node is atomic_rebind
    assert original == (
        core._prepare_execution_plan,
        GraphReduce.apply_execution_plan,
        GraphReduce._rebind_sqlop_dates_for_node,
    )


def test_collision_safe_replay_preserves_a_real_sql_window():
    original = datetime(2019, 1, 1)
    current = original - timedelta(days=3650)
    node = SimpleNamespace(
        prefix="root",
        compute_period_minutes=lambda: 3650 * 1440,
        ts_periods=[30],
        base_predicate_windows=[],
    )
    graph = SimpleNamespace(cut_date=current)
    expression = f"date < TIMESTAMP '{original}' AND date >= TIMESTAMP '{current}'"
    operation = sqlop(optype=SQLOpType.where, opval=expression)
    predicate = atomic_rebind(graph, node, [operation], original)[0].opval
    with duckdb.connect() as connection:
        rows = connection.execute(
            "SELECT date FROM (VALUES (?::TIMESTAMP), (?::TIMESTAMP), "
            "(?::TIMESTAMP), (?::TIMESTAMP)) AS events(date) WHERE "
            + predicate
            + " ORDER BY date",
            [
                current - timedelta(days=3650, microseconds=1),
                current - timedelta(days=3650),
                current - timedelta(microseconds=1),
                current,
            ],
        ).fetchall()
    assert rows == [
        (current - timedelta(days=3650),),
        (current - timedelta(microseconds=1),),
    ]


def test_patch_restores_original_methods_after_failure():
    import kurversc.core as core
    from graphreduce.graph_reduce import GraphReduce

    original = (
        core._prepare_execution_plan,
        GraphReduce.apply_execution_plan,
        GraphReduce._rebind_sqlop_dates_for_node,
    )
    with pytest.raises(RuntimeError, match="expected experimental failure"):
        with atomic_date_replay():
            raise RuntimeError("expected experimental failure")
    assert original == (
        core._prepare_execution_plan,
        GraphReduce.apply_execution_plan,
        GraphReduce._rebind_sqlop_dates_for_node,
    )


def test_serialized_marker_and_graph_local_binding_do_not_patch_other_graphs():
    import pickle
    import kurversc.core as core

    original = datetime(2019, 1, 1)
    current = original - timedelta(days=30)
    expression = f"root <= '{original}' AND root >= '{current}'"
    operation = sqlop(optype=SQLOpType.where, opval=expression)
    plan = {
        "cut_date": original,
        "kurversc_root_cut_date": original,
        "records": [{"node_prefix": "root", "ops": [operation]}],
    }
    marked = pickle.loads(pickle.dumps(atomic_execution_plan(plan)))
    assert "kurversc_date_rebinding" not in plan
    assert core._execution_plan_fingerprint(marked) != core._execution_plan_fingerprint(
        plan
    )
    prepared = core._prepare_execution_plan(
        marked, root_prefix="root", cut_date=current
    )
    assert prepared["records"][0]["ops"][0].opval == expression

    class Graph:
        def __init__(self):
            self.cut_date = current

        def _rebind_sqlop_dates_for_node(self, node, operations, original_cut):
            return "unmodified"

        def apply_execution_plan(self, prepared_plan):
            node = SimpleNamespace(
                prefix="root", ts_periods=[30], base_predicate_windows=[]
            )
            return self._rebind_sqlop_dates_for_node(
                node, prepared_plan["records"][0]["ops"], original
            )

    active, other = Graph(), Graph()
    result = apply_graph_plan(active, prepared)
    assert (
        result[0].opval
        == f"root <= '{current}' AND root >= '{current - timedelta(days=30)}'"
    )
    assert active._rebind_sqlop_dates_for_node(None, None, None) == "unmodified"
    assert other._rebind_sqlop_dates_for_node(None, None, None) == "unmodified"
    assert "_kurversc_atomic_root_dates" not in active.__dict__


def test_graph_local_binding_restores_prior_instance_attributes_after_failure():
    def prior_method(*args):
        return "prior"

    class Graph:
        def apply_execution_plan(self, plan):
            raise RuntimeError("expected experimental failure")

    graph = Graph()
    graph._rebind_sqlop_dates_for_node = prior_method
    graph._kurversc_atomic_root_dates = ("prior", "old", "new")
    with pytest.raises(RuntimeError, match="expected experimental failure"):
        apply_graph_plan(graph, atomic_execution_plan({}))
    assert graph._rebind_sqlop_dates_for_node is prior_method
    assert graph._kurversc_atomic_root_dates == ("prior", "old", "new")


def test_new_plans_default_to_atomic_and_old_plans_keep_legacy_preparation():
    import kurversc.core as core

    original, current = datetime(2024, 1, 1), datetime(2023, 1, 1)
    graph = SimpleNamespace(
        freeze_execution_plan=lambda: {"records": []}, nodes=lambda: []
    )
    assert (
        core._freeze_execution_plan(graph, original)["kurversc_date_rebinding"]
        == "atomic"
    )
    expression = f"root <= '{original}'"
    legacy = {
        "kurversc_root_cut_date": original,
        "records": [
            {
                "node_prefix": "root",
                "ops": [sqlop(optype=SQLOpType.where, opval=expression)],
            }
        ],
    }
    prepared = core._prepare_execution_plan(
        legacy, root_prefix="root", cut_date=current
    )
    assert prepared["records"][0]["ops"][0].opval == f"root <= '{current}'"
    assert legacy["records"][0]["ops"][0].opval == expression
