"""Collision-safe date replay and opt-in upgrades of historical frozen plans."""

import copy
from contextlib import contextmanager
from datetime import datetime, timedelta
import re
from types import MethodType


def atomic_execution_plan(plan):
    """Opt into serializable graph-local date replay without global patches.

    The marker persists in fitted artifacts, including subsequent prediction
    calls. Existing unmarked plans retain their historical replay behavior.
    """
    marked = copy.deepcopy(dict(plan))
    marked["kurversc_date_rebinding"] = "atomic"
    return marked


def prepare_atomic_plan(plan, *, root_prefix, cut_date):
    prepared = copy.deepcopy(dict(plan))
    original = prepared.get("kurversc_root_cut_date")
    if original is not None:
        prepared["kurversc_atomic_root_dates"] = (root_prefix, original, cut_date)
    return prepared


def apply_graph_plan(graph, plan):
    """Scope the correction to one graph while its frozen operations bind."""
    if plan.get("kurversc_date_rebinding") != "atomic":
        return graph.apply_execution_plan(plan)
    missing = object()
    attributes = ("_rebind_sqlop_dates_for_node", "_kurversc_atomic_root_dates")
    prior = {name: graph.__dict__.get(name, missing) for name in attributes}
    graph._rebind_sqlop_dates_for_node = MethodType(atomic_rebind, graph)
    graph._kurversc_atomic_root_dates = plan.get("kurversc_atomic_root_dates")
    try:
        return graph.apply_execution_plan(plan)
    finally:
        for name, value in prior.items():
            if value is missing:
                graph.__dict__.pop(name, None)
            else:
                setattr(graph, name, value)


def replace_dates_once(value: str, replacements: dict[str, str]) -> str:
    """Substitute original literals atomically, never reprocess new literals."""
    if not replacements:
        return value
    pattern = re.compile(
        "|".join(
            re.escape(key)
            for key in sorted(replacements, key=lambda key: (-len(key), key))
        )
    )
    return pattern.sub(lambda match: replacements[match.group(0)], value)


def _datetime(value):
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


def atomic_rebind(graph, node, ops, original_cut_date):
    rebound = copy.deepcopy(ops)
    if not original_cut_date or not graph.cut_date:
        return rebound
    original, current = _datetime(original_cut_date), _datetime(graph.cut_date)
    replacements = {str(original): str(current)}
    if hasattr(node, "compute_period_minutes"):
        delta = timedelta(minutes=node.compute_period_minutes())
        replacements[str(original - delta)] = str(current - delta)
    if (
        hasattr(node, "label_period_minutes")
        and getattr(node, "label_period_val", None) is not None
    ):
        delta = timedelta(minutes=node.label_period_minutes())
        replacements[str(original + delta)] = str(current + delta)
    periods = set(getattr(node, "ts_periods", []) or []) | set(
        getattr(node, "base_predicate_windows", []) or []
    )
    for period in sorted(periods):
        delta = timedelta(days=period)
        replacements[str(original - delta)] = str(current - delta)
    context = getattr(graph, "_kurversc_atomic_root_dates", None)
    if context and node.prefix == context[0]:
        replacements[str(_datetime(context[1]))] = str(_datetime(context[2]))
    for operation in rebound:
        operation.opval = replace_dates_once(operation.opval, replacements)
    return rebound


@contextmanager
def atomic_date_replay(enabled=True):
    """Keep exact-root and graph dates in ONE replay pass; restore afterward.

    The wrapper preserves all graph operations and window widths. It changes
    only substitution semantics. It is intentionally not enabled by defaults
    until direct SQL tests and matched temporal experiments establish safety.
    """
    if not enabled:
        yield
        return
    from graphreduce.graph_reduce import GraphReduce
    import kurversc.core as core

    old_prepare = core._prepare_execution_plan
    old_apply = GraphReduce.apply_execution_plan
    old_rebind = GraphReduce._rebind_sqlop_dates_for_node

    def prepare(plan, *, root_prefix, cut_date):
        return prepare_atomic_plan(plan, root_prefix=root_prefix, cut_date=cut_date)

    def apply(graph, plan):
        graph._kurversc_atomic_root_dates = plan.get("kurversc_atomic_root_dates")
        return old_apply(graph, plan)

    core._prepare_execution_plan = prepare
    GraphReduce.apply_execution_plan = apply
    GraphReduce._rebind_sqlop_dates_for_node = atomic_rebind
    try:
        yield
    finally:
        core._prepare_execution_plan = old_prepare
        GraphReduce.apply_execution_plan = old_apply
        GraphReduce._rebind_sqlop_dates_for_node = old_rebind
