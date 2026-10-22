"""Experimental, schema-derived response/context and mature-history summaries.

This module is deliberately not connected to fit defaults or graph search. It
tests existing conditional/temporal/state concepts at the original event grain.
Binary event payloads are discovered from values, not table or column names.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from .event_summaries import EventSummaryPlan, _compatible_frame, _q


def _stamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tz is not None:
        raise ValueError("A valid timezone-naive cutoff is required")
    return result


def _profile(table, cutoff):
    frame = table.frame
    if table.timestamp:
        frame = frame.loc[pd.to_datetime(frame[table.timestamp]).lt(cutoff)]
    return frame.sample(min(20_000, len(frame)), random_state=0).sort_index()


def _binary_columns(table, cutoff):
    structural = {table.key, table.timestamp, *table.foreign_keys}
    frame = _profile(table, cutoff)
    columns = []
    for name in sorted(frame.columns):
        if name in structural or not (
            pd.api.types.is_numeric_dtype(frame[name])
            or pd.api.types.is_bool_dtype(frame[name])
        ):
            continue
        values = pd.to_numeric(frame[name], errors="coerce").dropna().unique()
        if len(values) == 2 and set(values) == {0, 1}:
            columns.append(name)
    return columns


@dataclass
class ResponseSignalPlan:
    max_depth: int = 2
    max_paths: int = 4
    max_responses: int = 2
    max_contexts: int = 8
    memory_limit: str = "8GB"
    threads: int = 4

    def _connection(self, tables):
        con = duckdb.connect(config={"memory_limit": self.memory_limit, "threads": self.threads})
        refs = {}
        for number, (name, table) in enumerate(sorted(tables.items())):
            if table.key and table.frame[table.key].duplicated().any():
                con.close()
                raise ValueError(f"Non-unique declared key: {name}")
            refs[name] = f"t{number}"
            con.register(refs[name], _compatible_frame(table.frame))
        return con, refs

    def _query(self, tables, refs, plan, cutoff):
        cutoff = _stamp(cutoff)
        names = [self.root_, *(name for name, _ in plan["steps"])]
        joins = f"seeds s JOIN {refs[self.root_]} a0 ON s.entity = a0.{_q(tables[self.root_].key)}"
        where = []
        for number, (name, fk) in enumerate(plan["steps"], 1):
            joins += f" JOIN {refs[name]} a{number} ON a{number}.{_q(fk)} = a{number - 1}.{_q(tables[names[number - 1]].key)}"
        for number, (fk, name, _) in enumerate(plan["dimensions"]):
            dimension = tables[name]
            clause = f"d{number}.{_q(dimension.key)} = a{len(plan['steps'])}.{_q(fk)}"
            if dimension.timestamp:
                clause += f" AND d{number}.{_q(dimension.timestamp)} < TIMESTAMP '{cutoff}'"
            joins += f" LEFT JOIN {refs[name]} d{number} ON {clause}"
        for number, name in enumerate(names):
            if tables[name].timestamp:
                where.append(f"a{number}.{_q(tables[name].timestamp)} < TIMESTAMP '{cutoff}'")
        terminal = tables[names[-1]]
        outputs = ["s.entity AS __root", f"a{len(plan['steps'])}.{_q(terminal.timestamp)} AS __time"]
        outputs += [f"a{len(plan['steps'])}.{_q(name)}::DOUBLE AS r{number}"
                    for number, name in enumerate(plan["responses"])]
        outputs += [f"{context['expression']} AS c{number}"
                    for number, context in enumerate(plan["contexts"])]
        return f"SELECT {', '.join(outputs)} FROM {joins} WHERE {' AND '.join(where)}"

    def fit(self, tables, root, ids, cutoff, *, label_horizon):
        if min(self.max_depth, self.max_paths, self.max_responses, self.max_contexts) < 1:
            raise ValueError("All response-plan budgets must be positive")
        self.root_, self.calibration_cutoff_ = root, _stamp(cutoff)
        self.horizon_seconds_ = int(pd.Timedelta(label_horizon).total_seconds())
        if self.horizon_seconds_ < 4:
            raise ValueError("Positive label horizon of at least four seconds required")
        paths = EventSummaryPlan.fit(tables, root, cutoff, max_depth=self.max_depth,
                                    max_paths=16, max_numeric=8, max_categories=4,
                                    max_dimensions=4)
        self.plans_ = []
        con, refs = self._connection(tables)
        try:
            con.register("seeds", _compatible_frame(pd.DataFrame({"entity": pd.Index(ids).unique()})))
            for path in paths.paths:
                responses = _binary_columns(tables[path.steps[-1][0]], self.calibration_cutoff_)[:self.max_responses]
                if not responses:
                    continue
                names = [root, *(name for name, _ in path.steps)]
                pools = defaultdict(list)
                # Distinct context channels get a turn before one table consumes
                # the budget. FK values condition summaries; IDs are never model inputs.
                sources = [(f"a{i}", name, "terminal" if i == len(names) - 1 else "ancestor")
                           for i, name in enumerate(names) if i]
                sources += [(f"d{i}", name, "dimension") for i, (_, name, _) in enumerate(path.dimensions)]
                for alias, name, group in sources:
                    table, frame = tables[name], _profile(tables[name], self.calibration_cutoff_)
                    for column in sorted(frame.columns):
                        if column in {table.key, table.timestamp} or (alias == f"a{len(path.steps)}" and column in responses):
                            continue
                        if column in table.foreign_keys and table.foreign_keys[column] in names:
                            continue
                        values = frame[column].dropna()
                        try:
                            cardinality = values.nunique()
                        except TypeError:
                            continue
                        if cardinality < 2 or pd.api.types.is_datetime64_any_dtype(values):
                            continue
                        categorical = column in table.foreign_keys or not pd.api.types.is_numeric_dtype(values) or cardinality <= 16
                        if categorical and cardinality > 256:
                            continue
                        if not categorical and pd.api.types.is_integer_dtype(values) and cardinality > 256:
                            continue  # bounded identifier-like integer payloads
                        pools[group].append(dict(source=f"{name}.{column}", expression=f"{alias}.{_q(column)}",
                                                 categorical=categorical))
                candidates = []
                while any(pools.values()):
                    for group in ("terminal", "dimension", "ancestor"):
                        if pools[group]:
                            candidates.append(pools[group].pop(0))
                plan = dict(steps=path.steps, dimensions=path.dimensions, responses=responses,
                            contexts=candidates[:self.max_contexts * 3])
                query = self._query(tables, refs, plan, cutoff)
                order = ["__root", "__time", *(f"r{i}" for i in range(len(responses))),
                         *(f"c{i}" for i in range(len(plan["contexts"])))]
                profile = con.execute(f"SELECT * FROM ({query}) ORDER BY hash({', '.join(order)}), {', '.join(order)} LIMIT 20000").df()
                kept = []
                for number, context in enumerate(plan["contexts"]):
                    values = profile[f"c{number}"].dropna()
                    if values.nunique() < 2:
                        continue
                    context = dict(context)
                    if context["categorical"]:
                        text = values.astype(str)
                        if pd.api.types.is_bool_dtype(values):
                            text = text.str.lower()  # DuckDB VARCHAR booleans
                        counts = text.value_counts()
                        context["values"] = sorted(counts.index, key=lambda x: (-int(counts[x]), x))[:2]
                        context["edges"] = []
                    else:
                        values = pd.to_numeric(values).replace([np.inf, -np.inf], np.nan).dropna()
                        if values.nunique() < 2:
                            continue
                        context["edges"] = sorted(set(map(float, values.quantile([1/3, 2/3]))))
                        context["values"] = []
                    kept.append(context)
                    if len(kept) == self.max_contexts:
                        break
                plan["contexts"] = kept
                self.plans_.append(plan)
                if len(self.plans_) == self.max_paths:
                    break
        finally:
            con.close()
        return self

    def audit(self):
        return dict(root=self.root_, calibration_cutoff=str(self.calibration_cutoff_),
                    label_horizon_seconds=self.horizon_seconds_, paths=self.plans_,
                    availability="Ordinary event rows and dated joins strictly before the example cutoff; no labels accepted.")

    def transform(self, tables, ids, cutoff, *, training=False):
        cutoff = _stamp(cutoff)
        if cutoff < self.calibration_cutoff_ and not training:
            raise ValueError("Query precedes response encoder calibration")
        ids = pd.Index(ids)
        if ids.hasnans:
            raise ValueError("Root identities cannot be missing")
        con, refs = self._connection(tables)
        blocks = {arm: [] for arm in ("context_marginal", "response_context", "response_momentum")}
        try:
            con.register("seeds", _compatible_frame(pd.DataFrame({"entity": ids.unique()})))
            for p, plan in enumerate(self.plans_):
                con.execute(f"CREATE TEMP TABLE events AS {self._query(tables, refs, plan, cutoff)}")
                expressions = {arm: [] for arm in blocks}
                for r, _ in enumerate(plan["responses"]):
                    response = f"r{r}"
                    for scope, condition in (("all", "TRUE"), ("recent", f"__time >= TIMESTAMP '{cutoff}' - INTERVAL '{self.horizon_seconds_} SECOND'")):
                        support = f"count({response}) FILTER (WHERE {condition})"
                        rate = f"avg({response}) FILTER (WHERE {condition})"
                        expressions["context_marginal"] += [(f"r{r}_{scope}_support", support), (f"r{r}_{scope}_response_rate", rate)]
                        for c, context in enumerate(plan["contexts"]):
                            column = f"c{c}"
                            if context["categorical"]:
                                conditions = [f"CAST({column} AS VARCHAR) = '{v.replace(chr(39), chr(39)*2)}'" for v in context["values"]]
                                conditions.append(f"{column} IS NOT NULL AND NOT ({' OR '.join(conditions)})")
                            else:
                                edges = context["edges"]
                                conditions = [f"{column} <= {edges[0]}"]
                                conditions += [f"{column} > {a} AND {column} <= {b}" for a, b in zip(edges, edges[1:])]
                                conditions += [f"{column} > {edges[-1]}"]
                            for bucket, mask in enumerate(conditions):
                                condition2 = f"({condition}) AND ({mask})"
                                n = f"count({response}) FILTER (WHERE {condition2})"
                                avg = f"avg({response}) FILTER (WHERE {condition2})"
                                prefix = f"r{r}_{scope}_c{c}_b{bucket}"
                                expressions["context_marginal"].append((prefix + "_support", n))
                                expressions["response_context"] += [(prefix + "_responses", f"sum({response}) FILTER (WHERE {condition2})"),
                                                                     (prefix + "_rate", avg),
                                                                     (prefix + "_lift", f"({avg}) - ({rate})")]
                    expressions["response_momentum"] += [
                        (f"r{r}_seconds_since_response", f"date_diff('second', max(__time) FILTER (WHERE {response}=1), TIMESTAMP '{cutoff}')"),
                        (f"r{r}_active_response_hours", f"count(DISTINCT date_trunc('hour', __time)) FILTER (WHERE {response}=1)"),
                    ]
                    for number, seconds in enumerate([self.horizon_seconds_ // 4, self.horizon_seconds_ // 2, self.horizon_seconds_, self.horizon_seconds_ * 2]):
                        mask = f"__time >= TIMESTAMP '{cutoff}' - INTERVAL '{seconds} SECOND'"
                        expressions["response_momentum"] += [
                            (f"r{r}_w{number}_support", f"count({response}) FILTER (WHERE {mask})"),
                            (f"r{r}_w{number}_responses", f"sum({response}) FILTER (WHERE {mask})"),
                            (f"r{r}_w{number}_rate", f"avg({response}) FILTER (WHERE {mask})"),
                        ]
                    recent = f"__time >= TIMESTAMP '{cutoff}' - INTERVAL '{self.horizon_seconds_} SECOND'"
                    previous = f"__time < TIMESTAMP '{cutoff}' - INTERVAL '{self.horizon_seconds_} SECOND' AND __time >= TIMESTAMP '{cutoff}' - INTERVAL '{2*self.horizon_seconds_} SECOND'"
                    expressions["response_momentum"] += [
                        (f"r{r}_response_count_change", f"coalesce(sum({response}) FILTER (WHERE {recent}),0)-coalesce(sum({response}) FILTER (WHERE {previous}),0)"),
                        (f"r{r}_response_rate_change", f"avg({response}) FILTER (WHERE {recent})-avg({response}) FILTER (WHERE {previous})"),
                    ]
                for arm, exprs in expressions.items():
                    if not exprs:
                        continue
                    sql = ', '.join(f"{expr} AS {_q(name)}" for name, expr in exprs)
                    frame = con.execute(f"SELECT __root, {sql} FROM events GROUP BY __root").df().set_index("__root").reindex(ids)
                    counts = [c for c in frame if c.endswith(("_support", "_responses", "_hours", "_count_change"))]
                    frame[counts] = frame[counts].fillna(0)
                    blocks[arm].append(frame.add_prefix(f"response_p{p}_"))
                con.execute("DROP TABLE events")
        finally:
            con.close()
        return {arm: (pd.concat(parts, axis=1) if parts else pd.DataFrame(index=ids)).reset_index(drop=True).astype("float32")
                for arm, parts in blocks.items()}


def mature_own_history(history, ids, cutoff, *, entity, timestamp, target, label_horizon,
                       phase_cutoff=None, prior_strength=5.0):
    """Explicit same-root history; labels must mature strictly before query/phase."""
    horizon = pd.Timedelta(label_horizon)
    if horizon <= pd.Timedelta(0) or prior_strength <= 0:
        raise ValueError("A positive horizon and prior strength are required")
    cutoff = _stamp(cutoff)
    available = pd.to_datetime(history[timestamp]) + horizon
    limit = min(cutoff, _stamp(phase_cutoff)) if phase_cutoff is not None else cutoff
    rows = history.loc[available.lt(limit), [entity, timestamp, target]].copy()
    rows[target] = pd.to_numeric(rows[target], errors="raise").astype("float64")
    rows["__available"] = available.loc[rows.index]
    if rows.duplicated([entity, timestamp]).any():
        raise ValueError("Historical entity/timestamp labels must be unique")
    ids = pd.Index(ids)
    if ids.hasnans:
        raise ValueError("Root identities cannot be missing")
    prior = float(pd.to_numeric(rows[target]).mean()) if len(rows) else np.nan
    seconds = int(horizon.total_seconds())
    with duckdb.connect() as con:
        con.register("history", _compatible_frame(rows))
        con.register("seeds", _compatible_frame(pd.DataFrame({"entity": ids.unique()})))
        y = _q(target)
        result = con.execute(f"""
            SELECT s.entity AS __root, count(h.{y}) AS own_history_count,
              avg(h.{y}) AS own_history_mean, stddev_pop(h.{y}) AS own_history_std,
              arg_max(h.{y}, h.__available) AS own_history_latest,
              date_diff('second', max(h.__available), TIMESTAMP '{cutoff}') AS own_history_seconds_since_latest,
              count(h.{y}) FILTER (WHERE h.__available >= TIMESTAMP '{cutoff}' - INTERVAL '{seconds} SECOND') AS own_history_recent_count,
              avg(h.{y}) FILTER (WHERE h.__available >= TIMESTAMP '{cutoff}' - INTERVAL '{seconds} SECOND') AS own_history_recent_mean,
              count(h.{y}) FILTER (WHERE h.__available >= TIMESTAMP '{cutoff}' - INTERVAL '{2*seconds} SECOND') AS own_history_two_horizon_count,
              avg(h.{y}) FILTER (WHERE h.__available >= TIMESTAMP '{cutoff}' - INTERVAL '{2*seconds} SECOND') AS own_history_two_horizon_mean
            FROM seeds s LEFT JOIN history h ON s.entity=h.{_q(entity)} GROUP BY s.entity
        """).df().set_index("__root").reindex(ids)
    result["own_history_shrunk_mean"] = (result.own_history_mean.fillna(0) * result.own_history_count + prior_strength*prior) / (result.own_history_count + prior_strength)
    return result.reset_index(drop=True).astype("float32")
