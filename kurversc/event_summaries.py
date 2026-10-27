"""Experimental bounded, point-in-time summaries of schema-derived events.

This does not add a search family or change fit defaults. It supplies ordinary
numeric features for controlled tests of finer temporal resolution, numeric
payload preservation, and support-aware conditional aggregation. Plans are
learned at the training cutoff, without labels, and reused for later snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import duckdb
import numpy as np
import pandas as pd


def _q(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _compatible_frame(frame):
    # Older DuckDB versions do not understand pandas 3's inferred StringDtype.
    strings = [c for c in frame if isinstance(frame[c].dtype, pd.StringDtype)]
    if not strings:
        return frame
    result = frame.copy(deep=False)
    for column in strings:
        result[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return result


def _balanced_numeric_pairs(width, budget):
    """Cover distinct payloads before repeatedly pairing the first column."""
    remaining = list(combinations(range(width), 2))
    uses = [0] * width
    chosen = []
    while remaining and len(chosen) < budget:
        pair = min(remaining, key=lambda p: (uses[p[0]] + uses[p[1]], p))
        remaining.remove(pair)
        chosen.append(pair)
        uses[pair[0]] += 1
        uses[pair[1]] += 1
    return chosen


@dataclass(frozen=True)
class EventTable:
    frame: pd.DataFrame
    key: str | None
    timestamp: str | None
    foreign_keys: Mapping[str, str]


@dataclass(frozen=True)
class EventPath:
    # Each step is (child table, child's foreign key to the preceding table).
    steps: tuple[tuple[str, str], ...]
    numeric: tuple[str, ...]
    categories: tuple[tuple[str, tuple[object, ...]], ...]
    dimensions: tuple[tuple[str, str, tuple[str, ...]], ...]


def _payloads(table: EventTable, cutoff, max_numeric, max_categories, max_values):
    structural = {table.key, table.timestamp, *table.foreign_keys}
    frame = table.frame
    if table.timestamp:
        frame = frame.loc[frame[table.timestamp] <= pd.Timestamp(cutoff)]
    if len(frame) > 20_000:
        frame = frame.sample(20_000, random_state=0).sort_index()
    numeric, categories = [], []
    for column in frame.columns:
        if column in structural or pd.api.types.is_datetime64_any_dtype(frame[column]):
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        try:
            cardinality = values.nunique()
        except TypeError:
            # Lists/structs need a dedicated bounded encoder, not an enum of
            # whole containers or an accidental explosion into categories.
            continue
        if cardinality < 2:
            continue
        if pd.api.types.is_numeric_dtype(values):
            finite = pd.to_numeric(values).replace([np.inf, -np.inf], np.nan).dropna()
            if finite.nunique() > 1:
                numeric.append((column, float(len(finite) / max(len(frame), 1))))
        if cardinality <= 16:
            counts = values.value_counts()
            order = sorted(counts.index, key=lambda x: (-int(counts[x]), str(x)))
            categories.append((column, tuple(order[:max_values])))
    # Do not reinterpret a numeric enum as *only* categorical. Its measured
    # magnitude/order remains available alongside the original graph features.
    numeric.sort(key=lambda item: (-item[1], item[0]))
    return tuple(c for c, _ in numeric[:max_numeric]), tuple(
        categories[:max_categories]
    )


class EventSummaryPlan:
    """Bounded FK-only event paths, with one output row per requested root.

    Reverse-FK walks connect dated facts to the root. Every subsequent join is
    to a declared unique parent key: this cannot multiply a fact into several
    observations. Dimension payloads are lifted *before* reduction. Dated
    facts, intermediate tables, and dimensions obey the exact inclusive cutoff
    used for ordinary RelBench database snapshots (not target-label history).
    """

    def __init__(self, root: str, paths: tuple[EventPath, ...], omitted_paths=()):
        self.root = root
        self.paths = paths
        self.omitted_paths = tuple(omitted_paths)

    @classmethod
    def fit(
        cls,
        tables: Mapping[str, EventTable],
        root: str,
        cutoff,
        *,
        max_depth: int = 2,
        max_paths: int = 8,
        max_numeric: int = 4,
        max_categories: int = 2,
        max_values: int = 4,
        max_dimensions: int = 2,
    ):
        if min(max_depth, max_paths, max_numeric, max_categories, max_values) < 1:
            raise ValueError("Event plan budgets must be positive")
        if max_dimensions < 0:
            raise ValueError("The dimension budget cannot be negative")
        if root not in tables or not tables[root].key:
            raise ValueError("The root requires a declared single-column primary key")
        frontier, candidates = [(root, ())], []
        for _ in range(max_depth):
            following = []
            for parent, steps in frontier:
                if not tables[parent].key:
                    continue
                for name, table in sorted(tables.items()):
                    if name in (root, *(step[0] for step in steps)):
                        continue
                    for fk, destination in sorted(table.foreign_keys.items()):
                        if destination != parent:
                            continue
                        next_steps = (*steps, (name, fk))
                        following.append((name, next_steps))
                        if table.timestamp:
                            candidates.append(next_steps)
            frontier = following
        # Breadth first; give distinct fact tables a turn before allowing two
        # FK channels of one table to consume the entire path budget.
        ordinal, ranked = {}, []
        for steps in sorted(candidates, key=lambda p: (len(p), p)):
            identity = (len(steps), steps[-1][0])
            occurrence = ordinal.get(identity, 0)
            ordinal[identity] = occurrence + 1
            ranked.append((len(steps), occurrence, steps))
        ordered = [steps for _, _, steps in sorted(ranked)]
        paths = []
        for steps in ordered[:max_paths]:
            table = tables[steps[-1][0]]
            numeric, categories = _payloads(
                table, cutoff, max_numeric, max_categories, max_values
            )
            dimensions = []
            for dim_fk, dim_name in sorted(table.foreign_keys.items()):
                if dim_name in (root, *(step[0] for step in steps)):
                    continue
                dim = tables[dim_name]
                if not dim.key:
                    continue
                payload, _ = _payloads(dim, cutoff, max_numeric, 1, 1)
                if payload and len(dimensions) < max_dimensions:
                    dimensions.append((dim_fk, dim_name, payload))
            paths.append(EventPath(steps, numeric, categories, tuple(dimensions)))
        return cls(root, tuple(paths), ordered[max_paths:])

    def audit(self):
        return {
            "root": self.root,
            "coverage_complete_within_max_depth": not self.omitted_paths,
            "omitted_event_paths_due_to_budget": self.omitted_paths,
            "paths": [
                {
                    "steps": p.steps,
                    "numeric": p.numeric,
                    "categories": [
                        (c, [str(v) for v in values]) for c, values in p.categories
                    ],
                    "dimensions": p.dimensions,
                }
                for p in self.paths
            ],
            "availability": "all dated rows <= exact example cutoff; no target labels",
            "fit": "unlabeled training-cutoff payload profiles; fixed for validation",
        }

    def transform(
        self,
        tables: Mapping[str, EventTable],
        ids,
        cutoff,
        *,
        mode: str = "subday",
        numeric_scope: str = "both",
        numeric_history_days: int = 730,
        max_numeric_pairs: int = 2,
        memory_limit: str = "4GB",
        threads: int = 4,
    ) -> pd.DataFrame:
        if mode not in {"subday", "clock", "numeric", "conditional", "joint_numeric"}:
            raise ValueError("Unknown event summary mode")
        if numeric_scope not in {"both", "raw", "dimensions"}:
            raise ValueError("Unknown numeric payload scope")
        if not 1 <= max_numeric_pairs <= 8:
            raise ValueError("Numeric pair budget must be between 1 and 8")
        if not 730 <= numeric_history_days <= 3650:
            raise ValueError(
                "Numeric history must be bounded between 730 and 3650 days"
            )
        index = pd.Index(ids)
        if index.hasnans:
            raise ValueError("Root IDs cannot be missing")
        stamp = pd.Timestamp(cutoff)
        if pd.isna(stamp):
            raise ValueError("A valid cutoff is required")
        conn = duckdb.connect()
        try:
            conn.execute("SET memory_limit = ?", [memory_limit])
            conn.execute("SET threads = ?", [threads])
            conn.execute("SET max_temp_directory_size = '16GB'")
            for i, (name, table) in enumerate(sorted(tables.items())):
                conn.register(f"t{i}", _compatible_frame(table.frame))
            refs = {name: f"t{i}" for i, name in enumerate(sorted(tables))}
            # Distinct seeds avoid inflating counts if callers repeat an entity.
            conn.register(
                "seeds", _compatible_frame(pd.DataFrame({"entity": index.unique()}))
            )
            outputs = []
            for path_number, path in enumerate(self.paths):
                names = [self.root, *(s[0] for s in path.steps)]
                terminal = tables[names[-1]]
                for name in names[:-1]:
                    key = tables[name].key
                    if tables[name].frame[key].duplicated().any():
                        raise ValueError(
                            f"Declared parent key is not unique: {name}.{key}"
                        )
                terminal_alias = f"a{len(names) - 1}"
                where, params = [], []
                joins = f"seeds s JOIN {refs[self.root]} a0 ON s.entity = a0.{_q(tables[self.root].key)}"
                for i, (name, fk) in enumerate(path.steps, 1):
                    joins += (
                        f" JOIN {refs[name]} a{i} ON a{i}.{_q(fk)} = "
                        f"a{i - 1}.{_q(tables[names[i - 1]].key)}"
                    )
                for i, name in enumerate(names):
                    if tables[name].timestamp:
                        where.append(f"a{i}.{_q(tables[name].timestamp)} <= ?")
                        params.append(stamp)
                time = f"{terminal_alias}.{_q(terminal.timestamp)}"
                if mode == "subday":
                    where.append(f"{time} > ?")
                    params.append(stamp - pd.Timedelta(hours=24))
                elif mode == "clock":
                    where.append(f"{time} > ?")
                    params.append(stamp - pd.Timedelta(days=7))
                elif mode in {"numeric", "conditional", "joint_numeric"}:
                    where.append(f"{time} > ?")
                    params.append(
                        stamp
                        - pd.Timedelta(
                            days=numeric_history_days
                            if mode in {"numeric", "joint_numeric"}
                            else 730
                        )
                    )
                payloads = [(c, f"{terminal_alias}.{_q(c)}") for c in path.numeric]
                if (
                    mode in {"numeric", "joint_numeric"}
                    and numeric_scope == "dimensions"
                ):
                    payloads = []
                if mode in {"numeric", "joint_numeric"} and numeric_scope != "raw":
                    for j, (fk, name, columns) in enumerate(path.dimensions):
                        dim = tables[name]
                        if dim.frame[dim.key].duplicated().any():
                            raise ValueError(
                                f"Dimension key is not unique: {name}.{dim.key}"
                            )
                        clause = f"d{j}.{_q(dim.key)} = {terminal_alias}.{_q(fk)}"
                        if dim.timestamp:
                            clause += f" AND d{j}.{_q(dim.timestamp)} <= ?"
                            params.append(stamp)
                        # Keep dimension timestamp parameters before WHERE params.
                        joins += f" LEFT JOIN {refs[name]} d{j} ON {clause}"
                        payloads.extend(
                            (f"{name}.{c}", f"d{j}.{_q(c)}") for c in columns
                        )
                    dated_dimensions = sum(
                        bool(tables[n].timestamp) for _, n, _ in path.dimensions
                    )
                    if dated_dimensions:
                        params = params[-dated_dimensions:] + params[:-dated_dimensions]
                prefix = f"evt_{path_number:02d}"
                expressions, expression_params = [], []

                def add(expression, name, *values):
                    expressions.append(f"{expression} AS {_q(prefix + '_' + name)}")
                    expression_params.extend(
                        v.item() if isinstance(v, np.generic) else v for v in values
                    )

                if mode == "joint_numeric":
                    # Preserve within-event association, which cannot be
                    # recovered by combining independent marginal means.
                    # The balanced pair panel is target blind and strictly
                    # bounded; no additional search family is introduced.
                    pairs = _balanced_numeric_pairs(len(payloads), max_numeric_pairs)
                    for days in (7, 30, 365, 730):
                        lower = stamp - pd.Timedelta(days=days)
                        add(
                            f"count(*) FILTER (WHERE {time} > ?)",
                            f"{days}d_count",
                            lower,
                        )
                        for left, right in pairs:
                            x = f"try_cast({payloads[left][1]} AS DOUBLE)"
                            y = f"try_cast({payloads[right][1]} AS DOUBLE)"
                            valid = f"{time} > ? AND isfinite({x}) AND isfinite({y})"
                            name = f"{days}d_p{left}p{right}"
                            add(
                                f"count(*) FILTER (WHERE {valid})",
                                name + "_support",
                                lower,
                            )
                            add(
                                f"avg({x} * {y}) FILTER (WHERE {valid})",
                                name + "_product_mean",
                                lower,
                            )
                            add(
                                f"covar_pop({x}, {y}) FILTER (WHERE {valid})",
                                name + "_covariance",
                                lower,
                            )
                            add(
                                f"corr({x}, {y}) FILTER (WHERE {valid})",
                                name + "_correlation",
                                lower,
                            )
                elif mode in {"subday", "numeric"}:
                    windows = (
                        ((1, "1h"), (3, "3h"), (6, "6h"), (12, "12h"), (24, "24h"))
                        if mode == "subday"
                        else ((30, "30d"), (90, "90d"), (365, "365d"), (730, "730d"))
                    )
                    if mode == "numeric" and numeric_history_days > 730:
                        windows = (
                            *windows,
                            (numeric_history_days, f"{numeric_history_days}d"),
                        )
                    for width, label in windows:
                        delta = pd.Timedelta(
                            **(
                                {"hours": width}
                                if mode == "subday"
                                else {"days": width}
                            )
                        )
                        lower = stamp - delta
                        predicate = f"{time} > ?"
                        add(
                            f"count(*) FILTER (WHERE {predicate})",
                            f"{label}_count",
                            lower,
                        )
                        for j, (_, value) in enumerate(payloads):
                            finite = f"isfinite(try_cast({value} AS DOUBLE))"
                            valid = f"{predicate} AND {finite}"
                            add(
                                f"count(*) FILTER (WHERE {valid})",
                                f"{label}_p{j}_support",
                                lower,
                            )
                            add(
                                f"avg(try_cast({value} AS DOUBLE)) FILTER (WHERE {valid})",
                                f"{label}_p{j}_mean",
                                lower,
                            )
                            if mode == "numeric":
                                add(
                                    f"sum(try_cast({value} AS DOUBLE)) FILTER (WHERE {valid})",
                                    f"{label}_p{j}_sum",
                                    lower,
                                )
                                # Ties use a deterministic value rather than input order.
                                add(
                                    f"first(try_cast({value} AS DOUBLE) ORDER BY {time} DESC, {value} ASC) FILTER (WHERE {valid})",
                                    f"{label}_p{j}_latest",
                                    lower,
                                )
                elif mode == "clock":
                    # Habitual time-of-day activity is distinct from a short
                    # recent-window count. Use four fixed six-hour bands, not
                    # an unbounded categorical expansion of timestamps.
                    for hours in (24, 168):
                        lower = stamp - pd.Timedelta(hours=hours)
                        add(
                            f"count(*) FILTER (WHERE {time} > ?)",
                            f"{hours}h_count",
                            lower,
                        )
                        for bucket in range(4):
                            condition = f"{time} > ? AND floor(extract(hour FROM {time}) / 6) = {bucket}"
                            add(
                                f"count(*) FILTER (WHERE {condition})",
                                f"{hours}h_b{bucket}_count",
                                lower,
                            )
                            add(
                                f"count(DISTINCT cast({time} AS DATE)) FILTER (WHERE {condition})",
                                f"{hours}h_b{bucket}_active_days",
                                lower,
                            )
                else:
                    # Aggregate sums and counts from the same event rows, not
                    # an unweighted mean of means from intermediate neighbors.
                    add("count(*)", "all_support")
                    for j, (column, values) in enumerate(path.categories):
                        for k, value in enumerate(values):
                            equal = f"{terminal_alias}.{_q(column)} = ?"
                            add(
                                f"count(*) FILTER (WHERE {equal})",
                                f"c{j}v{k}_support",
                                value,
                            )
                            for p, (numeric, expression) in enumerate(payloads):
                                if numeric == column:
                                    continue
                                valid = f"{equal} AND isfinite(try_cast({expression} AS DOUBLE))"
                                add(
                                    f"count(*) FILTER (WHERE {valid})",
                                    f"c{j}v{k}_p{p}_support",
                                    value,
                                )
                                add(
                                    f"sum(try_cast({expression} AS DOUBLE)) FILTER (WHERE {valid})",
                                    f"c{j}v{k}_p{p}_sum",
                                    value,
                                )
                query = (
                    f"SELECT s.entity, {', '.join(expressions)} FROM {joins} "
                    f"WHERE {' AND '.join(where)} GROUP BY s.entity"
                )
                block = (
                    conn.execute(query, [*expression_params, *params])
                    .df()
                    .set_index("entity")
                )
                block = block.reindex(index)
                for column in block:
                    if column.endswith(("_count", "_support", "_sum")):
                        block[column] = block[column].fillna(0)
                derived = {}
                if mode == "subday":
                    for label in ("1h", "3h", "6h", "12h"):
                        derived[f"{prefix}_{label}_fraction24h"] = block[
                            f"{prefix}_{label}_count"
                        ] / block[f"{prefix}_24h_count"].clip(lower=1)
                elif mode == "clock":
                    for hours in (24, 168):
                        for bucket in range(4):
                            derived[f"{prefix}_{hours}h_b{bucket}_share"] = block[
                                f"{prefix}_{hours}h_b{bucket}_count"
                            ] / block[f"{prefix}_{hours}h_count"].clip(lower=1)
                elif mode == "conditional":
                    for column in list(block):
                        if column.endswith("_sum"):
                            support = block[column.removesuffix("_sum") + "_support"]
                            derived[column.removesuffix("_sum") + "_mean"] = block[
                                column
                            ] / support.replace(0, np.nan)
                        elif (
                            column.endswith("_support")
                            and "_p" not in column
                            and column != f"{prefix}_all_support"
                        ):
                            derived[column.removesuffix("_support") + "_share"] = block[
                                column
                            ] / block[f"{prefix}_all_support"].clip(lower=1)
                if derived:
                    block = pd.concat(
                        [block, pd.DataFrame(derived, index=block.index)], axis=1
                    )
                outputs.append(block.astype("float32"))
            return pd.concat(outputs, axis=1) if outputs else pd.DataFrame(index=index)
        finally:
            conn.close()
