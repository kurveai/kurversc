"""Deterministic source-column ranking for the capped feature funnel."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .specs import Relationship, Table


_IDENTIFIER = re.compile(r"(^id$|(^|_)\w*id$|uuid$|guid$|identifier$|key$)", re.I)
_TEXT_HINTS = {
    "body",
    "comment",
    "description",
    "message",
    "note",
    "notes",
    "review",
    "summary",
    "text",
    "title",
}


@dataclass(frozen=True)
class TableFeatureEstimate:
    """Static feature-volume estimate for one reachable graph table."""

    table: str
    hop: int
    source_columns: int
    annotation_features: int
    joined_features: int
    locally_generated_features: int
    propagated_features: int
    output_features: int
    materialized_width: int
    source_rows: int | None
    estimated_working_cells: int | None

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfigFeatureEstimate:
    """Topology-aware estimate for one GraphReduce configuration.

    The estimate is intentionally conservative. It is based on a bounded
    source sample and GraphReduce's configured feature budgets; it never
    executes relational joins or aggregate SQL.
    """

    config: Any
    estimated_output_features: int
    estimated_peak_features: int
    estimated_total_generated_features: int
    estimated_peak_working_cells: int | None
    estimated_total_working_cells: int | None
    reached_tables: int
    reached_relationships: int
    table_estimates: tuple[TableFeatureEstimate, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            **asdict(self.config),
            "estimated_output_features": self.estimated_output_features,
            "estimated_peak_features": self.estimated_peak_features,
            "estimated_total_generated_features": (
                self.estimated_total_generated_features
            ),
            "estimated_peak_working_cells": self.estimated_peak_working_cells,
            "estimated_total_working_cells": self.estimated_total_working_cells,
            "reached_tables": self.reached_tables,
            "reached_relationships": self.reached_relationships,
        }


@dataclass(frozen=True)
class FeaturePreflight:
    """Feature estimates and the bounded source-column audit behind them."""

    estimates: tuple[ConfigFeatureEstimate, ...]
    feature_audit: pd.DataFrame

    @property
    def results(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            estimate.as_record() for estimate in self.estimates
        )

    @property
    def details(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for candidate, estimate in enumerate(self.estimates, start=1):
            for table in estimate.table_estimates:
                records.append(
                    {
                        "candidate": candidate,
                        **asdict(estimate.config),
                        **table.as_record(),
                    }
                )
        return pd.DataFrame.from_records(records)


def _parts(value: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def _structural_columns(
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
) -> dict[str, set[str]]:
    structural = {
        name: {
            *_parts(table.key),
            *([table.date] if table.date else []),
            *table.context_keys,
        }
        for name, table in tables.items()
    }
    for relationship in relationships:
        structural[relationship.parent].update(_parts(relationship.parent_key))
        structural[relationship.child].update(_parts(relationship.child_key))
    return structural


def _is_collection(series: pd.Series) -> bool:
    values = series.dropna().head(20)
    return bool(
        len(values)
        and values.map(lambda value: isinstance(value, (list, dict, set, tuple))).any()
    )


def _role(column: str, series: pd.Series, *, structural: bool) -> str:
    if structural:
        return "structural"
    if _IDENTIFIER.search(column):
        return "identifier"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    values = series.dropna().astype(str).head(500)
    name_tokens = set(re.split(r"[^a-z0-9]+", column.lower()))
    average_length = float(values.str.len().mean()) if len(values) else 0.0
    if name_tokens.intersection(_TEXT_HINTS) or average_length >= 40:
        return "text"
    return "categorical"


def free_text_columns(sample: pd.DataFrame) -> frozenset[str]:
    """Return non-structural columns whose sampled values look like free text."""
    return frozenset(
        str(column)
        for column in sample.columns
        if _role(str(column), sample[column], structural=False) == "text"
    )


def _families_for_role(role: str) -> tuple[str, ...]:
    return {
        "structural": ("base", "temporal", "episode", "context"),
        "identifier": ("base", "temporal", "episode"),
        "boolean": ("base", "conditional", "temporal"),
        "numerical": ("base", "temporal", "sequence", "context"),
        "timestamp": ("base", "temporal", "sequence"),
        "text": ("base",),
        "categorical": ("base", "conditional"),
    }[role]


def _profile(column: str, series: pd.Series, *, structural: bool) -> dict[str, Any]:
    non_null = series.dropna()
    non_null_fraction = float(len(non_null) / max(1, len(series)))
    collection = _is_collection(series)
    if collection or non_null.empty:
        cardinality = 0
        unique_fraction = 0.0
        entropy = 0.0
    else:
        try:
            counts = non_null.value_counts(dropna=True)
        except TypeError:
            counts = pd.Series(dtype=float)
            collection = True
        cardinality = int(len(counts))
        unique_fraction = float(cardinality / max(1, len(non_null)))
        if cardinality > 1:
            probabilities = counts.to_numpy(dtype=float) / float(counts.sum())
            entropy = float(
                -(probabilities * np.log(probabilities)).sum() / math.log(cardinality)
            )
        else:
            entropy = 0.0

    role = _role(column, series, structural=structural)
    spread = 0.0
    if role == "numerical" and len(non_null):
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(numeric) > 1:
            q10, q50, q90 = numeric.quantile([0.1, 0.5, 0.9]).tolist()
            spread = float(abs(q90 - q10) / (abs(q50) + abs(q90 - q10) + 1e-12))
    elif role == "text" and len(non_null):
        lengths = non_null.astype(str).str.len()
        spread = float(min(1.0, lengths.std(ddof=0) / max(1.0, lengths.mean())))
    else:
        spread = entropy

    identifier_redundancy = (
        0.1 if role == "identifier" and unique_fraction >= 0.98 else 1.0
    )
    utility = (
        non_null_fraction
        * identifier_redundancy
        * (0.35 + 0.65 * max(entropy, spread))
        * math.log1p(max(1, cardinality))
    )
    if structural:
        utility = float("inf")
    reason = (
        "structural join/cutoff column"
        if structural
        else "collection-valued sample; retained after scalar columns"
        if collection
        else "no observed values in ranking sample"
        if non_null.empty
        else "constant in ranking sample"
        if cardinality <= 1
        else "eligible and utility-ranked"
    )
    return {
        "column": column,
        "role": role,
        "non_null_fraction": non_null_fraction,
        "cardinality": cardinality,
        "unique_fraction": unique_fraction,
        "entropy": entropy,
        "utility_score": utility,
        "eligible_families": _families_for_role(role),
        "structural": structural,
        "reason": reason,
    }


def rank_feature_tables(
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    samples: Mapping[str, pd.DataFrame],
    *,
    feature_family_max_columns: int | None,
    feature_family_max_column_options: Sequence[int | None] | None = None,
    feature_family_max_features_per_column: int | None = None,
) -> tuple[dict[str, Table], pd.DataFrame]:
    """Order source columns by sample utility and return a complete audit.

    No sample-derived value causes an eligible scalar column to be removed.
    The ranking only decides which columns reach GraphReduce first when a
    family budget is capped. This keeps the full-frame schema available while
    avoiding dependence on physical table column order.
    """

    structural = _structural_columns(tables, relationships)
    ranked_tables: dict[str, Table] = {}
    records: list[dict[str, Any]] = []

    for table_name, table in tables.items():
        sample = samples[table_name]
        if table.columns is not None:
            missing = set(table.columns) - set(sample.columns)
            if missing:
                raise ValueError(
                    f"Table {table_name!r} is missing configured columns: "
                    f"{sorted(missing)}"
                )
        configured = (
            set(table.columns) if table.columns is not None else set(sample.columns)
        )
        profiles = []
        for position, column in enumerate(sample.columns):
            profile = _profile(
                column,
                sample[column],
                structural=column in structural[table_name],
            )
            profile.update(
                {
                    "table": table_name,
                    "source_position": position,
                    "eligible": column in configured,
                }
            )
            if column not in configured:
                profile["reason"] = "excluded by Table.columns"
            profiles.append(profile)

        eligible = [profile for profile in profiles if profile["eligible"]]
        eligible.sort(
            key=lambda profile: (
                not profile["structural"],
                -profile["utility_score"],
                profile["source_position"],
                profile["column"],
            )
        )
        ordered_columns = tuple(profile["column"] for profile in eligible)
        ranked_tables[table_name] = replace(table, columns=ordered_columns)

        family_ranks: dict[str, int] = {}
        column_budgets = tuple(
            dict.fromkeys(
                feature_family_max_column_options or (feature_family_max_columns,)
            )
        )
        for rank, profile in enumerate(eligible, start=1):
            profile["utility_rank"] = rank
            ranks = []
            if not profile["structural"]:
                for family in profile["eligible_families"]:
                    family_ranks[family] = family_ranks.get(family, 0) + 1
                    ranks.append(f"{family}:{family_ranks[family]}")
            profile["family_ranks"] = ",".join(ranks)

            def within_budget(budget: int | None) -> bool:
                return bool(
                    profile["structural"]
                    or budget is None
                    or any(int(item.split(":", 1)[1]) <= budget for item in ranks)
                )

            profile["within_family_budget"] = within_budget(feature_family_max_columns)
            profile["within_expanded_family_budget"] = any(
                within_budget(budget) for budget in column_budgets
            )
            profile["eligible_budget_tiers"] = ",".join(
                "uncapped" if budget is None else str(budget)
                for budget in column_budgets
                if within_budget(budget)
            )
            profile["max_features_per_column"] = feature_family_max_features_per_column
        for profile in profiles:
            if not profile["eligible"]:
                profile["utility_rank"] = None
                profile["family_ranks"] = ""
                profile["within_family_budget"] = False
                profile["within_expanded_family_budget"] = False
                profile["eligible_budget_tiers"] = ""
                profile["max_features_per_column"] = (
                    feature_family_max_features_per_column
                )
            profile["eligible_families"] = ",".join(profile["eligible_families"])
            records.append(profile)

    audit = pd.DataFrame.from_records(records)
    return ranked_tables, audit


def available_feature_families(
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    audit: pd.DataFrame,
    *,
    auto_annotate_features: bool,
) -> frozenset[str]:
    """Return families that can produce operations for this graph.

    The check is deliberately conservative: it removes a family only when the
    graph lacks the structural prerequisite that GraphReduce itself needs.
    It does not use target values and therefore cannot leak validation signal.
    """

    available = {"base"}
    reduced_children = {
        relationship.child for relationship in relationships if relationship.reduce
    }
    if reduced_children:
        available.add("episode")
    dated_children = {
        name
        for name in reduced_children
        if name in tables and tables[name].date is not None
    }
    if dated_children:
        # Both families include relationship-level event-count trajectories,
        # even when no numeric value column is present.
        available.update(("temporal", "sequence"))

    if dated_children and not audit.empty:
        candidates = audit.loc[
            audit["eligible"]
            & audit["within_expanded_family_budget"]
            & audit["table"].isin(dated_children)
            & ~audit["structural"]
        ]
        generic_conditions = candidates["role"].eq("categorical")
        annotated_conditions = candidates["role"].isin(
            {"boolean", "numerical", "categorical"}
        )
        if bool(generic_conditions.any()) or (
            auto_annotate_features and bool(annotated_conditions.any())
        ):
            available.add("conditional")
    return frozenset(available)


def _family_rank(record: pd.Series, family: str) -> int | None:
    for value in str(record.get("family_ranks", "")).split(","):
        name, separator, rank = value.partition(":")
        if separator and name == family:
            return int(rank)
    return None


def _selected_for_family(
    table_audit: pd.DataFrame,
    family: str,
    budget: int | None,
) -> pd.DataFrame:
    selected = table_audit.loc[
        table_audit["eligible"]
        & ~table_audit["structural"]
        & table_audit["eligible_families"]
        .fillna("")
        .str.split(",")
        .map(lambda values: family in values)
    ]
    if budget is None:
        return selected
    return selected.loc[
        selected.apply(
            lambda record: (_family_rank(record, family) or budget + 1) <= budget,
            axis=1,
        )
    ]


def _bounded(value: int, limit: int | None) -> int:
    return value if limit is None else min(value, limit)


def _annotation_feature_estimate(
    table_audit: pd.DataFrame,
    config: Any,
) -> int:
    if not config.auto_annotate_features:
        return 0
    candidates = table_audit.loc[table_audit["eligible"] & ~table_audit["structural"]]
    candidates = candidates.sort_values(
        ["utility_score", "source_position"],
        ascending=[False, True],
        kind="stable",
    )
    identifier_named = (
        candidates["column"]
        .astype(str)
        .map(lambda column: bool(_IDENTIFIER.search(column)))
        .astype(bool)
    )
    numerical_candidates = candidates.loc[
        candidates["role"].eq("numerical") & ~identifier_named
    ]
    numerical = int(len(numerical_candidates))
    gated_numerical = min(4, numerical)
    categorical = candidates.loc[
        candidates["role"].isin(("categorical", "boolean"))
        | (
            candidates["role"].eq("numerical")
            & candidates["cardinality"].le(20)
            & ~identifier_named
        )
    ].head(10)
    generated = 0
    for record in categorical.to_dict("records"):
        cardinality = max(0, int(record["cardinality"]))
        encoded = min(cardinality, 20) if cardinality <= 20 else min(cardinality, 5) + 1
        generated += encoded
        generated += min(encoded, 3) * gated_numerical
    if config.auto_text_features:
        text_columns = int(candidates["role"].eq("text").sum())
        if config.auto_annotate_max_text_columns is not None:
            text_columns = min(text_columns, config.auto_annotate_max_text_columns)
        generated += text_columns * 5
    return generated


def _local_reduction_feature_estimate(
    table: Table,
    table_audit: pd.DataFrame,
    config: Any,
    *,
    annotation_features: int,
) -> int:
    """Approximate GraphReduce aggregate columns generated at one edge."""

    families = set(config.feature_families)
    budget = config.feature_family_max_columns
    per_column = config.feature_family_max_features_per_column
    generated = 0

    if "base" in families:
        # GraphReduce's current base loop visits every eligible scalar source;
        # the source-column budget is applied by the specialized temporal and
        # conditional selectors. Mirror execution here so deep base candidates
        # are not severely underestimated.
        base = _selected_for_family(table_audit, "base", None)
        natural_width = {
            "numerical": 5,
            "categorical": 2,
            "boolean": 2,
            "timestamp": 2,
            "text": 5 if config.auto_text_features else 1,
            "identifier": 1,
        }
        generated += sum(
            _bounded(natural_width.get(str(role), 1), per_column)
            for role in base["role"]
        )
        # Auto-annotated numerical features receive sum/avg/min/max.
        generated += annotation_features * _bounded(4, per_column)

    periods = 11
    if table.date is not None:
        # seconds-since-last, rolling counts, adjacent-period changes,
        # observed-history duration/rate, and bounded base predicates.
        generated += 1 + periods + (periods - 1)
        if "base" in families:
            generated += 2
            windows = len(table.base_predicate_windows)
            per_predicate = windows + max(0, windows - 1) + 1
            generated += table.auto_base_predicate_max * per_predicate

        if "temporal" in families:
            temporal = _selected_for_family(table_audit, "temporal", budget)
            generated += periods
            generated += sum(
                _bounded(2 * periods, per_column)
                if role == "identifier"
                else _bounded(6 * periods + periods - 1, per_column)
                for role in temporal["role"]
            )
        if "sequence" in families:
            generated += 2 * periods + (periods - 1) + 2
        if "conditional" in families:
            conditional = _selected_for_family(table_audit, "conditional", budget)
            generated += len(conditional) * _bounded(32, per_column)

    if "episode" in families:
        generated += 2 + (2 * periods if table.date is not None else 0)
    return int(generated)


def estimate_config_features(
    config: Any,
    audit: pd.DataFrame,
    relationships: Sequence[Relationship],
    tables: Mapping[str, Table],
    *,
    root_name: str | None = None,
    source_rows: Mapping[str, int] | None = None,
) -> ConfigFeatureEstimate:
    """Estimate root, peak-intermediate, and total generated feature counts.

    Planning uses only the source-column audit and graph topology. It does not
    execute joins, annotations, or aggregations, making it suitable as a
    pre-execution guard for large relational graphs.
    """

    if root_name is None:
        children = {relationship.child for relationship in relationships}
        root_name = next((name for name in tables if name not in children), None)
        root_name = root_name or next(iter(tables), "")
    if root_name not in tables:
        raise ValueError(f"Unknown root table: {root_name!r}")

    outgoing: dict[str, list[Relationship]] = {}
    for relationship in relationships:
        if relationship.parent not in tables or relationship.child not in tables:
            raise ValueError(
                "Unknown relationship table: "
                f"{relationship.parent} -> {relationship.child}"
            )
        outgoing.setdefault(relationship.parent, []).append(relationship)

    depths = {root_name: 0}
    frontier = [root_name]
    while frontier:
        parent = frontier.pop(0)
        if depths[parent] >= config.depth:
            continue
        for relationship in outgoing.get(parent, ()):
            candidate_depth = depths[parent] + 1
            previous = depths.get(relationship.child)
            if previous is None or candidate_depth < previous:
                depths[relationship.child] = candidate_depth
                frontier.append(relationship.child)

    active_relationships = tuple(
        relationship
        for relationship in relationships
        if relationship.parent in depths
        and relationship.child in depths
        and depths[relationship.parent] < config.depth
        and depths[relationship.child] == depths[relationship.parent] + 1
    )
    active_outgoing: dict[str, list[Relationship]] = {}
    for relationship in active_relationships:
        active_outgoing.setdefault(relationship.parent, []).append(relationship)

    source_rows = dict(source_rows or {})
    if any(rows < 0 for rows in source_rows.values()):
        raise ValueError("source_rows must contain non-negative row counts")

    outputs: dict[str, int] = {}
    materialized_widths: dict[str, int] = {}
    estimates: dict[str, TableFeatureEstimate] = {}
    total_generated = 0
    peak = 0
    working_cells: list[int] = []
    # The configured propagation cap applies when GraphReduce recognizes the
    # prior aggregate suffix (sum/min/max/count/avg). Window, ratio, recency,
    # and other derived names currently fall through to the full five-function
    # numerical map. Five is therefore the safe static propagation bound even
    # when the configured canonical-continuation cap is one.
    propagation_limit = config.feature_propagation_max_functions_per_column
    propagation_factor = max(5, propagation_limit or 0)

    for table_name in sorted(depths, key=lambda name: depths[name], reverse=True):
        table_audit = audit.loc[(audit["table"] == table_name) & audit["eligible"]]
        source_columns = int(len(table_audit))
        annotations = _annotation_feature_estimate(table_audit, config)
        if table_name == root_name and tables[table_name].date is not None:
            annotations += 1

        joined_features = 0
        for relationship in active_outgoing.get(table_name, ()):
            joined_features += (
                outputs[relationship.child]
                if relationship.reduce
                else materialized_widths[relationship.child]
            )
        materialized_width = source_columns + annotations + joined_features

        incoming = [
            relationship
            for relationship in active_relationships
            if relationship.child == table_name
        ]
        reduced = table_name != root_name and any(
            relationship.reduce for relationship in incoming
        )
        local = (
            _local_reduction_feature_estimate(
                tables[table_name],
                table_audit,
                config,
                annotation_features=annotations,
            )
            if reduced
            else 0
        )
        propagated = joined_features * propagation_factor if reduced else 0
        output = local + propagated if reduced else materialized_width
        table_rows = source_rows.get(table_name)
        table_working_cells = (
            table_rows * max(materialized_width, local + propagated)
            if table_rows is not None
            else None
        )

        materialized_widths[table_name] = materialized_width
        outputs[table_name] = output
        total_generated += annotations + local + propagated
        peak = max(peak, materialized_width, output)
        if table_working_cells is not None:
            working_cells.append(table_working_cells)
        estimates[table_name] = TableFeatureEstimate(
            table=table_name,
            hop=depths[table_name],
            source_columns=source_columns,
            annotation_features=annotations,
            joined_features=joined_features,
            locally_generated_features=local,
            propagated_features=propagated,
            output_features=output,
            materialized_width=materialized_width,
            source_rows=table_rows,
            estimated_working_cells=table_working_cells,
        )

    ordered = tuple(
        estimates[name]
        for name in sorted(estimates, key=lambda name: (depths[name], name))
    )
    return ConfigFeatureEstimate(
        config=config,
        estimated_output_features=outputs.get(root_name, 0),
        estimated_peak_features=peak,
        estimated_total_generated_features=total_generated,
        estimated_peak_working_cells=(max(working_cells) if working_cells else None),
        estimated_total_working_cells=(sum(working_cells) if working_cells else None),
        reached_tables=len(depths),
        reached_relationships=len(active_relationships),
        table_estimates=ordered,
    )


def estimate_config_feature_width(
    config: Any,
    audit: pd.DataFrame,
    relationships: Sequence[Relationship],
    tables: Mapping[str, Table],
) -> int:
    """Estimate final root-frame width without materializing GraphReduce SQL."""

    return estimate_config_features(
        config,
        audit,
        relationships,
        tables,
    ).estimated_output_features
