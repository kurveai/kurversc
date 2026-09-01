"""Deterministic source-column ranking for the capped feature funnel."""

from __future__ import annotations

import math
import re
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


def estimate_config_feature_width(
    config: Any,
    audit: pd.DataFrame,
    relationships: Sequence[Relationship],
    tables: Mapping[str, Table],
) -> int:
    """Estimate candidate width before GraphReduce materializes its SQL.

    This is an intentionally approximate planning bound. Runtime promotion
    refines it with observed parent widths; the audit-only estimate primarily
    protects first-generation candidates for which no parent frame exists.
    """

    if audit.empty:
        return 0
    eligible = audit.loc[audit["eligible"] & audit["within_expanded_family_budget"]]
    raw_columns = int(len(eligible))
    max_per_column = config.feature_family_max_features_per_column
    per_column = 32 if max_per_column is None else int(max_per_column)
    family_slots = 0
    for family in config.feature_families:
        if family == "episode":
            continue
        family_slots += int(
            eligible["eligible_families"]
            .fillna("")
            .str.split(",")
            .map(lambda values: family in values)
            .sum()
        )
    reduced_edges = sum(1 for relationship in relationships if relationship.reduce)
    dated_edges = sum(
        1
        for relationship in relationships
        if relationship.reduce
        and relationship.child in tables
        and tables[relationship.child].date is not None
    )
    overhead = reduced_edges * 8
    if "episode" in config.feature_families:
        overhead += reduced_edges * 24
    if "sequence" in config.feature_families:
        overhead += dated_edges * 35
    if "temporal" in config.feature_families:
        overhead += dated_edges * 24
    propagation = 1.0 + 0.5 * max(0, config.depth - 1)
    return int(raw_columns + propagation * (family_slots * per_column + overhead))
