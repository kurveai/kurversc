from __future__ import annotations

import pandas as pd

import kurversc
from kurversc.feature_audit import (
    available_feature_families,
    estimate_config_feature_width,
    rank_feature_tables,
)


def test_feature_funnel_uses_utility_instead_of_source_column_order() -> None:
    tables = {
        "entities": kurversc.Table(
            pd.DataFrame(), name="entities", key="entity_id", timeless=True
        ),
        "events": kurversc.Table(
            pd.DataFrame(), name="events", key="event_id", date="occurred_at"
        ),
    }
    relationships = (
        kurversc.Relationship(
            parent="entities",
            child="events",
            parent_key="entity_id",
            child_key="entity_id",
        ),
    )
    samples = {
        "entities": pd.DataFrame(
            {
                "entity_id": [1, 2, 3, 4],
                "constant_first": [1, 1, 1, 1],
                "predictive_range": [0.0, 2.0, 8.0, 20.0],
            }
        ),
        "events": pd.DataFrame(
            {
                "event_id": [1, 2, 3, 4],
                "entity_id": [1, 1, 2, 3],
                "occurred_at": pd.date_range("2025-01-01", periods=4),
                "constant_first": ["x"] * 4,
                "varied_category": ["a", "b", "a", "c"],
            }
        ),
    }

    ranked, audit = rank_feature_tables(
        tables,
        relationships,
        samples,
        feature_family_max_columns=1,
        feature_family_max_column_options=(1, 2),
        feature_family_max_features_per_column=8,
    )

    assert ranked["entities"].columns.index("predictive_range") < ranked[
        "entities"
    ].columns.index("constant_first")
    assert ranked["events"].columns.index("varied_category") < ranked[
        "events"
    ].columns.index("constant_first")
    entity_key = audit.loc[
        (audit["table"] == "entities") & (audit["column"] == "entity_id")
    ].iloc[0]
    varied = audit.loc[
        (audit["table"] == "events") & (audit["column"] == "varied_category")
    ].iloc[0]
    constant = audit.loc[
        (audit["table"] == "events") & (audit["column"] == "constant_first")
    ].iloc[0]
    assert entity_key["structural"]
    assert entity_key["within_family_budget"]
    assert varied["utility_score"] > constant["utility_score"]
    assert "conditional:" in varied["family_ranks"]
    assert varied["within_expanded_family_budget"]
    assert varied["eligible_budget_tiers"] == "1,2"
    assert varied["max_features_per_column"] == 8


def test_feature_funnel_audits_explicitly_excluded_columns() -> None:
    tables = {
        "entities": kurversc.Table(
            pd.DataFrame(),
            name="entities",
            key="entity_id",
            timeless=True,
            columns=("entity_id", "kept"),
        )
    }
    samples = {
        "entities": pd.DataFrame(
            {
                "entity_id": [1, 2, 3],
                "kept": [1.0, 2.0, 3.0],
                "omitted": [9.0, 8.0, 7.0],
            }
        )
    }

    ranked, audit = rank_feature_tables(
        tables,
        (),
        samples,
        feature_family_max_columns=1,
    )

    assert ranked["entities"].columns == ("entity_id", "kept")
    omitted = audit.loc[audit["column"] == "omitted"].iloc[0]
    assert not omitted["eligible"]
    assert omitted["reason"] == "excluded by Table.columns"


def test_public_feature_preflight_estimates_topology_without_relational_sql() -> None:
    customers = pd.DataFrame(
        {"customer_id": range(10), "age": [float(value) for value in range(10)]}
    )
    events = pd.DataFrame(
        {
            "event_id": range(30),
            "customer_id": [value // 3 for value in range(30)],
            "occurred_at": pd.date_range("2025-01-01", periods=30, freq="h"),
            "amount": [float(value % 7) for value in range(30)],
        }
    )
    preflight = kurversc.estimate_features(
        kurversc.Table(customers, name="customers", key="customer_id", timeless=True),
        tables=(
            kurversc.Table(
                events,
                name="events",
                key="event_id",
                date="occurred_at",
            ),
        ),
        relationships=(
            kurversc.Relationship(
                parent="customers",
                child="events",
                parent_key="customer_id",
                child_key="customer_id",
            ),
        ),
        graph_configs=(
            kurversc.GraphConfig(depth=1, auto_annotate_features=False),
            kurversc.GraphConfig(depth=2, auto_annotate_features=False),
        ),
        feature_ranking_rows=5,
    )

    assert len(preflight.estimates) == 2
    assert preflight.results["estimated_output_features"].gt(0).all()
    assert preflight.results["estimated_peak_features"].gt(0).all()
    assert set(preflight.details["table"]) == {"customers", "events"}


def test_capabilities_prune_only_structurally_unavailable_families() -> None:
    tables = {
        "entities": kurversc.Table(
            pd.DataFrame(), name="entities", key="entity_id", timeless=True
        ),
        "events": kurversc.Table(
            pd.DataFrame(), name="events", key="event_id", date="occurred_at"
        ),
    }
    relationships = (
        kurversc.Relationship(
            parent="entities",
            child="events",
            parent_key="entity_id",
            child_key="entity_id",
        ),
    )
    samples = {
        "entities": pd.DataFrame({"entity_id": [1, 2]}),
        "events": pd.DataFrame(
            {
                "event_id": [1, 2],
                "entity_id": [1, 2],
                "occurred_at": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "amount": [1.0, 2.0],
            }
        ),
    }
    _, audit = rank_feature_tables(
        tables,
        relationships,
        samples,
        feature_family_max_columns=4,
    )

    without_annotations = available_feature_families(
        tables,
        relationships,
        audit,
        auto_annotate_features=False,
    )
    with_annotations = available_feature_families(
        tables,
        relationships,
        audit,
        auto_annotate_features=True,
    )

    assert without_annotations == {"base", "temporal", "sequence", "episode"}
    assert "conditional" in with_annotations
    assert estimate_config_feature_width(
        kurversc.GraphConfig(feature_families=("base", "temporal"), depth=2),
        audit,
        relationships,
        tables,
    ) > len(audit)
