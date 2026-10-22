from __future__ import annotations

import pandas as pd

import kurversc
from kurversc.feature_audit import (
    available_feature_families,
    estimate_config_features,
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
    assert varied["selection_rank"] < constant["selection_rank"]
    assert varied["column_selection_strategy"] == "metadata_utility"


def test_feature_funnel_can_preserve_source_column_order() -> None:
    tables = {
        "entities": kurversc.Table(
            pd.DataFrame(),
            name="entities",
            key="entity_id",
            timeless=True,
            columns=("entity_id", "varied_second", "constant_first"),
        )
    }
    samples = {
        "entities": pd.DataFrame(
            {
                "entity_id": [1, 2, 3, 4],
                "constant_first": [1, 1, 1, 1],
                "varied_second": [0.0, 2.0, 8.0, 20.0],
            }
        )
    }

    ordered, audit = rank_feature_tables(
        tables,
        (),
        samples,
        feature_family_max_columns=1,
        intelligent_column_selection=False,
    )

    assert ordered["entities"].columns == (
        "entity_id",
        "varied_second",
        "constant_first",
    )
    constant = audit.loc[audit["column"] == "constant_first"].iloc[0]
    varied = audit.loc[audit["column"] == "varied_second"].iloc[0]
    assert varied["selection_rank"] < constant["selection_rank"]
    assert varied["utility_rank"] < constant["utility_rank"]
    assert varied["within_family_budget"]
    assert not constant["within_family_budget"]
    assert constant["column_selection_strategy"] == "configured_source_order"


def test_intelligent_column_selection_keeps_release_tie_break() -> None:
    tables = {
        "entities": kurversc.Table(
            pd.DataFrame(),
            name="entities",
            key="entity_id",
            timeless=True,
            columns=("entity_id", "configured_first", "source_first"),
        )
    }
    samples = {
        "entities": pd.DataFrame(
            {
                "entity_id": [1, 2, 3, 4],
                "source_first": [0.0, 1.0, 0.0, 1.0],
                "configured_first": [1.0, 0.0, 1.0, 0.0],
            }
        )
    }

    ordered, audit = rank_feature_tables(
        tables,
        (),
        samples,
        feature_family_max_columns=1,
        intelligent_column_selection=True,
    )

    assert ordered["entities"].columns == (
        "entity_id",
        "source_first",
        "configured_first",
    )
    source = audit.loc[audit["column"] == "source_first"].iloc[0]
    configured = audit.loc[audit["column"] == "configured_first"].iloc[0]
    assert source["utility_score"] == configured["utility_score"]
    assert source["utility_rank"] < configured["utility_rank"]


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

    assert without_annotations == {
        "base",
        "temporal",
        "state",
        "sequence",
        "episode",
    }
    assert "conditional" in with_annotations
    assert estimate_config_feature_width(
        kurversc.GraphConfig(feature_families=("base", "temporal"), depth=2),
        audit,
        relationships,
        tables,
    ) > len(audit)


def test_capabilities_enable_context_for_reduced_child_with_peer_key() -> None:
    tables = {
        "drivers": kurversc.Table(
            pd.DataFrame(), name="drivers", key="driver_id", timeless=True
        ),
        "results": kurversc.Table(
            pd.DataFrame(),
            name="results",
            key="result_id",
            date="occurred_at",
            context_keys=("race_id",),
        ),
    }
    relationships = (
        kurversc.Relationship(
            parent="drivers",
            child="results",
            parent_key="driver_id",
            child_key="driver_id",
        ),
    )
    samples = {
        "drivers": pd.DataFrame({"driver_id": [1, 2]}),
        "results": pd.DataFrame(
            {
                "result_id": [1, 2],
                "driver_id": [1, 2],
                "race_id": [10, 10],
                "occurred_at": pd.to_datetime(["2025-01-01", "2025-01-01"]),
                "grid": [1.0, 3.0],
            }
        ),
    }
    _, audit = rank_feature_tables(
        tables,
        relationships,
        samples,
        feature_family_max_columns=4,
    )

    available = available_feature_families(
        tables,
        relationships,
        audit,
        auto_annotate_features=False,
    )
    base = estimate_config_features(
        kurversc.GraphConfig(auto_annotate_features=False),
        audit,
        relationships,
        tables,
        root_name="drivers",
    )
    context = estimate_config_features(
        kurversc.GraphConfig(
            feature_families=("base", "context"),
            auto_annotate_features=False,
        ),
        audit,
        relationships,
        tables,
        root_name="drivers",
    )

    assert "context" in available
    assert "state" in available
    assert context.estimated_total_generated_features > (
        base.estimated_total_generated_features
    )


def test_topology_estimate_exposes_intermediate_and_propagated_features() -> None:
    tables = {
        "customers": kurversc.Table(
            pd.DataFrame(), name="customers", key="customer_id", timeless=True
        ),
        "sessions": kurversc.Table(
            pd.DataFrame(), name="sessions", key="session_id", date="started_at"
        ),
        "events": kurversc.Table(
            pd.DataFrame(), name="events", key="event_id", date="occurred_at"
        ),
    }
    relationships = (
        kurversc.Relationship(
            parent="customers",
            child="sessions",
            parent_key="customer_id",
            child_key="customer_id",
        ),
        kurversc.Relationship(
            parent="sessions",
            child="events",
            parent_key="session_id",
            child_key="session_id",
        ),
    )
    samples = {
        "customers": pd.DataFrame({"customer_id": [1, 2], "age": [30.0, 40.0]}),
        "sessions": pd.DataFrame(
            {
                "session_id": [10, 20],
                "customer_id": [1, 2],
                "started_at": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "channel": ["web", "app"],
            }
        ),
        "events": pd.DataFrame(
            {
                "event_id": [100, 200],
                "session_id": [10, 20],
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
        feature_family_max_features_per_column=32,
    )

    shallow = estimate_config_features(
        kurversc.GraphConfig(depth=1, auto_annotate_features=False),
        audit,
        relationships,
        tables,
        root_name="customers",
    )
    deep = estimate_config_features(
        kurversc.GraphConfig(depth=2, auto_annotate_features=False),
        audit,
        relationships,
        tables,
        root_name="customers",
        source_rows={"customers": 2, "sessions": 20, "events": 200},
    )

    assert shallow.reached_tables == 2
    assert shallow.reached_relationships == 1
    assert deep.reached_tables == 3
    assert deep.reached_relationships == 2
    assert deep.estimated_output_features > shallow.estimated_output_features
    assert (
        deep.estimated_total_generated_features
        > shallow.estimated_total_generated_features
    )
    assert deep.estimated_peak_working_cells is not None
    assert deep.estimated_total_working_cells is not None
    sessions = next(
        estimate for estimate in deep.table_estimates if estimate.table == "sessions"
    )
    assert sessions.joined_features > 0
    assert sessions.propagated_features == sessions.joined_features * 5


def test_public_feature_preflight_uses_bounded_source_samples() -> None:
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
    configs = (
        kurversc.GraphConfig(depth=1, auto_annotate_features=False),
        kurversc.GraphConfig(depth=2, auto_annotate_features=False),
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
        graph_configs=configs,
        feature_ranking_rows=5,
    )

    assert len(preflight.estimates) == 2
    assert len(preflight.results) == 2
    assert {
        "estimated_output_features",
        "estimated_peak_features",
        "estimated_total_generated_features",
        "estimated_peak_working_cells",
        "estimated_total_working_cells",
        "reached_tables",
        "reached_relationships",
    }.issubset(preflight.results.columns)
    assert set(preflight.details["table"]) == {"customers", "events"}
    assert preflight.results["estimated_peak_working_cells"].notna().all()
    assert int(preflight.feature_audit.groupby("table").size().max()) > 0
