import pytest

from kurversc import (
    GraphConfig,
    SearchStageBudgets,
    incremental_configs,
    published_compatible_configs,
    staged_challenger_configs,
    staged_modern_seed_configs,
    state_focused_configs,
)
from kurversc.search import (
    COMPACT_MODERN_FAMILY_TEMPLATES,
    DEFAULT_FAMILY_STAGES,
    WIDE_TARGET_HISTORY_TEMPLATES,
    Trial,
    adaptive_depth_candidate_allowed,
    campaign_modern_configs,
    confidence_frontier_trials,
    compact_modern_configs,
    diverse_collection_configs,
    diverse_confirmation_trials,
    diverse_fidelity_trials,
    forward_candidate_allowed,
    stability_adjusted_winner,
    wide_modern_configs,
)


def test_search_stage_budgets_validate_hard_evaluation_limits() -> None:
    budgets = SearchStageBudgets(
        screening=24,
        challenger=12,
        combined_challenger=6,
        confirmation=10,
        fidelity_promotion=8,
        categorical_refinement=8,
        rerank=4,
    )

    assert budgets.screening == 24
    assert budgets.categorical_refinement == 8
    with pytest.raises(ValueError, match="screening budget must be positive"):
        SearchStageBudgets(screening=0)
    with pytest.raises(ValueError, match="must be non-negative"):
        SearchStageBudgets(challenger=-1)


def test_incremental_configs_start_with_requested_baseline() -> None:
    configs = incremental_configs(max_depth=3)

    assert configs[0] == GraphConfig(
        feature_families=("base",),
        depth=1,
        auto_annotate_features=True,
        feature_family_max_columns=4,
    )
    assert configs[1].auto_annotate_features is False
    assert [config.depth for config in configs[:4]] == [1, 1, 2, 2]
    assert configs[2].auto_annotate_features is True
    assert configs[6].feature_families == ("base", "temporal")
    assert len(configs) == 272
    assert {config.feature_family_max_columns for config in configs} == {4}
    assert all(
        config.feature_family_max_features_per_column == 32 for config in configs
    )
    assert all(
        config.feature_propagation_max_functions_per_column == 1 for config in configs
    )
    assert all("semantic" not in c.feature_families for c in configs)
    assert any("context" in c.feature_families for c in configs)
    assert {c.auto_annotate_features for c in configs if c.depth > 1} == {True, False}
    assert all(
        set(c.feature_families).issubset({"base", "temporal", "state", "sequence"})
        for c in configs
        if c.depth == 3
    )
    assert {c.feature_families for c in configs if len(c.feature_families) == 2} == {
        ("base", "temporal"),
        ("base", "state"),
        ("base", "sequence"),
        ("base", "conditional"),
        ("base", "episode"),
        ("base", "context"),
    }
    assert configs[-1].feature_families == (
        "base",
        "temporal",
        "state",
        "sequence",
        "conditional",
        "episode",
        "context",
    )
    assert configs[-1].feature_family_max_columns == 4


def test_incremental_configs_default_includes_guarded_depth_four() -> None:
    configs = incremental_configs()

    assert len(configs) == 288
    assert {config.depth for config in configs} == {1, 2, 3, 4}
    assert all(
        set(config.feature_families).issubset({"base", "temporal", "state", "sequence"})
        for config in configs
        if config.depth >= 3
    )


def test_incremental_configs_cross_history_and_bounded_topology_options() -> None:
    configs = incremental_configs(
        max_depth=1,
        feature_family_stages=(("base", "state"),),
        target_history_options=(False, True),
        max_base_table_occurrence_options=(1, 2),
        limit_expensive_family_depth=False,
    )

    assert len(configs) == 16
    assert {config.include_target_history for config in configs} == {False, True}
    assert {config.max_base_table_occurrences for config in configs} == {1, 2}
    assert {config.auto_annotate_features for config in configs} == {False, True}
    assert {config.feature_families for config in configs} == {
        ("base",),
        ("base", "state"),
    }


def test_graph_config_rejects_invalid_topology_occurrence_limit() -> None:
    with pytest.raises(ValueError, match="max_base_table_occurrences"):
        GraphConfig(max_base_table_occurrences=0)


def test_published_compatible_configs_retain_the_72_shape_anchor() -> None:
    configs = published_compatible_configs()

    assert len(configs) == 72
    assert {config.depth for config in configs} == {1, 2, 3}
    assert all("state" not in config.feature_families for config in configs)
    assert all("context" not in config.feature_families for config in configs)
    assert all(config.include_target_history is False for config in configs)
    assert all(config.published_cutoff_compatibility for config in configs)
    assert all(config.high_cardinality_policy == "top_k" for config in configs)
    assert all(config.categorical_low_cardinality_policy == "all" for config in configs)
    assert all(config.text_categorical_fallback for config in configs)
    assert all(config.categorical_tie_break == "legacy" for config in configs)
    assert all(config.legacy_training_row_order for config in configs)


def test_published_anchor_can_use_deterministic_plan_and_row_order() -> None:
    configs = published_compatible_configs(legacy_plan_compatibility=False)

    assert len(configs) == 72
    assert all(config.published_cutoff_compatibility for config in configs)
    assert all(config.categorical_tie_break == "value" for config in configs)
    assert all(not config.legacy_training_row_order for config in configs)


def test_published_compatible_configs_cross_explicit_orthogonal_dimensions() -> None:
    configs = published_compatible_configs(
        auto_annotate_options=(False,),
        target_history_options=(False, True),
        max_base_table_occurrence_options=(1, 2),
    )

    assert len(configs) == 144
    assert {config.include_target_history for config in configs} == {False, True}
    assert {config.max_base_table_occurrences for config in configs} == {1, 2}
    assert {config.auto_annotate_features for config in configs} == {False}
    assert all(
        config.published_cutoff_compatibility == (not config.include_target_history)
        for config in configs
    )


def test_published_cutoff_compatibility_rejects_target_history() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        GraphConfig(
            include_target_history=True,
            published_cutoff_compatibility=True,
        )


def test_staged_challengers_are_shallow_canonical_and_inherit_controls() -> None:
    parent = GraphConfig(
        feature_families=("base", "temporal", "sequence"),
        depth=3,
        auto_annotate_features=False,
        feature_family_max_columns=8,
        high_cardinality_policy="top_k",
        categorical_low_cardinality_policy="all",
        text_categorical_fallback=True,
        include_target_history=False,
        published_cutoff_compatibility=True,
        categorical_tie_break="legacy",
        legacy_training_row_order=True,
    )

    configs = staged_challenger_configs((parent,))

    assert len(configs) == 4
    assert {config.depth for config in configs} == {1, 2}
    assert {config.feature_families for config in configs} == {
        ("base", "temporal", "state", "sequence"),
        ("base", "temporal", "sequence", "context"),
    }
    assert all(config.auto_annotate_features is False for config in configs)
    assert all(config.feature_family_max_columns == 8 for config in configs)
    assert all(config.high_cardinality_policy == "top_k" for config in configs)
    assert all(config.text_categorical_fallback for config in configs)
    assert all(config.include_target_history is False for config in configs)
    assert all(config.published_cutoff_compatibility is False for config in configs)
    assert all(config.categorical_tie_break == "value" for config in configs)
    assert all(not config.legacy_training_row_order for config in configs)

    without_history = staged_challenger_configs((parent,), include_target_history=False)
    assert all(config.include_target_history is False for config in without_history)


def test_staged_context_challengers_do_not_inherit_root_reachable_scope() -> None:
    parent = GraphConfig(
        feature_families=("base", "temporal"),
        depth=1,
        include_target_history=False,
        root_reachable_materialization=True,
    )

    challengers = staged_challenger_configs(
        (parent,),
        additions=(("state",), ("context",), ("state", "context")),
        depths=(1,),
    )

    assert len(challengers) == 3
    assert next(
        config for config in challengers if config.feature_families == ("base", "temporal", "state")
    ).root_reachable_materialization
    assert all(
        not config.root_reachable_materialization
        for config in challengers
        if "context" in config.feature_families
    )

    with pytest.raises(ValueError, match="cross-root context"):
        GraphConfig(
            feature_families=("base", "context"),
            root_reachable_materialization=True,
        )


def test_staged_modern_seeds_cover_sparse_native_controls() -> None:
    controls = (
        GraphConfig(
            auto_annotate_features=annotate,
            high_cardinality_policy="top_k",
            categorical_low_cardinality_policy="all",
            text_categorical_fallback=True,
            include_target_history=False,
            published_cutoff_compatibility=True,
        )
        for annotate in (True, False)
    )

    configs = staged_modern_seed_configs(tuple(controls))

    assert len(configs) == 18
    assert {config.auto_annotate_features for config in configs} == {True, False}
    assert all(config.published_cutoff_compatibility is False for config in configs)
    assert all(
        config.categorical_low_cardinality_policy == "top_k" for config in configs
    )
    assert {
        (config.feature_families, config.depth, config.include_target_history)
        for config in configs
    } == {
        (("base", "state"), 1, False),
        (("base", "state"), 2, False),
        (("base", "context"), 1, False),
        (("base", "context"), 2, False),
        (("base", "state", "context"), 1, False),
        (("base", "state", "context"), 2, False),
        (("base",), 1, True),
        (("base", "state"), 1, True),
        (("base", "context"), 1, True),
    }


def test_incremental_configs_only_expand_column_budget_when_requested() -> None:
    configs = incremental_configs(
        max_depth=3,
        feature_family_max_column_options=(4, 8),
    )

    assert len(configs) == 544
    assert {config.feature_family_max_columns for config in configs} == {4, 8}


def test_incremental_configs_can_opt_out_of_context_family() -> None:
    configs = incremental_configs(
        max_depth=3,
        feature_family_stages=tuple(
            tuple(family for family in stage if family != "context")
            for stage in DEFAULT_FAMILY_STAGES
        ),
    )

    assert len(configs) == 144
    assert all("context" not in config.feature_families for config in configs)


def test_state_focused_configs_isolate_signal_in_eight_shallow_candidates() -> None:
    configs = state_focused_configs()

    assert len(configs) == 8
    assert {config.depth for config in configs} == {1, 2}
    assert {config.auto_annotate_features for config in configs} == {True, False}
    assert {config.feature_families for config in configs} == {
        ("base",),
        ("base", "state"),
        ("base", "temporal", "state"),
    }
    assert sum(config.feature_families == ("base",) for config in configs) == 2
    assert sum("state" in config.feature_families for config in configs) == 6


def test_compact_modern_configs_guarantee_state_interactions() -> None:
    configs = compact_modern_configs()

    assert len(configs) == 40
    assert len(COMPACT_MODERN_FAMILY_TEMPLATES) == 10
    assert {config.depth for config in configs} == {1, 2}
    assert {config.auto_annotate_features for config in configs} == {True, False}
    assert all(config.include_target_history for config in configs)
    winning_shape = ("base", "temporal", "state", "sequence", "episode")
    assert {
        (config.depth, config.auto_annotate_features)
        for config in configs
        if config.feature_families == winning_shape
    } == {(1, True), (1, False), (2, True), (2, False)}


def test_wide_modern_configs_cover_depth_three_and_history_controls() -> None:
    configs = wide_modern_configs()

    assert len(configs) == 80
    assert {config.depth for config in configs} == {1, 2, 3}
    assert sum(not config.include_target_history for config in configs) == 60
    assert sum(config.include_target_history for config in configs) == 20
    assert all(config.depth <= 2 for config in configs if config.include_target_history)
    assert {
        config.feature_families for config in configs if config.include_target_history
    } == set(WIDE_TARGET_HISTORY_TEMPLATES)
    state_shape = ("base", "temporal", "state", "sequence", "episode")
    assert any(
        config.feature_families == state_shape
        and config.depth == 3
        and not config.include_target_history
        for config in configs
    )


def test_campaign_modern_configs_balance_modern_signal_in_32_candidates() -> None:
    configs = campaign_modern_configs()

    assert len(configs) == 32
    assert {config.depth for config in configs} == {1, 2, 3}
    assert {config.auto_annotate_features for config in configs} == {True, False}
    assert {config.include_target_history for config in configs} == {True, False}
    assert {
        family
        for config in configs
        for family in config.feature_families
        if family != "base"
    } == {"temporal", "state", "sequence", "conditional", "episode", "context"}
    assert any(
        config.depth == 3 and "state" in config.feature_families for config in configs
    )


def test_collection_design_balances_families_depths_annotations_and_budgets() -> None:
    lattice = incremental_configs(
        max_depth=4,
        feature_family_max_column_options=(2, 4, 8),
    )

    selected = diverse_collection_configs(lattice, max_candidates=24)

    assert len(selected) == 24
    assert {config.feature_family_max_columns for config in selected} == {2, 4, 8}
    assert {config.depth for config in selected} == {1, 2, 3, 4}
    assert {config.auto_annotate_features for config in selected} == {True, False}
    assert {
        family
        for config in selected
        for family in config.feature_families
        if family != "base"
    } == {"temporal", "state", "sequence", "conditional", "episode", "context"}


def test_collection_design_reserves_history_and_cyclic_topology_dimensions() -> None:
    lattice = incremental_configs(
        max_depth=4,
        target_history_options=(False, True),
        max_base_table_occurrence_options=(1, 2),
    )

    selected = diverse_collection_configs(lattice, max_candidates=24)

    assert {config.depth for config in selected} == {1, 2, 3, 4}
    assert {config.include_target_history for config in selected} == {False, True}
    assert {config.max_base_table_occurrences for config in selected} == {1, 2}


def test_collection_design_can_retain_complete_lattice() -> None:
    lattice = incremental_configs(max_depth=2)

    assert diverse_collection_configs(lattice, max_candidates=None) == lattice


def test_collection_design_rotates_higher_order_combinations_by_seed() -> None:
    lattice = incremental_configs(
        max_depth=4,
        feature_family_max_column_options=(2, 4, 8),
    )

    first = diverse_collection_configs(lattice, max_candidates=24, random_state=1)
    second = diverse_collection_configs(lattice, max_candidates=24, random_state=2)

    first_combinations = {
        config.feature_families for config in first if len(config.feature_families) >= 3
    }
    second_combinations = {
        config.feature_families
        for config in second
        if len(config.feature_families) >= 3
    }
    assert first_combinations != second_combinations


def _trial(config: GraphConfig, score: float, features: int = 10) -> Trial:
    return Trial(
        config=config,
        metric="roc_auc",
        validation_score=score,
        objective_score=score,
        feature_count=features,
        train_rows=100,
        validation_rows=50,
        feature_seconds=1.0,
        model_seconds=1.0,
    )


def test_confidence_frontier_retains_low_rank_plausible_graph() -> None:
    leader = _trial(GraphConfig(feature_families=("base",)), 0.66)
    leader.validation_standard_error = 0.012
    challengers = []
    for offset, family in enumerate(
        ("temporal", "state", "sequence", "conditional", "episode"), start=1
    ):
        trial = _trial(GraphConfig(feature_families=("base", family)), 0.66 - offset * 0.003)
        trial.validation_standard_error = 0.012
        challengers.append(trial)
    implausible = _trial(GraphConfig(feature_families=("base", "context")), 0.50)
    implausible.validation_standard_error = 0.001

    frontier = confidence_frontier_trials(
        [leader, *challengers, implausible], top_k=7, fill_budget=False
    )

    assert len(frontier) == 6
    assert challengers[-1] in frontier
    assert implausible not in frontier
    assert len(confidence_frontier_trials(frontier, top_k=4)) == 4
    assert len(
        confidence_frontier_trials(
            [leader, *challengers, implausible],
            top_k=7,
            extra_controls=0,
        )
    ) == 6


def test_confirmation_finalists_preserve_family_and_shape_diversity() -> None:
    raw = _trial(GraphConfig(auto_annotate_features=True), 0.95)
    simple = _trial(GraphConfig(auto_annotate_features=False), 0.94)
    temporal = _trial(GraphConfig(feature_families=("base", "temporal")), 0.93)
    sequence = _trial(GraphConfig(feature_families=("base", "sequence")), 0.92)
    conditional = _trial(GraphConfig(feature_families=("base", "conditional")), 0.91)
    episode = _trial(GraphConfig(feature_families=("base", "episode")), 0.90)
    deeper = _trial(GraphConfig(depth=2, auto_annotate_features=True), 0.89)
    alternate = _trial(
        GraphConfig(
            feature_families=("base", "temporal"),
            auto_annotate_features=False,
        ),
        0.88,
    )

    selected = diverse_confirmation_trials(
        [
            raw,
            simple,
            temporal,
            sequence,
            conditional,
            episode,
            deeper,
            alternate,
        ],
        top_k=8,
        complexity_recommendation=simple,
    )

    assert len(selected) == 8
    assert selected[:2] == (raw, simple)
    assert {"temporal", "sequence", "conditional", "episode"}.issubset(
        {family for trial in selected for family in trial.config.feature_families}
    )
    assert any(trial.config.depth > 1 for trial in selected)
    assert {trial.config.auto_annotate_features for trial in selected} == {
        True,
        False,
    }


def test_fidelity_promotion_retains_sparse_family_controls_across_annotations() -> None:
    raw = _trial(GraphConfig(auto_annotate_features=False), 0.95)
    temporal_plain = _trial(
        GraphConfig(
            feature_families=("base", "temporal"),
            auto_annotate_features=False,
        ),
        0.94,
    )
    temporal_annotated = _trial(
        GraphConfig(feature_families=("base", "temporal")),
        0.93,
    )
    rich_state = _trial(
        GraphConfig(
            feature_families=("base", "temporal", "state"),
            auto_annotate_features=False,
        ),
        0.92,
    )
    state_plain = _trial(
        GraphConfig(
            feature_families=("base", "state"),
            auto_annotate_features=False,
        ),
        0.40,
    )
    state_annotated = _trial(
        GraphConfig(feature_families=("base", "state")),
        0.39,
    )

    selected = diverse_fidelity_trials(
        [
            raw,
            temporal_plain,
            temporal_annotated,
            rich_state,
            state_plain,
            state_annotated,
        ],
        top_k=6,
        complexity_recommendation=raw,
    )

    assert state_plain in selected
    assert state_annotated in selected
    assert rich_state in selected


def test_fidelity_promotion_reserves_widest_interactions_before_main_effects() -> None:
    raw = _trial(GraphConfig(auto_annotate_features=False), 0.99)
    pair_plain = _trial(
        GraphConfig(
            feature_families=("base", "temporal", "sequence"),
            auto_annotate_features=False,
        ),
        0.98,
    )
    pair_annotated = _trial(
        GraphConfig(feature_families=("base", "temporal", "sequence")),
        0.97,
    )
    widest_plain = _trial(
        GraphConfig(
            feature_families=("base", "temporal", "sequence", "episode"),
            auto_annotate_features=False,
        ),
        0.20,
    )
    widest_annotated = _trial(
        GraphConfig(feature_families=("base", "temporal", "sequence", "episode")),
        0.19,
    )

    selected = diverse_fidelity_trials(
        [raw, pair_plain, pair_annotated, widest_plain, widest_annotated],
        top_k=3,
        complexity_recommendation=raw,
    )

    assert selected == (raw, widest_plain, widest_annotated)


def test_fidelity_promotion_reserves_every_sparse_family_before_score_fill() -> None:
    raw = _trial(GraphConfig(auto_annotate_features=False), 0.99)
    distractors = [
        _trial(
            GraphConfig(
                feature_families=("base", "temporal", "sequence"),
                depth=depth,
                auto_annotate_features=annotate,
            ),
            0.98 - index * 0.01,
        )
        for index, (depth, annotate) in enumerate(
            ((1, False), (1, True), (2, False), (2, True))
        )
    ]
    sparse = [
        _trial(
            GraphConfig(
                feature_families=("base", family),
                auto_annotate_features=annotate,
            ),
            0.10,
        )
        for family in ("temporal", "sequence")
        for annotate in (False, True)
    ]

    selected = diverse_fidelity_trials(
        [raw, *distractors, *sparse],
        top_k=7,
        complexity_recommendation=raw,
    )

    assert all(trial in selected for trial in sparse)


def test_structural_promotion_protects_base_and_orthogonal_axes() -> None:
    raw = _trial(
        GraphConfig(
            feature_families=("base", "temporal", "sequence"),
            auto_annotate_features=False,
            include_target_history=False,
        ),
        0.99,
    )
    annotated_base = _trial(
        GraphConfig(feature_families=("base",), include_target_history=False),
        0.20,
    )
    history = _trial(
        GraphConfig(
            feature_families=("base", "temporal"),
            auto_annotate_features=False,
            include_target_history=True,
        ),
        0.70,
    )
    cyclic = _trial(
        GraphConfig(
            feature_families=("base", "temporal"),
            auto_annotate_features=False,
            include_target_history=False,
            max_base_table_occurrences=2,
        ),
        0.60,
    )
    sparse = _trial(
        GraphConfig(
            feature_families=("base", "sequence"),
            auto_annotate_features=False,
            include_target_history=False,
        ),
        0.98,
    )

    selected = diverse_fidelity_trials(
        [raw, annotated_base, history, cyclic, sparse],
        top_k=4,
        structural_first=True,
    )

    assert selected == (raw, annotated_base, history, cyclic)


def test_structural_promotion_prefers_shallow_base_control() -> None:
    raw = _trial(
        GraphConfig(
            depth=2,
            feature_families=("base", "temporal"),
            auto_annotate_features=False,
        ),
        0.99,
    )
    deep_base = _trial(
        GraphConfig(depth=2, feature_families=("base",), auto_annotate_features=False),
        0.98,
    )
    shallow_base = _trial(
        GraphConfig(depth=1, feature_families=("base",), auto_annotate_features=False),
        0.20,
    )
    annotated_deep_base = _trial(
        GraphConfig(depth=2, feature_families=("base",), auto_annotate_features=True),
        0.97,
    )
    annotated_shallow_base = _trial(
        GraphConfig(depth=1, feature_families=("base",), auto_annotate_features=True),
        0.10,
    )

    selected = diverse_fidelity_trials(
        [raw, deep_base, shallow_base, annotated_deep_base, annotated_shallow_base],
        top_k=3,
        structural_first=True,
    )

    assert selected == (raw, shallow_base, annotated_shallow_base)


def test_structural_promotion_reserves_distinct_shallow_family_controls() -> None:
    raw = _trial(GraphConfig(depth=2, feature_families=("base",)), 0.99)
    deep_base = _trial(
        GraphConfig(
            depth=2,
            feature_families=("base",),
            auto_annotate_features=False,
        ),
        0.98,
    )
    shallow_base = _trial(GraphConfig(depth=1, feature_families=("base",)), 0.50)
    shallow_base_unannotated = _trial(
        GraphConfig(
            depth=1,
            feature_families=("base",),
            auto_annotate_features=False,
        ),
        0.40,
    )
    sparse = [
        _trial(GraphConfig(depth=1, feature_families=("base", family)), score)
        for family, score in (("context", 0.95), ("temporal", 0.94), ("state", 0.93))
    ]
    deep_state = _trial(GraphConfig(depth=2, feature_families=("base", "state")), 0.97)

    selected = diverse_fidelity_trials(
        [
            raw,
            deep_base,
            shallow_base,
            shallow_base_unannotated,
            deep_state,
            *sparse,
        ],
        top_k=6,
        complexity_recommendation=deep_base,
        structural_first=True,
        shallow_sparse_controls=3,
    )

    assert selected == (raw, shallow_base_unannotated, shallow_base, *sparse)


def test_rerank_winner_uses_raw_stability_adjusted_objective() -> None:
    unstable = _trial(GraphConfig(feature_families=("base", "temporal")), 0.90)
    unstable.objective_score = 0.88
    stable = _trial(GraphConfig(feature_families=("base",)), 0.89)

    assert stability_adjusted_winner([unstable, stable]) is stable


def test_depth_three_requires_meaningful_depth_two_gain() -> None:
    depth1 = _trial(GraphConfig(depth=1), 0.80)
    depth2 = _trial(GraphConfig(depth=2), 0.801)
    depth3 = GraphConfig(depth=3)

    assert not adaptive_depth_candidate_allowed(
        depth3,
        [depth1, depth2],
        classification_gain=0.002,
        regression_relative_gain=0.005,
        uncertainty_multiplier=0.0,
    )
    depth2.validation_score = depth2.objective_score = 0.81
    assert adaptive_depth_candidate_allowed(
        depth3,
        [depth1, depth2],
        classification_gain=0.002,
        regression_relative_gain=0.005,
        uncertainty_multiplier=0.0,
    )


def test_depth_four_requires_a_successful_depth_three_parent() -> None:
    depth1 = _trial(GraphConfig(depth=1), 0.80)
    depth2 = _trial(GraphConfig(depth=2), 0.81)
    depth3 = _trial(GraphConfig(depth=3), 0.801)
    depth4 = GraphConfig(depth=4)

    assert not adaptive_depth_candidate_allowed(
        depth4,
        [depth1, depth2],
        classification_gain=0.002,
        regression_relative_gain=0.005,
        uncertainty_multiplier=0.0,
    )
    assert not adaptive_depth_candidate_allowed(
        depth4,
        [depth1, depth2, depth3],
        classification_gain=0.002,
        regression_relative_gain=0.005,
        uncertainty_multiplier=0.0,
    )
    depth3.validation_score = depth3.objective_score = 0.82
    assert adaptive_depth_candidate_allowed(
        depth4,
        [depth1, depth2, depth3],
        classification_gain=0.002,
        regression_relative_gain=0.005,
        uncertainty_multiplier=0.0,
    )


def test_depth_two_promotes_only_the_stronger_annotation_policy() -> None:
    annotated = _trial(GraphConfig(depth=1, auto_annotate_features=True), 0.81)
    plain = _trial(GraphConfig(depth=1, auto_annotate_features=False), 0.80)

    assert adaptive_depth_candidate_allowed(
        GraphConfig(depth=2, auto_annotate_features=True),
        [plain, annotated],
        classification_gain=0.002,
        regression_relative_gain=0.005,
        uncertainty_multiplier=0.0,
    )
    assert not adaptive_depth_candidate_allowed(
        GraphConfig(depth=2, auto_annotate_features=False),
        [plain, annotated],
        classification_gain=0.002,
        regression_relative_gain=0.005,
        uncertainty_multiplier=0.0,
    )


def test_forward_beam_only_expands_the_strongest_compatible_parent() -> None:
    weak = GraphConfig(depth=1, auto_annotate_features=True)
    strong = GraphConfig(depth=2, auto_annotate_features=False)
    trials = [_trial(weak, 0.70), _trial(strong, 0.80)]

    assert forward_candidate_allowed(
        GraphConfig(
            feature_families=("base", "episode"),
            depth=2,
            auto_annotate_features=False,
        ),
        trials,
        beam_width=1,
    )
    assert not forward_candidate_allowed(
        GraphConfig(
            feature_families=("base", "episode"),
            depth=1,
            auto_annotate_features=True,
        ),
        trials,
        beam_width=1,
    )


def test_forward_beam_requires_one_family_at_a_time() -> None:
    parent = GraphConfig(depth=1, auto_annotate_features=False)

    assert not forward_candidate_allowed(
        GraphConfig(
            feature_families=("base", "temporal", "episode"),
            depth=1,
            auto_annotate_features=False,
        ),
        [_trial(parent, 0.80)],
        beam_width=1,
    )


def test_forward_beam_refines_only_top_shape_to_next_column_budget() -> None:
    weak = GraphConfig(
        feature_families=("base",),
        depth=1,
        auto_annotate_features=True,
        feature_family_max_columns=4,
    )
    strong = GraphConfig(
        feature_families=("base", "temporal"),
        depth=2,
        auto_annotate_features=False,
        feature_family_max_columns=4,
    )
    trials = [_trial(weak, 0.70), _trial(strong, 0.80)]

    assert forward_candidate_allowed(
        GraphConfig(
            feature_families=("base", "temporal"),
            depth=2,
            auto_annotate_features=False,
            feature_family_max_columns=8,
        ),
        trials,
        beam_width=1,
        feature_family_max_column_options=(4, 8),
    )
    assert forward_candidate_allowed(
        GraphConfig(
            feature_families=("base",),
            depth=1,
            auto_annotate_features=True,
            feature_family_max_columns=8,
        ),
        trials,
        beam_width=1,
        feature_family_max_column_options=(4, 8),
        column_budget_promotions=(weak,),
    )
    assert not forward_candidate_allowed(
        GraphConfig(
            feature_families=("base",),
            depth=1,
            auto_annotate_features=True,
            feature_family_max_columns=8,
        ),
        trials,
        beam_width=1,
        feature_family_max_column_options=(4, 8),
    )


def test_graphreduce_kwargs_leave_uncapped_budget_to_node_builder() -> None:
    config = GraphConfig(feature_family_max_columns=None)

    assert "feature_family_max_columns" not in config.graphreduce_kwargs()


def test_graph_config_rejects_nonpositive_feature_family_cap() -> None:
    with pytest.raises(ValueError, match="feature_family_max_columns"):
        GraphConfig(feature_family_max_columns=0)
    with pytest.raises(ValueError, match="feature_family_max_features_per_column"):
        GraphConfig(feature_family_max_features_per_column=0)
    with pytest.raises(
        ValueError, match="feature_propagation_max_functions_per_column"
    ):
        GraphConfig(feature_propagation_max_functions_per_column=0)


def test_graph_config_exposes_graphreduce_compatibility_policies() -> None:
    config = GraphConfig(
        categorical_cardinality_threshold=32,
        categorical_top_k=20,
        categorical_tie_break="hash",
        categorical_tie_break_seed=17,
        high_cardinality_policy="top_k",
        categorical_low_cardinality_policy="all",
        text_categorical_fallback=True,
    )

    assert config.graphreduce_kwargs()["categorical_cardinality_threshold"] == 32
    assert config.graphreduce_kwargs()["categorical_top_k"] == 20
    assert config.graphreduce_kwargs()["high_cardinality_policy"] == "top_k"
    assert config.graphreduce_kwargs()["categorical_low_cardinality_policy"] == "all"
    assert config.graphreduce_kwargs()["text_categorical_fallback"] is True
    assert config.graphreduce_kwargs()["categorical_tie_break"] == "hash"
    assert config.graphreduce_kwargs()["categorical_tie_break_seed"] == 17

    compatible = GraphConfig(
        include_target_history=False,
        published_cutoff_compatibility=True,
    )
    assert compatible.graphreduce_kwargs()["categorical_tie_break"] == "value"

    exact_legacy = GraphConfig(
        include_target_history=False,
        published_cutoff_compatibility=True,
        categorical_tie_break="legacy",
        legacy_training_row_order=True,
    )
    assert exact_legacy.graphreduce_kwargs()["categorical_tie_break"] == "legacy"

    with pytest.raises(ValueError, match="high_cardinality_policy"):
        GraphConfig(high_cardinality_policy="invalid")
    with pytest.raises(ValueError, match="categorical_low_cardinality_policy"):
        GraphConfig(categorical_low_cardinality_policy="invalid")
    with pytest.raises(ValueError, match="categorical_tie_break"):
        GraphConfig(categorical_tie_break="invalid")
    with pytest.raises(ValueError, match="categorical_cardinality_threshold"):
        GraphConfig(categorical_cardinality_threshold=0)
    with pytest.raises(ValueError, match="categorical_top_k"):
        GraphConfig(categorical_top_k=-1)
    (compact_modern_configs,)
