import pytest

from kurversc import GraphConfig, incremental_configs
from kurversc.search import (
    Trial,
    adaptive_depth_candidate_allowed,
    diverse_confirmation_trials,
    forward_candidate_allowed,
    stability_adjusted_winner,
)


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
    assert len(configs) == 72
    assert {config.feature_family_max_columns for config in configs} == {4}
    assert all(
        config.feature_family_max_features_per_column == 32 for config in configs
    )
    assert all(
        config.feature_propagation_max_functions_per_column == 1 for config in configs
    )
    assert all("semantic" not in c.feature_families for c in configs)
    assert all("context" not in c.feature_families for c in configs)
    assert {c.auto_annotate_features for c in configs if c.depth > 1} == {True, False}
    assert all(
        set(c.feature_families).issubset({"base", "temporal", "sequence"})
        for c in configs
        if c.depth == 3
    )
    assert {c.feature_families for c in configs if len(c.feature_families) == 2} == {
        ("base", "temporal"),
        ("base", "sequence"),
        ("base", "conditional"),
        ("base", "episode"),
    }
    assert configs[-1].feature_families == (
        "base",
        "temporal",
        "sequence",
        "conditional",
        "episode",
    )
    assert configs[-1].feature_family_max_columns == 4


def test_incremental_configs_only_expand_column_budget_when_requested() -> None:
    configs = incremental_configs(
        max_depth=3,
        feature_family_max_column_options=(4, 8),
    )

    assert len(configs) == 144
    assert {config.feature_family_max_columns for config in configs} == {4, 8}


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
