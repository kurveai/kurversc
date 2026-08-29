import pytest

from kurversc import GraphConfig, incremental_configs


def test_incremental_configs_start_with_requested_baseline() -> None:
    configs = incremental_configs(max_depth=3)

    assert configs[0] == GraphConfig(
        feature_families=("base",),
        depth=1,
        auto_annotate_features=True,
        feature_family_max_columns=None,
    )
    assert [config.depth for config in configs[:3]] == [1, 2, 3]
    assert configs[3].auto_annotate_features is False
    assert configs[6].feature_families == ("base", "temporal")
    assert len(configs) == 26
    assert all("semantic" not in c.feature_families for c in configs)
    assert all("context" not in c.feature_families for c in configs)
    assert all(
        c.depth < 3
        for c in configs
        if len(c.feature_families) > 3
    )
    assert configs[-1].feature_families == (
        "base",
        "temporal",
        "sequence",
        "conditional",
        "episode",
    )


def test_graphreduce_kwargs_leave_uncapped_budget_to_node_builder() -> None:
    config = GraphConfig(feature_family_max_columns=None)

    assert "feature_family_max_columns" not in config.graphreduce_kwargs()


def test_graph_config_rejects_nonpositive_feature_family_cap() -> None:
    with pytest.raises(ValueError, match="feature_family_max_columns"):
        GraphConfig(feature_family_max_columns=0)
