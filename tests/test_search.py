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


def test_graphreduce_kwargs_leave_uncapped_budget_to_node_builder() -> None:
    config = GraphConfig(feature_family_max_columns=None)

    assert "feature_family_max_columns" not in config.graphreduce_kwargs()
