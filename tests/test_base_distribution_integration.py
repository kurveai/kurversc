"""The base-CDF extension must work without expanding graph configuration search."""

from dataclasses import replace
import pickle

import numpy as np
import pandas as pd
import pytest

import kurversc
from graphreduce.node import SQLNode


@pytest.mark.parametrize(
    ("published", "cdf_override", "expected_cdf"),
    [
        (False, None, True),
        (True, None, False),
        (False, False, False),
        (True, True, True),
    ],
)
def test_base_cdf_default_legacy_reference_and_frozen_prediction(
    published, cdf_override, expected_cdf, monkeypatch
):
    root = kurversc.Table(
        pd.DataFrame({"entity_id": range(30)}),
        name="entities",
        key="entity_id",
        timeless=True,
    )
    events = kurversc.Table(
        pd.DataFrame(
            {
                "event_id": range(180),
                "entity_id": np.repeat(range(30), 6),
                "amount": np.arange(180) % 17 + 0.25,
                "date": pd.to_datetime(["2024-01-01"] * 180),
            }
        ),
        name="events",
        key="event_id",
        date="date",
        prefix="evt",
    )
    relationships = [
        kurversc.Relationship("entities", "events", "entity_id", "entity_id")
    ]
    labels = pd.DataFrame(
        {
            "entity_id": range(30),
            "target": np.arange(30) % 2,
            "split": ["train"] * 20 + ["validation"] * 10,
            "timestamp": pd.to_datetime(["2024-06-01"] * 20 + ["2025-06-01"] * 10),
        }
    )
    config = kurversc.GraphConfig(
        auto_annotate_features=False,
        include_target_history=False,
        published_cutoff_compatibility=published,
    )
    result = kurversc.fit(
        parent_node=root,
        label_node=kurversc.Labels(
            labels,
            key="entity_id",
            target="target",
            split="split",
            timestamp="timestamp",
            target_history=False,
        ),
        tables=[events],
        relationships=relationships,
        graph_configs=[config],
        search_full_data=True,
        rerank_top_k=0,
        model_params={"iterations": 3, "depth": 2, "thread_count": 1},
        base_distribution_features=cdf_override,
    )
    assert len(result.trials) == 1
    assert result.trials[0].status == "completed"
    assert result.best_config.feature_families == ("base",)
    cdf_columns = [c for c in result.fitted_model.feature_columns if "__cdf_" in c]
    assert bool(cdf_columns) == expected_cdf
    assert len(cdf_columns) <= 8

    def forbid(*args, **kwargs):
        raise AssertionError("Prediction must not learn CDF thresholds")

    monkeypatch.setattr(SQLNode, "get_base_distribution_sample", forbid)
    restored = pickle.loads(pickle.dumps(result))
    queries = labels[["entity_id", "timestamp"]].iloc[:5]
    kwargs = dict(
        parent_node=root,
        prediction_node=queries,
        label_key="entity_id",
        label_timestamp="timestamp",
        tables=[events],
        relationships=relationships,
    )
    pd.testing.assert_frame_equal(
        kurversc.predict(result, **kwargs), kurversc.predict(restored, **kwargs)
    )


def test_preflight_accounts_for_cdf_width_without_new_family():
    from kurversc.feature_audit import _local_reduction_feature_estimate

    audit = pd.DataFrame(
        {
            "role": ["numerical"] * 20,
            "eligible": [True] * 20,
            "structural": [False] * 20,
            "eligible_families": ["base"] * 20,
        }
    )
    table = kurversc.Table(pd.DataFrame(), key="id", timeless=True)
    config = kurversc.GraphConfig(
        auto_annotate_features=False,
        include_target_history=False,
        feature_family_max_columns=None,
    )
    modern = _local_reduction_feature_estimate(
        table, audit, config, annotation_features=0
    )
    legacy = _local_reduction_feature_estimate(
        table,
        audit,
        replace(config, published_cutoff_compatibility=True),
        annotation_features=0,
    )
    disabled = _local_reduction_feature_estimate(
        table,
        audit,
        replace(config, base_distribution_features=False),
        annotation_features=0,
    )
    assert modern - legacy == 16 * 8
    assert disabled == legacy
