"""Cyclic replay preserves strict event and independent availability boundaries."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import kurversc
import kurversc.core as core
from kurversc.relbench import _edge_instances


def graph(extra=None):
    people = pd.DataFrame({"key": [1, 2], "measure": [10., 20.]})
    events = pd.DataFrame({"key": [10, 11], "owner": [1, 2], "peer": [2, 1],
        "amount": [1., 2.], "clock": pd.Timestamp("2020-01-01"), "ready": pd.Timestamp("2020-01-01")})
    if extra is not None:
        events = pd.concat([events, extra], ignore_index=True)
    for field in ["clock", "ready"]:
        events[field] = pd.to_datetime(events[field]).astype("datetime64[ns]")
    metadata = {
        "people": SimpleNamespace(df=people, pkey_col="key", time_col=None, fkey_col_to_pkey_table={}),
        "events": SimpleNamespace(df=events, pkey_col="key", time_col="clock", available_col="ready",
            cutoff_boundary="strict", fkey_col_to_pkey_table={"owner": "people", "peer": "people"}),
    }
    parent, children, edges = _edge_instances(SimpleNamespace(table_dict=metadata, exclude_structural_aliases=True),
        root_table="people", root_frame=people, search_root_frame=people, schema_depth=3,
        sample_rows=None, random_state=0, max_base_table_occurrences=2)
    return parent, children, edges


def test_cyclic_frozen_replay_excludes_exact_cutoff_future_and_unavailable_source_rows():
    parent, children, edges = graph()
    labels = pd.DataFrame({"key": [1, 2, 1, 2], "target": [0, 1, 0, 1],
        "when": pd.to_datetime(["2020-01-02"] * 2 + ["2020-01-03"] * 2),
        "split": ["train"] * 2 + ["validation"] * 2})
    result = kurversc.fit(parent_node=parent, tables=children, relationships=edges,
        label_node=kurversc.Labels(labels, key="key", target="target", timestamp="when", split="split"),
        graph_configs=[kurversc.GraphConfig(depth=3, max_base_table_occurrences=2,
            auto_annotate_features=False, include_target_history=False, published_cutoff_compatibility=True)],
        full_training_frames=1, rerank_top_k=0, search_full_data=True, infer_ts_periods=False,
        model_params={"iterations": 2, "depth": 2, "thread_count": 1})
    assert result.execution_plan["kurversc_source_blueprint"]["max_base_table_occurrences"] == 2
    prediction = kurversc.Labels(pd.DataFrame({"key": [1, 2], "when": pd.Timestamp("2020-01-04")}),
        key="key", timestamp="when")
    materialize = core._materialize_at_cutoff

    def predict(source):
        frames = []
        def capture(*args, **kwargs):
            frame = materialize(*args, **kwargs)
            frames.append(frame.copy())
            return frame
        parent, children, edges = source
        with patch.object(core, "_materialize_at_cutoff", capture):
            output = kurversc.predict(result, parent_node=parent, tables=children, relationships=edges,
                prediction_node=prediction, use_validation_model=True)
        return output.prediction.to_numpy(), frames

    expected, original = predict(graph())
    invisible = pd.DataFrame({"key": [20, 21, 22], "owner": [1, 2, 1], "peer": [2, 1, 2],
        "amount": [1e9, 1e9, 1e9],
        "clock": pd.to_datetime(["2020-01-04", "2020-01-05", "2020-01-02"]),
        "ready": pd.to_datetime(["2020-01-04", "2020-01-05", "2020-01-06"])})
    actual, augmented = predict(graph(invisible))
    np.testing.assert_array_equal(actual, expected)
    assert len(original) == len(augmented) > 0
    for first, second in zip(original, augmented, strict=True):
        pd.testing.assert_frame_equal(first, second, check_exact=True)
