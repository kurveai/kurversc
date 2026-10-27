import pickle

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from kurversc import PeerContext, Relationship, Table
from kurversc.peer_sampling import CutoffLocalPeerContext, sample_mature_peer_history


def _fixture(*, limit=2):
    tables = {
        "roots": Table(pd.DataFrame({"id": [1, 2, 3, 4]}), key="id", timeless=True),
        "events": Table(
            pd.DataFrame(
                {
                    "event": [1, 2, 3, 4],
                    "root": [1, 2, 3, 4],
                    "group": [10] * 4,
                    "date": pd.to_datetime(["2000-01-01"] * 4),
                }
            ),
            key="event",
            date="date",
        ),
        "groups": Table(pd.DataFrame({"gid": [10]}), key="gid", timeless=True),
    }
    edges = [
        Relationship("roots", "events", "id", "root"),
        Relationship("events", "groups", "group", "gid", reduce=False),
    ]
    history = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "time": pd.to_datetime(["2001-01-01"] * 3),
            "ready": pd.to_datetime(["2002-01-01"] * 3),
            "y": [0.0, 1.0, 0.2],
        }
    )
    reference = PeerContext(
        mode="both",
        max_history_rows=limit,
        max_paths=1,
        large_max_paths=1,
        max_neighbors=2,
        large_max_neighbors=2,
        batch_rows=2,
    ).fit(
        history,
        tables=tables,
        relationships=edges,
        root_name="roots",
        entity_key="id",
        timestamp="time",
        target="y",
        available_at="ready",
    )
    return reference, history


def _query(time="2003-01-01"):
    return pd.DataFrame({"id": [1, 2, 3, 4], "time": pd.to_datetime([time] * 4)})


def test_future_labels_cannot_evict_eligible_history_or_change_features():
    reference, history = _fixture()
    future = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "time": pd.to_datetime(["2025-01-01"] * 4),
            "ready": pd.to_datetime(["2026-01-01"] * 4),
            "y": [900.0] * 4,
        }
    )
    original = CutoffLocalPeerContext(reference, history).transform(_query())
    augmented = CutoffLocalPeerContext(
        reference, pd.concat([history, future], ignore_index=True)
    ).transform(_query())
    assert_frame_equal(original, augmented)


def test_all_eligible_rows_retained_when_they_fit_even_with_future_labels():
    reference, history = _fixture(limit=3)
    future = history.assign(
        time=pd.Timestamp("2025-01-01"), ready=pd.Timestamp("2026-01-01")
    )
    provider = CutoffLocalPeerContext(reference, pd.concat([history, future]))
    result = provider.transform(_query())
    assert provider.audit[0]["selected_history_rows"] == 3
    assert result.loc[3, "peer_p0__support__mean"] == 3


def test_sample_is_bounded_target_blind_and_source_order_invariant():
    _, history = _fixture()
    args = dict(
        entity="id",
        timestamp="time",
        available_at="ready",
        cutoff="2003-01-01",
        limit=2,
    )
    selected = sample_mature_peer_history(history, **args)
    shuffled = sample_mature_peer_history(
        history.sample(frac=1, random_state=7), **args
    )
    assert_frame_equal(selected, shuffled)
    changed = sample_mature_peer_history(history.assign(y=999.0), **args)
    assert_frame_equal(selected.drop(columns="y"), changed.drop(columns="y"))
    assert len(selected) == 2


def test_strict_boundary_empty_history_and_leave_own_entity_out():
    reference, history = _fixture(limit=3)
    provider = CutoffLocalPeerContext(reference, history)
    boundary = provider.transform(_query("2002-01-01"))
    assert list(boundary) == list(reference.transform(_query("2002-01-01")))
    assert boundary.peer_p0__supported_entities.eq(0).all()
    assert boundary.weighted_p0__weighted_mass.eq(0).all()
    assert boundary.peer_p0__mean__mean.isna().all()
    baseline = provider.transform(_query())
    own_changed = CutoffLocalPeerContext(
        reference, history.assign(y=[500.0, 1.0, 0.2])
    ).transform(_query())
    assert_frame_equal(baseline.iloc[:1], own_changed.iloc[:1])
    assert len(provider._snapshots) == 1


def test_mixed_cutoffs_batch_order_and_pickle_replay():
    reference, history = _fixture()
    query = _query()
    query.loc[[0, 2], "time"] = pd.Timestamp("2002-01-01")
    provider = CutoffLocalPeerContext(reference, history)
    mixed = provider.transform(query)
    restored = pickle.loads(pickle.dumps(provider))
    assert_frame_equal(mixed, restored.transform(query))
    reordered = query.iloc[[2, 0, 3, 1]].reset_index(drop=True)
    assert_frame_equal(
        mixed.iloc[[2, 0, 3, 1]].reset_index(drop=True), restored.transform(reordered)
    )
    single = restored.transform(query.iloc[1:2])
    assert_frame_equal(mixed.iloc[1:2].reset_index(drop=True), single)
    assert len(restored._snapshots) == 1
    assert tuple(restored._snapshots.values())[0].peer_.paths_ == reference.peer_.paths_


def test_invalid_history_rejected():
    reference, history = _fixture()
    with pytest.raises(ValueError, match="unique"):
        CutoffLocalPeerContext(reference, pd.concat([history, history]))
    with pytest.raises(ValueError, match="precede"):
        CutoffLocalPeerContext(
            reference, history.assign(ready=pd.Timestamp("1900-01-01"))
        )
    with pytest.raises(ValueError, match="finite"):
        CutoffLocalPeerContext(reference, history.assign(y=np.inf))


def test_weighted_prefix_in_entity_metadata_is_not_dropped():
    base, history = _fixture()
    history = history.rename(columns={"id": "weighted_id"})
    reference = PeerContext(max_history_rows=3).fit(
        history,
        tables=base.tables_,
        relationships=base.relationships_,
        root_name="roots",
        entity_key="weighted_id",
        timestamp="time",
        target="y",
        available_at="ready",
    )
    query = _query().rename(columns={"id": "weighted_id"})
    result = CutoffLocalPeerContext(reference, history).transform(query)
    assert result.weighted_id.tolist() == query.weighted_id.tolist()
