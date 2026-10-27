import pickle

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from kurversc import PeerContext, Relationship, Table
from kurversc.paired_peer_context import PairedPeerContext, _bounded_memberships, _paired_memberships


def fixture(history=None):
    pairs = [(0, 0)] * 3 + [(0, 1)] * 3 + [(1, 0)] * 3 + [(1, 1)] * 3 + [(0, 0), (0, 1)]
    tables = {
        "roots": Table(pd.DataFrame({"id": range(1, 15)}), key="id", timeless=True),
        "events": Table(pd.DataFrame({"eid": range(1, 15), "root": range(1, 15),
                                      "a": [a for a, _ in pairs], "b": [b for _, b in pairs],
                                      "date": pd.Timestamp("1999-01-01")}), key="eid", date="date"),
        "a": Table(pd.DataFrame({"aid": [0, 1]}), key="aid", timeless=True),
        "b": Table(pd.DataFrame({"bid": [0, 1]}), key="bid", timeless=True),
    }
    edges = [Relationship("roots", "events", "id", "root"),
             Relationship("events", "a", "a", "aid", reduce=False),
             Relationship("events", "b", "b", "bid", reduce=False)]
    if history is None:
        history = pd.DataFrame({"id": range(1, 13), "time": pd.Timestamp("2000-01-01"),
                                "ready": pd.Timestamp("2001-01-01"),
                                "y": [float(a != b) for a, b in pairs[:12]]})
    reference = PeerContext(mode="unweighted", max_paths=2, large_max_paths=2,
                            max_neighbors=8, large_max_neighbors=8, max_history_rows=100).fit(
        history, tables=tables, relationships=edges, root_name="roots", entity_key="id",
        timestamp="time", target="y", available_at="ready")
    return PairedPeerContext(reference, history, max_contexts=2), history, tables


def query(ids=(13, 14), cutoff="2002-01-01"):
    return pd.DataFrame({"id": ids, "time": pd.Timestamp(cutoff)})


def test_conjunction_separates_identical_single_context_means():
    plan, _, _ = fixture()
    single, joint = plan.transform(query())
    assert single.filter(regex="__mean$").eq(.5).all().all()
    assert joint.paired_joint_p0__mean.tolist() == [0., 1.]
    assert joint.paired_joint_p0__support_mean.tolist() == [3., 3.]
    assert joint.paired_joint_p0__shrunk_mean.tolist() == [.3125, .6875]
    assert not any("key" in name for name in joint)


def test_strict_label_boundary_and_leave_own_root_out():
    plan, history, _ = fixture()
    _, boundary = plan.transform(query(cutoff="2001-01-01"))
    assert boundary.paired_joint_p0__supported_groups.eq(0).all()
    assert boundary.paired_joint_p0__mean.isna().all()
    before = plan.transform(query(ids=(1,)))
    changed, _, _ = fixture(history.assign(y=[900.] + history.y.tolist()[1:]))
    after = changed.transform(query(ids=(1,)))
    for a, b in zip(before, after):
        assert_frame_equal(a, b)
    assert before[1].paired_joint_p0__support_mean.iloc[0] == 2


def test_future_labels_and_source_rows_do_not_change_past_features():
    plan, history, tables = fixture()
    expected = plan.transform(query())
    future = history.assign(time=pd.Timestamp("2025-01-01"), ready=pd.Timestamp("2026-01-01"), y=900.)
    other, _, _ = fixture(pd.concat([history, future], ignore_index=True))
    actual = other.transform(query())
    for a, b in zip(expected, actual):
        assert_frame_equal(a, b)
    tables["events"].source.loc[tables["events"].source.root.eq(13), "date"] = pd.Timestamp("2025-01-01")
    _, hidden = plan.transform(query(ids=(13,)))
    assert hidden.paired_joint_p0__groups.iloc[0] == 0


def test_reordering_batching_and_pickle_replay():
    plan, history, _ = fixture()
    expected = plan.transform(query(ids=(13, 14, 1)))
    restored = pickle.loads(pickle.dumps(plan))
    actual = restored.transform(query(ids=(1, 13, 14)))
    for a, b in zip(expected, actual):
        assert_frame_equal(a.iloc[[2, 0, 1]].reset_index(drop=True), b)
    one = restored.transform(query(ids=(14,)))
    for a, b in zip(expected, one):
        assert_frame_equal(a.iloc[[1]].reset_index(drop=True), b)
    shuffled, _, _ = fixture(history.sample(frac=1, random_state=5))
    for a, b in zip(expected, shuffled.transform(query(ids=(13, 14, 1)))):
        assert_frame_equal(a, b)


def test_pair_budget_and_selection_are_order_invariant():
    rows = pd.DataFrame({"__example_id": [0] * 20 + [1] * 20, "__neighbor_key_0": list(range(20)) * 2})
    left = _bounded_memberships(rows, 8)
    assert left.groupby("__example_id").size().eq(8).all()
    bounded = _paired_memberships(left, left, 13)
    assert bounded.groupby("__example_id").size().eq(13).all()
    assert_frame_equal(left, _bounded_memberships(rows.sample(frac=1, random_state=5), 8))
    assert_frame_equal(bounded, _paired_memberships(left.sample(frac=1), left.sample(frac=1), 13))


def test_empty_queries_and_invalid_budgets():
    plan, _, _ = fixture()
    empty = query().iloc[:0]
    single, joint = plan.transform(empty)
    assert single.shape == (0, 16)
    assert joint.shape == (0, 8)
    with pytest.raises(ValueError, match="Positive"):
        PairedPeerContext(plan.reference, plan.history, max_pairs=0)


def test_mixed_cutoffs_duplicate_queries_and_unseen_entities():
    plan, _, _ = fixture()
    rows = query(ids=(13, 13, 14, 99))
    rows.loc[0, "time"] = pd.Timestamp("2001-01-01")
    one, two = plan.transform(rows)
    assert two.paired_joint_p0__supported_groups.tolist() == [0., 1., 1., 0.]
    assert two.paired_joint_p0__mean.iloc[[0, 3]].isna().all()
    for position in range(len(rows)):
        for expected, actual in zip((one, two), plan.transform(rows.iloc[[position]])):
            assert_frame_equal(expected.iloc[[position]].reset_index(drop=True), actual)


def test_sparse_joint_groups_fall_back_to_broader_contexts():
    plan, _, _ = fixture()
    plan.minimum_support = 4
    single, joint = plan.transform(query())
    assert joint.paired_joint_p0__supported_groups.eq(0).all()
    assert joint.paired_joint_p0__mean.isna().all()
    assert joint.paired_joint_p0__shrunk_mean.eq(.5).all()
    assert single.filter(regex="__support_mean$").eq(6.).all().all()


def test_pair_plan_and_memberships_do_not_depend_on_target_values():
    original, history, _ = fixture()
    changed, _, _ = fixture(history.assign(y=np.arange(len(history), dtype=float)))
    assert original.contract() == changed.contract()
    for path in original.paths_:
        assert_frame_equal(original.reference.peer_.memberships_[path], changed.reference.peer_.memberships_[path])
