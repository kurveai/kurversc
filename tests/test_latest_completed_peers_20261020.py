import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from kurversc import Relationship, Table
from scripts.latest_completed_peers_20261020 import LatestCompletedPeers, select_latest


def history():
    return pd.DataFrame(dict(__event_entity=[1, 1, 2, 3],
        __event_cutoff=pd.to_datetime(["2000-01-01", "2001-01-01", "2000-01-01", "2000-01-01"]),
        __available_at=pd.to_datetime(["2001-01-01", "2002-01-01", "2001-01-01", "2001-01-01"]),
        __outcome=[0., 1., 1., .2]))


def provider(rows):
    tables = {
        "roots": Table(pd.DataFrame(dict(id=[1, 2, 3, 4])), key="id", timeless=True),
        "sessions": Table(pd.DataFrame(dict(sid=[1, 2, 3, 4], root=[1, 2, 3, 4],
            time=pd.to_datetime(["2000-07-01"]*4))), key="sid", date="time"),
        "events": Table(pd.DataFrame(dict(eid=[1, 2, 3, 4], session=[1, 2, 3, 4], obj=[10]*4,
            time=pd.to_datetime(["2000-07-01"]*4))), key="eid", date="time"),
        "objects": Table(pd.DataFrame(dict(oid=[10])), key="oid", timeless=True)}
    edges = [Relationship("roots", "sessions", "id", "root"), Relationship("sessions", "events", "sid", "session"),
        Relationship("events", "objects", "obj", "oid", reduce=False)]
    return LatestCompletedPeers().fit(rows, tables=tables, relationships=edges, root_name="roots", phase_boundary="2002-01-01")


def query(entity=4, date="2001-01-01"):
    return pd.DataFrame(dict(__event_entity=[entity], __event_cutoff=pd.to_datetime([date])))


def test_latest_before_sampling_is_target_blind_order_invariant_and_point_in_time():
    rows = history()
    kwargs = dict(entity="__event_entity", timestamp="__event_cutoff", available_at="__available_at", cutoff="2001-01-01", limit=2)
    a = select_latest(rows, **kwargs)
    b = select_latest(rows.sample(frac=1, random_state=42), **kwargs)
    assert_frame_equal(a, b)
    c = select_latest(rows.assign(__outcome=[9., 8., 7., 6.]), **kwargs)
    assert_frame_equal(a.drop(columns="__outcome"), c.drop(columns="__outcome"))
    all_rows = select_latest(rows, **{**kwargs, "limit": 3, "cutoff": "2002-01-01"})
    assert all_rows.__event_entity.nunique() == len(all_rows) == 3
    assert all_rows.loc[all_rows.__event_entity.eq(1), "__outcome"].item() == 1.


def test_actual_provider_future_labels_and_own_labels_are_excluded_with_fixed_schema():
    rows = history()
    first = provider(rows)
    at = first.transform(query())
    assert at.loc[0, "latest_completed__peer_p0__mean__mean"] == pytest.approx(.4)
    assert at.loc[0, "latest_completed__peer_p0__support__mean"] == 3
    future_changed = provider(rows.assign(__outcome=[0., 999., 1., .2]))
    assert_frame_equal(at, future_changed.transform(query()))
    own_changed = provider(rows.assign(__outcome=[999., 999., 1., .2]))
    assert_frame_equal(first.transform(query(entity=1)), own_changed.transform(query(entity=1)))
    empty = first.transform(query(date="2000-12-31"))
    assert list(at) == list(empty)
    assert empty.loc[0, "latest_completed__peer_p0__supported_entities"] == 0
    assert_frame_equal(first.transform(query(date="2002-01-01")), first.transform(query(date="2025-01-01")))
