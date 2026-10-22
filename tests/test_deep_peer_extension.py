import pickle

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from kurversc import Relationship, Table
from kurversc.signal_challengers import ChallengerUnavailable
from scripts.deep_peer_extension import DeepPeerExtension


def fixture():
    tables = {
        "roots": Table(pd.DataFrame({"id": [1, 2, 3, 4]}), key="id", timeless=True),
        "sessions": Table(pd.DataFrame({"sid": [1, 2, 3, 4], "root": [1, 2, 3, 4],
            "date": pd.to_datetime(["1999-01-01"] * 4)}), key="sid", date="date"),
        "events": Table(pd.DataFrame({"eid": [1, 2, 3, 4], "session": [1, 2, 3, 4],
            "object": [10] * 4, "date": pd.to_datetime(["1999-01-01"] * 4)}), key="eid", date="date"),
        "objects": Table(pd.DataFrame({"oid": [10]}), key="oid", timeless=True),
    }
    edges = [Relationship("roots", "sessions", "id", "root"),
             Relationship("sessions", "events", "sid", "session"),
             Relationship("events", "objects", "object", "oid", reduce=False)]
    history = pd.DataFrame({"__event_entity": [1, 2, 3],
        "__event_cutoff": pd.to_datetime(["2000-01-01"] * 3),
        "__available_at": pd.to_datetime(["2001-01-01"] * 3), "__outcome": [0., 1., .2]})
    return tables, edges, history


def fit(history=None, **kwargs):
    tables, edges, original = fixture()
    return DeepPeerExtension().fit(original if history is None else history,
        tables=tables, relationships=edges, root_name="roots", phase_boundary="2002-01-01", **kwargs)


def query(time="2002-01-01"):
    return pd.DataFrame({"__event_entity": [1, 2, 3, 4],
        "__event_cutoff": pd.to_datetime([time] * 4)})


def test_only_three_hop_bridge_and_bounded_aggregate_payload():
    provider = fit()
    result = provider.transform(query())
    assert provider.audit["paths"] == [("sessions", "events", "objects")]
    assert result.shape == (4, 25)
    assert result.loc[3, "deep__peer_p0__mean__mean"] == pytest.approx(.4)
    assert result.loc[3, "deep__peer_p0__support__mean"] == 3
    assert not any(c in result for c in ("id", "__event_entity", "__event_cutoff", "oid"))


def test_strict_maturity_and_own_label_exclusion():
    provider = fit()
    boundary = provider.transform(query("2001-01-01"))
    assert boundary.deep__peer_p0__supported_entities.eq(0).all()
    assert boundary.deep__weighted_p0__weighted_mass.eq(0).all()
    _, _, history = fixture()
    changed = fit(history.assign(__outcome=[900., 1., .2]))
    assert_frame_equal(provider.transform(query()).iloc[:1], changed.transform(query()).iloc[:1])


def test_phase_future_labels_do_not_affect_features_or_sample():
    _, _, history = fixture()
    future = history.assign(__event_cutoff=pd.Timestamp("2025-01-01"),
                            __available_at=pd.Timestamp("2026-01-01"), __outcome=999.)
    original = fit(history, max_history_rows=2)
    augmented = fit(pd.concat([history, future], ignore_index=True), max_history_rows=2)
    assert_frame_equal(original.transform(query()), augmented.transform(query()))


def test_batch_order_source_order_and_pickle_replay():
    _, _, history = fixture()
    provider = fit(history, max_history_rows=2)
    shuffled = fit(history.sample(frac=1, random_state=7), max_history_rows=2)
    result = provider.transform(query())
    assert_frame_equal(result, shuffled.transform(query()))
    restored = pickle.loads(pickle.dumps(provider))
    order = [2, 0, 3, 1]
    assert_frame_equal(result.iloc[order].reset_index(drop=True),
                       restored.transform(query().iloc[order]))
    assert_frame_equal(result.iloc[1:2].reset_index(drop=True), restored.transform(query().iloc[1:2]))


def test_dimension_lookup_chains_do_not_consume_bridge_slots():
    tables, edges, history = fixture()
    tables["sessions"].source["dimension"] = 20
    tables["a_dimension"] = Table(pd.DataFrame({"did": [20], "other": [30]}), key="did", timeless=True)
    tables["aa_dimension"] = Table(pd.DataFrame({"did": [30]}), key="did", timeless=True)
    edges.extend([Relationship("sessions", "a_dimension", "dimension", "did", reduce=False),
                  Relationship("a_dimension", "aa_dimension", "other", "did", reduce=False)])
    provider = DeepPeerExtension().fit(history, tables=tables, relationships=edges,
        root_name="roots", phase_boundary="2002-01-01", max_paths=1)
    assert provider.audit["paths"] == [("sessions", "events", "objects")]


def test_unavailable_schema_fails_closed():
    tables, edges, history = fixture()
    with pytest.raises(ChallengerUnavailable, match="three-hop"):
        DeepPeerExtension().fit(history, tables=tables, relationships=edges[:2],
            root_name="roots", phase_boundary="2002-01-01")


def test_serialized_date_strings_have_identical_memberships():
    _, _, history = fixture()
    encoded = history.copy()
    for column in ("__event_cutoff", "__available_at"):
        encoded[column] = encoded[column].astype(str)
    request = query()
    request["__event_cutoff"] = request.__event_cutoff.astype(str)
    assert_frame_equal(fit(history).transform(query()), fit(encoded).transform(request))
