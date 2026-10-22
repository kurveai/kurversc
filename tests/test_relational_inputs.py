import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from graphreduce.learned_messages import extract_memberships, extract_path
from kurversc import (
    Table, Relationship, SemanticPayloadFamily, PeerHistoryFamily, PeerContext,
)
from kurversc.relational_inputs import PayloadEncoder


def fixture():
    tables = {
        "roots": Table(
            pd.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "title": [
                        "blue drug trial",
                        "blue drug study",
                        "red disease trial",
                        "future secret word",
                    ],
                    "size": [10.0, 20.0, 30.0, 40.0],
                }
            ),
            key="id",
            timeless=True,
        ),
        "links": Table(
            pd.DataFrame(
                {
                    "row": [0, 1, 2, 3, 4, 5],
                    "root": [1, 2, 3, 4, 1, 1],
                    "group": [10, 10, 20, 20, 20, 10],
                    "date": pd.to_datetime(
                        ["2000-01-01"] * 4 + ["2040-01-01", "2000-01-01"]
                    ),
                }
            ),
            key="row",
            date="date",
        ),
        "groups": Table(
            pd.DataFrame(
                {"gid": [10, 20], "description": ["blue treatment", "red condition"]}
            ),
            key="gid",
            timeless=True,
        ),
    }
    edges = [
        Relationship("roots", "links", "id", "root"),
        Relationship("links", "groups", "group", "gid", reduce=False),
    ]
    training = pd.DataFrame(
        {
            "id": [1, 2, 3, 1],
            "time": pd.to_datetime(["2001-01-01"] * 3 + ["2003-01-01"]),
            "ready": pd.to_datetime(["2002-01-01"] * 3 + ["2004-01-01"]),
            "y": [0.0, 1.0, 0.2, 0.8],
        }
    )
    query = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "time": pd.to_datetime(
                ["2002-01-01", "2003-01-01", "2003-01-01", "2005-01-01"]
            ),
        }
    )
    return tables, edges, training, query


def fit_peer(training=None):
    tables, edges, history, query = fixture()
    family = PeerHistoryFamily(prior_strength=2, memory_limit="1GB")
    values = family.fit_transform(
        history if training is None else training,
        query,
        tables=tables,
        relationships=edges,
        root_name="roots",
        entity_key="id",
        timestamp="time",
        target="y",
        available_at="ready",
    )
    return family, values


def test_graphreduce_memberships_are_bounded_identity_metadata_only():
    tables, edges, _, query = fixture()
    args = dict(
        root_name="roots",
        entity_key="id",
        timestamp="time",
        path=("links", "groups"),
        max_neighbors=2,
        memory_limit="1GB",
    )
    identities = extract_memberships(query, tables, edges, **args)
    assert list(identities) == ["__example_id", "__neighbor_key_0"]
    assert identities.groupby("__example_id").size().max() == 1
    assert identities.loc[
        identities.__example_id.eq(0), "__neighbor_key_0"
    ].tolist() == [10]
    payload = extract_path(query, tables, edges, include_terminal_keys=True, **args)
    assert_frame_equal(identities, payload[list(identities)])
    shuffled = {
        k: replace(v, source=v.source.sample(frac=1, random_state=8))
        for k, v in tables.items()
    }
    assert_frame_equal(identities, extract_memberships(query, shuffled, edges, **args))


def test_peer_memberships_are_stable_for_unkeyed_intermediate_events():
    events = pd.DataFrame(
        {
            "root": [1] * 40,
            "group": list(range(40)),
            "date": pd.to_datetime(["2000-01-01"] * 40),
        }
    )
    base = {
        "roots": Table(pd.DataFrame({"id": [1]}), key="id", timeless=True),
        "events": Table(events, date="date"),
        "groups": Table(
            pd.DataFrame({"gid": list(range(40))}), key="gid", timeless=True
        ),
    }
    edges = [
        Relationship("roots", "events", "id", "root"),
        Relationship("events", "groups", "group", "gid", reduce=False),
    ]
    history = pd.DataFrame(
        {
            "id": [1], "time": pd.to_datetime(["2001-01-01"]),
            "ready": pd.to_datetime(["2001-01-02"]), "y": [1.0],
        }
    )

    def identities(tables):
        peer = PeerHistoryFamily(max_neighbors=4, memory_limit="1GB").fit(
            history, tables=tables, relationships=edges, root_name="roots",
            entity_key="id", timestamp="time", target="y", available_at="ready",
        )
        rows = peer._members(
            history[["id", "time"]], tables, edges, ("events", "groups")
        )
        return rows["__neighbor_key_0"].tolist()

    shuffled = {
        **base,
        "events": replace(
            base["events"], source=events.sample(frac=1, random_state=8)
        ),
    }
    assert identities(base) == identities(shuffled)


def test_peer_strict_cutoff_deduplication_and_other_entity_only():
    family, values = fit_peer()
    assert values.loc[0, "peer_p0__supported_entities"] == 0
    assert np.isnan(values.loc[0, "peer_p0__shrunk_mean__mean"])
    assert values.loc[1, "peer_p0__support__mean"] == 1
    assert values.loc[1, "peer_p0__mean__mean"] == 0.0
    # Query root 3 has only itself in group 20. Global prior excludes root 3.
    assert values.loc[2, "peer_p0__support__mean"] == 0
    assert values.loc[2, "peer_p0__shrunk_mean__mean"] == 0.5
    assert values.loc[3, "peer_p0__mean__mean"] == 0.2
    assert all("neighbor_key" not in c for c in values)
    tables, edges, _, query = fixture()
    restored = pickle.loads(pickle.dumps(family))
    query["y"] = 999
    assert_frame_equal(
        values, restored.transform(query, tables=tables, relationships=edges)
    )
    assert_frame_equal(
        values.iloc[2:3],
        restored.transform(query.iloc[2:3], tables=tables, relationships=edges),
    )
    assert family.as_table(values, name="peer")[0].cutoff_boundary == "exact"


def test_future_and_own_labels_cannot_change_a_peer_feature():
    _, original = fit_peer()
    _, _, training, _ = fixture()
    training.loc[3, "y"] = 999
    _, future = fit_peer(training)
    assert_frame_equal(original.iloc[:3], future.iloc[:3])
    training.loc[1, "y"] = 500
    _, own = fit_peer(training)
    assert_frame_equal(original.iloc[1:2], own.iloc[1:2])


def test_bounded_peer_context_preserves_strict_and_leave_own_out_rules():
    tables, edges, training, query = fixture()

    def features(history):
        return PeerContext(mode="both", batch_rows=2).fit(
            history,
            tables=tables,
            relationships=edges,
            root_name="roots",
            entity_key="id",
            timestamp="time",
            target="y",
            available_at="ready",
        ).transform(query)

    baseline = features(training)
    assert baseline.loc[0, "peer_p0__supported_entities"] == 0
    assert baseline.loc[1, "peer_p0__mean__mean"] == 0.0
    assert baseline.loc[1, "weighted_p0__weighted_mean"] == 0.0
    future = training.copy()
    future.loc[3, "y"] = 999.0
    assert_frame_equal(baseline.iloc[:3], features(future).iloc[:3])
    own = training.copy()
    own.loc[1, "y"] = 500.0
    assert_frame_equal(baseline.iloc[1:2], features(own).iloc[1:2])


def test_peer_uses_latest_mature_label_per_root_and_historical_memberships():
    family, _ = fit_peer()
    tables, edges, _, _ = fixture()
    query = pd.DataFrame({"id": [2], "time": pd.to_datetime(["2005-01-01"])})
    result = family.transform(query, tables=tables, relationships=edges)
    assert result.loc[0, "peer_p0__support__mean"] == 1
    assert result.loc[0, "peer_p0__mean__mean"] == 0.8
    # The future root-1/group-20 link cannot carry the mature root-1 label.
    query["id"] = 4
    assert (
        family.transform(query, tables=tables, relationships=edges).loc[
            0, "peer_p0__mean__mean"
        ]
        == 0.2
    )


def test_peer_invalid_availability_and_empty_input_rejected():
    _, _, training, _ = fixture()
    training.loc[0, "ready"] = pd.Timestamp("1990-01-01")
    with pytest.raises(ValueError, match="availability"):
        fit_peer(training)
    with pytest.raises(ValueError, match="nonempty"):
        fit_peer(training.iloc[:0])


def test_payload_encoder_preserves_structured_values_and_shares_tokens():
    frame = pd.DataFrame(
        {
            "description": [
                "blue medicine trial",
                "blue medicine study",
                "red disease study",
                "red disease trial",
            ],
            "phase": ["one", "one", "two", "two"],
            "enrollment": [1.0, 2.0, 3.0, 4.0],
        }
    )
    encoder = PayloadEncoder(components=2).fit(frame)
    assert "enrollment" in encoder.numeric_columns_
    assert ("phase", "one") in encoder.categories_
    assert "medicine" in encoder.vectorizer_.vocabulary_
    expected = encoder.transform(frame)
    np.testing.assert_array_equal(
        expected, pickle.loads(pickle.dumps(encoder)).transform(frame)
    )
    np.testing.assert_allclose(
        expected.iloc[:1], encoder.transform(frame.iloc[:1]), atol=1e-6
    )


def test_semantic_family_training_only_vocabulary_and_frozen_replay():
    tables, edges, training, query = fixture()
    earliest = training.iloc[:3][["id", "time"]]
    family = SemanticPayloadFamily(components=2, memory_limit="1GB")
    output = family.fit_transform(
        earliest,
        query,
        tables=tables,
        relationships=edges,
        root_name="roots",
        entity_key="id",
        timestamp="time",
    )
    assert all(
        "secret" not in e.vectorizer_.vocabulary_
        for e in family.encoders_.values()
        if e.vectorizer_
    )
    assert not any("numeric_id" in c or "numeric_gid" in c for c in output)
    query["y"] = 999
    restored = pickle.loads(pickle.dumps(family))
    assert_frame_equal(
        output, restored.transform(query, tables=tables, relationships=edges)
    )
    np.testing.assert_allclose(
        output.iloc[1:2, 2:],
        restored.transform(query.iloc[1:2], tables=tables, relationships=edges).iloc[
            :, 2:
        ],
        atol=1e-6,
    )
    assert family.as_table(output, name="payload")[0].cutoff_boundary == "exact"
    before = query.iloc[:1].copy()
    before["time"] = pd.Timestamp("1990-01-01")
    with pytest.raises(ValueError, match="precede"):
        family.transform(before, tables=tables, relationships=edges)


def test_semantic_family_honors_explicit_column_allowlists():
    tables, edges, training, query = fixture()
    tables["groups"] = replace(tables["groups"], columns=())
    tables["roots"] = replace(tables["roots"], columns=("id", "size"))
    family = SemanticPayloadFamily(components=2, memory_limit="1GB")
    family.fit_transform(
        training,
        query,
        tables=tables,
        relationships=edges,
        root_name="roots",
        entity_key="id",
        timestamp="time",
    )
    assert family.paths_ == ((),)
    assert family.encoders_[()].text_columns_ == ()
