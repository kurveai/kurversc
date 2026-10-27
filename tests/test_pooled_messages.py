import pickle

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

pytest.importorskip("torch")

from kurversc import PooledMessageFamily, Table, Relationship
from kurversc.pooled_messages import NeighborhoodEncoder


def bag_data():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(192, 4, 1)).astype(np.float32)
    y = x.mean(axis=1).ravel()
    return x.reshape(-1, 1), np.repeat(y, 4), np.repeat(np.arange(192), 4), y


def root_predictions(model, x):
    return model.predict_pooled(model.encode(x).reshape(192, 4, -1).mean(axis=1))


def test_parent_loss_learns_neighborhood_target_and_differs_from_row_loss():
    x, y, groups, parent_y = bag_data()
    shared = dict(width=1, hidden=16, epochs=80, batch_size=64, random_state=4)
    parent = NeighborhoodEncoder(objective="parent", **shared).fit(x, y, groups)
    row = NeighborhoodEncoder(objective="row", **shared).fit(x, y, groups)
    parent_error = np.mean(abs(root_predictions(parent, x) - parent_y))
    row_error = np.mean(abs(root_predictions(row, x) - parent_y))
    assert parent_error < row_error * 0.8
    assert parent.fit_audit_["objective"] == "parent"
    assert (
        parent.fit_audit_["loss_trace"][-1]["loss"]
        < parent.fit_audit_["loss_trace"][0]["loss"]
    )


def test_encoder_is_deterministic_serializable_and_handles_missing_values():
    x, y, groups, _ = bag_data()
    x[::9] = np.nan
    kwargs = dict(width=3, epochs=3, hidden=8, random_state=0)
    first = NeighborhoodEncoder(**kwargs).fit(x, y, groups)
    second = NeighborhoodEncoder(**kwargs).fit(x, y, groups)
    np.testing.assert_array_equal(first.encode(x), second.encode(x))
    restored = pickle.loads(pickle.dumps(first))
    np.testing.assert_array_equal(
        root_predictions(first, x), root_predictions(restored, x)
    )
    np.testing.assert_allclose(first.encode(x[:1]), first.encode(x)[:1], atol=1e-6)


def test_parent_ids_cannot_mix_different_targets():
    with pytest.raises(ValueError, match="same target"):
        NeighborhoodEncoder(epochs=1).fit(np.array([[0.0], [1.0]]), [0, 1], [2, 2])


def fixture():
    tables = {
        "roots": Table(pd.DataFrame({"id": range(5)}), key="id", timeless=True),
        "events": Table(
            pd.DataFrame(
                {
                    "record": range(10),
                    "root": np.repeat(range(5), 2),
                    "signal": np.arange(10) * 0.2,
                    "date": pd.to_datetime(["2000-01-01"] * 8 + ["2040-01-01"] * 2),
                }
            ),
            key="record",
            date="date",
        ),
    }
    edges = [Relationship("roots", "events", "id", "root")]
    training = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "y": [0.0, 1.0, 0.3, 0.7],
            "time": pd.to_datetime(["2001-01-01"] * 2 + ["2003-01-01"] * 2),
            "ready": pd.to_datetime(["2002-01-01"] * 2 + ["2004-01-01"] * 2),
        }
    )
    query = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "time": pd.to_datetime(
                ["2002-01-01", "2003-01-01", "2003-01-01", "2005-01-01", "2005-01-01"]
            ),
        }
    )
    return tables, edges, training, query


def fit_family(training=None, classification=False):
    tables, edges, source, query = fixture()
    source = source if training is None else training
    if classification:
        source["y"] = [0, 1, 0, 1]
    family = PooledMessageFamily(message_width=3, encoder_epochs=3, thread_count=1)
    result = family.fit_transform(
        source,
        query,
        tables=tables,
        relationships=edges,
        root_name="roots",
        entity_key="id",
        timestamp="time",
        target="y",
        available_at="ready",
        task="classification" if classification else "regression",
    )
    return family, result


def test_family_keeps_strict_label_availability_and_frozen_inference():
    family, (raw, messages) = fit_family()
    tables, edges, training, query = fixture()
    training.loc[2:, "y"] = [999.0, -999.0]
    _, (_, poisoned) = fit_family(training)
    assert_frame_equal(messages.iloc[:3], poisoned.iloc[:3])
    assert messages.loc[0, "learned_p0__available"] == 0
    assert messages.loc[4, "learned_p0__available"] == 0
    assert pd.isna(messages.loc[0, "learned_p0__parent_prediction"])
    restored = pickle.loads(pickle.dumps(family))
    query["y"] = 777
    replay = restored.transform(query, tables=tables, relationships=edges)
    assert_frame_equal(raw, replay[0])
    assert_frame_equal(messages, replay[1])
    alone = restored.transform(query.iloc[1:2], tables=tables, relationships=edges)
    np.testing.assert_allclose(messages.iloc[1:2, 2:], alone[1].iloc[:, 2:], atol=1e-6)
    assert all(
        pd.Timestamp(a["latest_label_available"]) < pd.Timestamp(a["cutoff"])
        for a in family.audit
    )
    assert family.as_table(messages)[0].cutoff_boundary == "exact"


def test_binary_family_emits_bounded_parent_probabilities():
    _, (_, messages) = fit_family(classification=True)
    values = messages["learned_p0__parent_prediction"].dropna()
    assert len(values) == 3
    assert values.between(0, 1).all()


def test_encoder_seed_does_not_change_extracted_neighborhoods():
    tables, edges, training, query = fixture()
    outputs = []
    for seed in (0, 1):
        family = PooledMessageFamily(
            random_state=0,
            encoder_random_state=seed,
            max_neighbors=1,
            encoder_epochs=2,
            message_width=1,
        )
        outputs.append(
            family.fit_transform(
                training,
                query,
                tables=tables,
                relationships=edges,
                root_name="roots",
                entity_key="id",
                timestamp="time",
                target="y",
                available_at="ready",
            )
        )
    assert_frame_equal(outputs[0][0], outputs[1][0])
    assert not outputs[0][1].equals(outputs[1][1])
