import pickle

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

import torch
from kurversc.joint_neighborhoods import JointNeighborhoodEncoder
from kurversc.neighborhood_inputs import BoundedNeighborhoodInputs
from kurversc import Table, Relationship


def toy():
    torch.set_num_threads(1)
    rng = np.random.default_rng(3)
    roots = pd.Index(np.arange(96) * 3 + 1)
    x = rng.normal(size=(96, 4, 2)).astype(np.float32)
    root = rng.normal(size=96).astype(np.float32)
    y = pd.Series(0.4 * root + (x[:, :, 0] * x[:, :, 1]).mean(axis=1), index=roots)
    bags = {
        "root": pd.DataFrame({"__example_id": roots, "v": root}),
        "neighbors": pd.DataFrame(
            {
                "__example_id": np.repeat(roots, 4),
                "a": x[:, :, 0].ravel(),
                "b": x[:, :, 1].ravel(),
            }
        ),
    }
    return bags, y


def test_joint_readout_is_deterministic_serializable_and_batch_independent():
    bags, y = toy()
    options = dict(epochs=4, hidden=8, width=3, batch_size=32, random_state=0)
    first = JointNeighborhoodEncoder(**options).fit(bags, y)
    second = JointNeighborhoodEncoder(**options).fit(bags, y)
    prediction = first.predict(bags, y.index)
    np.testing.assert_array_equal(prediction, second.predict(bags, y.index))
    restored = pickle.loads(pickle.dumps(first))
    np.testing.assert_array_equal(prediction, restored.predict(bags, y.index))
    np.testing.assert_allclose(
        prediction[4:7], first.predict(bags, y.index[4:7]), atol=1e-6
    )
    shuffled = {
        name: frame.sample(frac=1, random_state=8) for name, frame in bags.items()
    }
    np.testing.assert_allclose(prediction, first.predict(shuffled, y.index), atol=1e-6)


def test_missing_bags_and_binary_predictions():
    bags, y = toy()
    binary = y.gt(y.median()).astype(int)
    model = JointNeighborhoodEncoder(epochs=2, task="classification").fit(bags, binary)
    prediction = model.predict(bags, pd.Index([10000, *y.index[:2]]))
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 1)).all()
    with pytest.raises(ValueError, match="schema/order"):
        model.predict(
            {name: df.rename(columns={"v": "changed"}) for name, df in bags.items()},
            y.index,
        )


def source_fixture():
    tables = {
        "roots": Table(
            pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "number": [1.0, 2.0, 3.0],
                    "narrative": ["small red", "large blue", "secret future"],
                }
            ),
            key="id",
            timeless=True,
        ),
        "children": Table(
            pd.DataFrame(
                {
                    "id": [10, 11, 12],
                    "root": [1, 2, 1],
                    "value": [0.2, 0.8, 999.0],
                    "date": pd.to_datetime(["2000-01-01", "2000-01-01", "2040-01-01"]),
                }
            ),
            key="id",
            date="date",
        ),
    }
    edges = [Relationship("roots", "children", "id", "root")]
    examples = pd.DataFrame(
        {
            "root": [1, 2],
            "time": pd.to_datetime(["2001-01-01"] * 2),
            "y": [1000, -1000],
        },
        index=[7, 9],
    )
    return tables, edges, examples


def test_inputs_include_root_exclude_keys_future_rows_and_query_targets():
    tables, edges, examples = source_fixture()
    family = BoundedNeighborhoodInputs(context_depth=0, text_components=2)
    first = family.fit_transform(
        examples,
        tables=tables,
        relationships=edges,
        root_name="roots",
        entity_key="root",
        timestamp="time",
    )
    assert family.paths_["p0"] == ()
    assert len(first) == 2
    assert "secret" not in family.encoders_["p0"].vectorizer_.vocabulary_
    assert family.audit[1]["neighbors"] == 2
    for entry in family.audit:
        assert not {"id", "root", "date", "y"}.intersection(entry["payload_columns"])
    poisoned = examples.copy()
    poisoned["y"] = -1e20
    restored = pickle.loads(pickle.dumps(family))
    replay = restored.transform(poisoned, tables=tables, relationships=edges)
    for name in first:
        pd.testing.assert_frame_equal(first[name], replay[name])
