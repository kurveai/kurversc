import numpy as np
import pandas as pd

from kurversc import modeling
from kurversc.modeling import (
    DEFAULT_CATBOOST_TUNING_CONFIGS,
    prepare_features,
    prepare_prediction_features,
    tune_catboost,
)


def test_nested_relational_cells_are_stable_categorical_features() -> None:
    train = pd.DataFrame(
        {
            "nested": [np.array(["a", "b"]), np.array(["c"])],
            "target": [0, 1],
        }
    )
    validation = pd.DataFrame(
        {
            "nested": [np.array(["a", "b"]), np.array(["d"])],
            "target": [0, 1],
        }
    )

    train_x, _, validation_x, _, categorical = prepare_features(
        train,
        validation,
        target="target",
        excluded=set(),
    )
    prediction_x = prepare_prediction_features(
        validation,
        columns=["nested"],
        categorical_columns={"nested"},
        datetime_columns=set(),
    )

    assert categorical == [0]
    assert train_x["nested"].tolist() == ['["a","b"]', '["c"]']
    assert validation_x["nested"].tolist() == ['["a","b"]', '["d"]']
    assert prediction_x.equals(validation_x)


def test_catboost_tuning_selects_score_and_reuses_best_tree_count(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeModel:
        def __init__(self, tree_count: int) -> None:
            self.tree_count_ = tree_count

    def fake_fit(*args, task, model_params, **kwargs):
        del args, kwargs
        params = dict(model_params)
        calls.append(params)
        score = float(params["depth"]) / 10
        return FakeModel(int(params["iterations"]) // 2), "roc_auc", score, 1.5

    monkeypatch.setattr(modeling, "fit_catboost", fake_fit)
    selected = tune_catboost(
        pd.DataFrame({"x": [0, 1]}),
        pd.Series([0, 1]),
        pd.DataFrame({"x": [2, 3]}),
        pd.Series([0, 1]),
        task="classification",
        categorical=[],
        random_state=42,
        model_params=None,
        tuning_configs=(
            {"name": "shallow", "iterations": 20, "depth": 4},
            {
                "name": "deep",
                "iterations": 30,
                "depth": 7,
                "early_stopping_rounds": 5,
            },
        ),
    )

    _, metric, score, seconds, params, records = selected
    assert metric == "roc_auc"
    assert score == 0.7
    assert seconds == 3.0
    assert params == {"iterations": 15, "depth": 7}
    assert [record["name"] for record in records] == ["shallow", "deep"]
    assert [record["tree_count"] for record in records] == [10, 15]
    assert len(calls) == 2


def test_default_catboost_tuning_space_is_bounded() -> None:
    assert [config["name"] for config in DEFAULT_CATBOOST_TUNING_CONFIGS] == [
        "depth6_regularized",
        "depth7_regularized",
        "depth5_regularized",
        "depth6_fast",
    ]
    assert (
        max(config["iterations"] for config in DEFAULT_CATBOOST_TUNING_CONFIGS) == 2500
    )
