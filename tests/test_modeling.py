import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kurversc.modeling import (
    _tabpfn_estimator,
    _tabicl_estimator,
    _tabfm_estimator,
    fit_final_tabicl,
    fit_tabicl,
    prepare_features,
    prepare_prediction_features,
    sample_training_rows,
)


def test_tabpfn_factory_selects_v3_5_checkpoint(monkeypatch) -> None:
    class FakeEstimator:
        @classmethod
        def create_default_for_version(cls, version, **params):
            return SimpleNamespace(version=version, params=params)

    versions = SimpleNamespace(V3="v3", V3_5="v3.5")
    monkeypatch.setitem(
        sys.modules,
        "tabpfn",
        SimpleNamespace(
            TabPFNClassifier=FakeEstimator,
            TabPFNRegressor=FakeEstimator,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tabpfn.constants",
        SimpleNamespace(ModelVersion=versions),
    )

    estimator = _tabpfn_estimator(
        task="classification",
        categorical=[1],
        random_state=7,
        model_params={"n_estimators": 4},
        version_attribute="V3_5",
    )

    assert estimator.version == "v3.5"
    assert estimator.params["n_estimators"] == 4
    assert estimator.params["categorical_features_indices"] == [1]


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


def test_estimator_training_sample_is_capped_stratified_and_deterministic() -> None:
    features = pd.DataFrame({"value": range(100)})
    labels = pd.Series([0] * 80 + [1] * 20)

    first_x, first_y = sample_training_rows(
        features,
        labels,
        limit=20,
        task="classification",
        random_state=42,
    )
    second_x, second_y = sample_training_rows(
        features,
        labels,
        limit=20,
        task="classification",
        random_state=42,
    )

    assert len(first_x) == 20
    assert first_y.value_counts().to_dict() == {0: 16, 1: 4}
    assert first_x.equals(second_x)
    assert first_y.equals(second_y)


def test_tabicl_factory_selects_task_estimator_and_allows_overrides(
    monkeypatch,
) -> None:
    class FakeClassifier:
        def __init__(self, **params) -> None:
            self.params = params

    class FakeRegressor:
        def __init__(self, **params) -> None:
            self.params = params

    monkeypatch.setitem(
        sys.modules,
        "tabicl",
        SimpleNamespace(
            TabICLClassifier=FakeClassifier,
            TabICLRegressor=FakeRegressor,
        ),
    )

    classifier = _tabicl_estimator(
        task="classification",
        random_state=7,
        model_params={"n_estimators": 2, "n_jobs": 3},
    )
    regressor = _tabicl_estimator(
        task="regression",
        random_state=11,
        model_params=None,
    )

    assert isinstance(classifier, FakeClassifier)
    assert classifier.params == {"n_jobs": 3, "random_state": 7, "n_estimators": 2}
    assert isinstance(regressor, FakeRegressor)
    assert regressor.params == {"n_jobs": 8, "random_state": 11}


def test_tabfm_factory_loads_frozen_weights_and_allows_overrides(monkeypatch) -> None:
    class FakeClassifier:
        def __init__(self, **params) -> None:
            self.params = params

    class FakeRegressor:
        def __init__(self, **params) -> None:
            self.params = params

    class FakeWeights:
        @staticmethod
        def load(**params):
            return params

    monkeypatch.setitem(
        sys.modules,
        "tabfm",
        SimpleNamespace(
            TabFMClassifier=FakeClassifier,
            TabFMRegressor=FakeRegressor,
            tabfm_v1_0_0_pytorch=FakeWeights,
        ),
    )

    estimator = _tabfm_estimator(
        task="classification",
        random_state=7,
        model_params={"max_num_rows": 64},
    )

    assert estimator.params["model"] == {
        "model_type": "classification",
        "device": "cpu",
        "dtype": None,
    }
    assert estimator.params["max_num_rows"] == 64
    assert estimator.params["use_amp"] is False


def test_tabicl_regression_adapter_fits_and_scores(monkeypatch) -> None:
    class FakeRegressor:
        def __init__(self, **params) -> None:
            self.params = params
            self.fit_y = None

        def fit(self, features, labels) -> None:
            self.fit_y = labels

        def predict(self, features) -> np.ndarray:
            return np.array([0.25, 1.5])

    monkeypatch.setitem(
        sys.modules,
        "tabicl",
        SimpleNamespace(TabICLClassifier=object, TabICLRegressor=FakeRegressor),
    )
    train_x = pd.DataFrame({"category": pd.Series(["a", "b"], dtype="string")})
    validation_x = pd.DataFrame({"category": pd.Series(["a", "c"], dtype="string")})

    model, metric, score, seconds = fit_tabicl(
        train_x,
        pd.Series([0, 2]),
        validation_x,
        pd.Series([0.0, 1.0]),
        task="regression",
        categorical=[0],
        random_state=5,
        model_params={"n_estimators": 1},
    )

    assert model.fit_y.dtype == "float64"
    assert metric == "mae"
    assert score == 0.375
    assert seconds >= 0


def test_final_tabicl_classifier_encodes_and_returns_target_classes(
    monkeypatch,
) -> None:
    class FakeClassifier:
        def __init__(self, **params) -> None:
            self.fit_y = None

        def fit(self, features, labels) -> None:
            self.fit_y = labels

    monkeypatch.setitem(
        sys.modules,
        "tabicl",
        SimpleNamespace(TabICLClassifier=FakeClassifier, TabICLRegressor=object),
    )

    model, seconds, classes = fit_final_tabicl(
        pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}),
        pd.Series(["yes", "no", "yes", "no"]),
        task="classification",
        categorical=[],
        random_state=3,
        model_params=None,
    )

    assert model.fit_y.tolist() == [1, 0, 1, 0]
    assert classes == ("no", "yes")
    assert seconds >= 0
