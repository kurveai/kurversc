import numpy as np
import pandas as pd

from kurversc.modeling import (
    prepare_features,
    prepare_prediction_features,
    sample_training_rows,
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
