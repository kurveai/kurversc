import numpy as np
import pandas as pd

from kurversc.modeling import prepare_features, prepare_prediction_features


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
