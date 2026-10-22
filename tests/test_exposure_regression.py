import pickle

import numpy as np
import pandas as pd
import pytest

from kurversc.exposure_regression import ExposureScaledRegressor


def test_weighted_rate_mae_is_original_mae_and_scale_choice_is_training_only():
    scale = np.geomspace(1, 1000, 64)
    x = pd.DataFrame({"amount": scale, "type": np.arange(64) % 3})
    exposure = x.rename(columns={"amount": "anonymous_size"})
    y = scale * (0.3 + 0.01 * x.type)
    model = ExposureScaledRegressor(iterations=4, depth=2, thread_count=1).fit(
        x, y, exposure
    )
    assert model.column_ == "anonymous_size"
    e = model._scale(exposure)
    rate_prediction = model.model_.predict(x)
    np.testing.assert_allclose(
        np.mean(abs(y - e * rate_prediction)),
        np.mean(e * abs(y / e - rate_prediction)),
    )
    prediction = model.predict(x, exposure)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(prediction, restored.predict(x, exposure))
    np.testing.assert_array_equal(
        prediction[8:12], restored.predict(x.iloc[8:12], exposure.iloc[8:12])
    )
    with pytest.raises(ValueError, match="indexes"):
        model.predict(x, exposure.iloc[::-1])


def test_no_valid_exposure_has_plain_mae_fallback():
    x = pd.DataFrame({"bounded": np.arange(24) % 3, "signed": np.arange(24) - 12})
    model = ExposureScaledRegressor(iterations=3, depth=2, thread_count=1).fit(
        x, np.arange(24), x
    )
    assert model.fit_audit_["fallback"]
    np.testing.assert_array_equal(model.predict(x, x), model.model_.predict(x))
