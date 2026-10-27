import pickle

import numpy as np
import pandas as pd
import pytest

from scripts.portable_offset_readout import PortableComparisonOffsetRanker, RECIPES


@pytest.mark.parametrize("recipe", RECIPES)
def test_portable_recipes_use_no_randomized_fit_and_keep_maturity_checks(recipe):
    x = pd.DataFrame({"state":np.arange(40), "second":np.tile([0., 1., 2., 3.], 10)})
    y, p = np.tile([0, 1], 20), np.linspace(.1, .7, 40)
    kwargs = dict(entities=np.arange(40), timestamps=np.repeat(pd.Timestamp("2020-01-01"), 40),
                  origin="2020-03-01", horizon="30d")
    model = PortableComparisonOffsetRanker(iterations=3, **RECIPES[recipe]).fit(x, y, p, **kwargs)
    np.testing.assert_array_equal(model.predict_values(x, p, strength=1.),
        pickle.loads(pickle.dumps(model)).predict_values(x, p, strength=1.))
    members = model.members if model.consensus else [model]
    for member in members:
        assert member.model.get_params()["bootstrap_type"] == "No"
        assert member.model.get_params()["random_strength"] == 0.
        assert member.audit["iterations"] == 3 and member.thread_count == 1
    kwargs["origin"] = "2020-01-02"
    with pytest.raises(ValueError, match="matured"):
        PortableComparisonOffsetRanker(iterations=3, **RECIPES[recipe]).fit(x, y, p, **kwargs)
