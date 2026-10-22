import pickle

import numpy as np
import pandas as pd
import pytest

from scripts.portable_bagged_offset import PortableBaggedOffsetRanker, PortableBaggedRecipeReadout


def test_portable_bag_ships_exactly_the_validation_declared_fixed_members():
    x = pd.DataFrame({"state":np.arange(40), "second":np.tile([0., 1., 2., 3.], 10)})
    y, p = np.tile([0, 1], 20), np.linspace(.1, .7, 40)
    model = PortableBaggedOffsetRanker(seed=1).fit(x, y, p, entities=np.arange(40),
        timestamps=np.repeat(pd.Timestamp("2020-01-01"), 40), origin="2020-03-01", horizon="30d")
    assert [m.seed for m in model.models] == [3, 4, 5]
    expected = np.mean([m.predict_values(x, p, strength=1.) for m in model.models], axis=0)
    np.testing.assert_array_equal(model.predict_values(x, p, strength=1.), expected)
    np.testing.assert_array_equal(model.predict_values(x, p, strength=1.),
        pickle.loads(pickle.dumps(model)).predict_values(x, p, strength=1.))
    assert all(m.model.get_params()["bootstrap_type"] == "No" for m in model.models)
    assert model.audit["chosen_seeds_or_weights_from_outcomes"] is False
    with pytest.raises(ValueError):
        model.predict_values(x, p, strength=.25)
    with pytest.raises(ValueError):
        PortableBaggedRecipeReadout(recipe="test-picked-seed")
