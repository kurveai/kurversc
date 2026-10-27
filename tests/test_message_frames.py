import pickle

import numpy as np
import pandas as pd
import pytest

from kurversc import LearnedMessageFamily, PooledMessageFamily
from kurversc.message_frames import freeze_message_family


def test_frame_encoder_cannot_advance_to_a_later_supervised_model():
    family = PooledMessageFamily()
    family.paths_ = (("events",),)
    family.models_ = {
        ("events",): [
            (pd.Timestamp("2001-01-01"), None, (), 0.2),
            (pd.Timestamp("2003-01-01"), None, (), 0.8),
        ]
    }
    frozen = freeze_message_family(family, "2003-01-01")
    frozen = pickle.loads(pickle.dumps(frozen))
    assert len(family.models_[("events",)]) == 2
    assert len(frozen.models_[("events",)]) == 1
    assert frozen.models_[("events",)][0][-1] == 0.2  # strict equality exclusion
    assert frozen.fitted_frame_cutoff_ == pd.Timestamp("2003-01-01")
    assert freeze_message_family(family, "2001-01-01").models_[("events",)] == []


def test_unfitted_frame_encoder_is_rejected():
    with pytest.raises(ValueError, match="Fit"):
        freeze_message_family(PooledMessageFamily(), "2003-01-01")


class ScalingScorer:
    def __init__(self, factor):
        self.factor = factor

    def predict(self, x):
        return x["signal"].to_numpy() * self.factor


def test_frozen_weights_apply_to_current_rows_without_advancing_the_encoder():
    family = LearnedMessageFamily()
    family.paths_ = (("events",),)
    family.timestamp_, family.task_ = "time", "regression"
    family.models_ = {
        ("events",): [
            (pd.Timestamp("2002-01-01"), ScalingScorer(1), ("signal",), 0.0),
            (pd.Timestamp("2004-01-01"), ScalingScorer(10), ("signal",), 0.0),
        ]
    }
    examples = pd.DataFrame({"time": pd.to_datetime(["2003-01-01", "2005-01-01"])})
    rows = pd.DataFrame({"__example_id": [0, 0, 1, 1, 1]})
    encoded = pd.DataFrame({"signal": [1.0, 2.0, 1.0, 2.0, 3.0]})
    frozen = pickle.loads(pickle.dumps(freeze_message_family(family, "2003-01-01")))
    fixed = frozen._transform_encoded_path(rows, encoded, examples, path=("events",))
    rolling = family._transform_encoded_path(rows, encoded, examples, path=("events",))
    np.testing.assert_allclose(fixed["score__mean"], [1.5, 2.0])
    np.testing.assert_allclose(rolling["score__mean"], [1.5, 20.0])
    np.testing.assert_array_equal(fixed.neighbor_count, [2, 3])
