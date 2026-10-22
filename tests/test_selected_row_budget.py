from dataclasses import asdict

import numpy as np
import pytest

from kurversc.temporal_policy import TemporalFitPolicy
from scripts.benchmark_selected_row_budget import primitive_policy, replacement_prediction


def test_changing_pooled_component_preserves_original_strong_blend():
    root, old, new = np.array([.2, .8]), np.array([.4, .6]), np.array([.5, .9])
    strong = .875 * root + .125 * old
    np.testing.assert_allclose(replacement_prediction(strong, old, new, .125, 1),
                               .875 * root + .125 * new)
    np.testing.assert_allclose(replacement_prediction(strong, old, old, .125, .5), strong)


def test_policy_is_recovered_without_schema_or_benchmark_specific_names():
    primitive = TemporalFitPolicy("generic", mode="pooled")
    selected = {**asdict(primitive), "name": "generic__blend_0.125", "blend_weight": .125}
    policy, base = primitive_policy(selected, [asdict(primitive)])
    assert policy.blend_weight == .125 and base == primitive


def test_unmatched_or_unpooled_policy_is_rejected():
    with pytest.raises(ValueError):
        primitive_policy(asdict(TemporalFitPolicy("incumbent")), [])


@pytest.mark.parametrize("values", [([.1], [.1, .2], [.3]), ([np.nan], [.1], [.3])])
def test_wrong_cohorts_and_nonfinite_predictions_fail_closed(values):
    with pytest.raises(ValueError):
        replacement_prediction(*values, .125, 1)
