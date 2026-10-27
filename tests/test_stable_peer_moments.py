import numpy as np
import pandas as pd
import pytest

from scripts.stable_peer_moments import stable_peer_moments


def test_unresolved_constant_population_roundoff_is_zero_not_a_split_signal():
    frame = pd.DataFrame({"weighted_p0__weighted_mean": [1., 2., 0., 1.],
                          "weighted_p0__weighted_std": [1e-8, 2e-8, 0., .1]})
    result, audit = stable_peer_moments(frame)
    np.testing.assert_array_equal(result.weighted_p0__weighted_std, np.array([0., 0., 0., .1], dtype="float32"))
    assert audit[0]["cancellation_noise_rows"] == 2


def test_bound_is_scale_equivariant_and_does_not_flatten_real_small_variance():
    mean, std = np.array([1., 2.]), np.array([1e-8, 1e-5])
    def transformed(mean, std):
        frame = pd.DataFrame({"weighted_p0__weighted_mean": mean, "weighted_p0__weighted_std": std})
        result, audit = stable_peer_moments(frame)
        return result.weighted_p0__weighted_std.to_numpy(), audit
    expected, changed = transformed(mean, std)
    for factor in (1e-6, 1e6):
        actual, mask = transformed(mean*factor, std*factor)
        assert mask == changed
        np.testing.assert_allclose(actual, expected*factor, rtol=1e-6, atol=0)
    assert expected[1] == np.float32(std[1])


def test_missing_values_and_exact_zeros_remain_missing_or_zero():
    frame = pd.DataFrame({"weighted_p0__weighted_mean": [np.nan, 0., 1.],
                          "weighted_p0__weighted_std": [np.nan, 0., np.nan]})
    result, audit = stable_peer_moments(frame)
    np.testing.assert_array_equal(result.weighted_p0__weighted_std, [np.nan, 0., np.nan])
    assert audit[0]["cancellation_noise_rows"] == 0


def test_research_normalizer_preserves_source_and_other_columns():
    frame = pd.DataFrame({"weighted_p0__weighted_mean": [1., 1.],
        "weighted_p0__weighted_std": [1e-8, .25], "ordinary_std": [1e-8, .25]})
    original = frame.copy(deep=True)
    actual, audit = stable_peer_moments(frame)
    pd.testing.assert_frame_equal(frame, original)
    assert actual.weighted_p0__weighted_std.tolist() == [0., .25]
    np.testing.assert_array_equal(actual.ordinary_std, frame.ordinary_std.astype("float32"))
    assert audit == [dict(column="weighted_p0__weighted_std", cancellation_noise_rows=1)]


def test_unpaired_or_invalid_moments_fail_closed():
    with pytest.raises(ValueError):
        stable_peer_moments(pd.DataFrame({"weighted_p2__weighted_std": [1.]}))
