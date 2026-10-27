import numpy as np
import pytest

from scripts.qualify_bounded_view_consensus_20261020 import compose, RECIPES


def test_fixed_consensus_preserves_exact_component_fallback_and_shapes():
    native = np.array([.2, .4, .6])
    heads = np.array([native, native, native])
    heads[:, 1] = [.5, .6, .7]
    result = compose(native, heads)
    for p in result.values():
        np.testing.assert_array_equal(p[[0, 2]], native[[0, 2]])
    assert result[RECIPES[-1]][1] == pytest.approx(.6)
    with pytest.raises(ValueError, match="three"):
        compose(native, heads[:2])
