import numpy as np
import pytest

from scripts import benchmark_sequence_recovery_20261020 as runner


def test_fixed_composition_retains_both_demonstrated_sequence_components():
    values = runner.candidate_vectors([.1, .4], [.2, .3], [.6, .5])
    np.testing.assert_array_equal(values[runner.RECIPES[0]], .875*np.array([.1,.4])+.125*np.array([.2,.3]))
    np.testing.assert_array_equal(values[runner.RECIPES[1]],
        .765625*np.array([.1,.4])+.125*np.array([.2,.3])+.109375*np.array([.6,.5]))
    same = runner.candidate_vectors([.25,.5], [.25,.5], [.25,.5])
    for value in same.values():
        np.testing.assert_array_equal(value, [.25,.5])


@pytest.mark.parametrize("bad", [[np.nan], [1.,2.], []])
def test_invalid_components_cannot_be_admitted(bad):
    with pytest.raises(ValueError):
        runner.candidate_vectors([.2], bad, [.3])


@pytest.mark.parametrize("recent,temporal,older,expected", [
    (.01,.01,-.001,runner.RECIPES[0]),
    (-.01,.01,-.001,"incumbent"),
    (.01,-.01,-.001,"incumbent"),
    (.01,.01,-.02,"incumbent"),
])
def test_original_confidence_and_complete_seed_safety_are_not_relaxed(monkeypatch, recent, temporal, older, expected):
    def select(evidence, **kwargs):
        assert kwargs["minimum_origins"] == 2 and kwargs["minimum_seeds"] == 3
        assert kwargs["alpha"] == .05*len(runner.RECIPES)/1024
        assert kwargs["deployment_seed"] == 0 and kwargs["bootstrap_repeats"] == 1600
        before = dict(seed_gains=[older]*3, practical_loss_cap=.005)
        latest = dict(seed_gains=[.01]*3, gain=.01, recent_lower_bound=recent,
            temporal_gain_lower_bound=temporal)
        return dict(candidates={runner.RECIPES[0]:dict(reason="eligible",folds=[before,latest])})
    monkeypatch.setattr(runner,"frozen_auc_select_temporal_policy",select)
    assert runner.choose([], origins=2)[0] == expected
