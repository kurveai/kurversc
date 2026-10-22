import json
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from scripts.frozen_offset_ranker import (
    FrozenOffsetRanker, cohort_pairs, probability_logits, replace_strong_half,
)


def observations():
    x = pd.DataFrame({"state": np.tile(np.arange(20), 2), "missing": [np.nan] * 40,
                      "constant": np.zeros(40)})
    y = np.tile(np.r_[np.zeros(10), np.ones(10)], 2)
    return x, y, np.repeat(pd.to_datetime(["2020-01-01", "2020-02-01"]), 20)


def test_cohort_pairs_are_bounded_reproducible_and_never_cross_time():
    _, y, times = observations()
    pairs = cohort_pairs(y, times, maximum=17, seed=2)
    assert pairs.shape == (17, 2)
    assert (y[pairs[:, 0]] == 1).all() and (y[pairs[:, 1]] == 0).all()
    np.testing.assert_array_equal(times[pairs[:, 0]], times[pairs[:, 1]])
    np.testing.assert_array_equal(pairs, cohort_pairs(y, times, maximum=17, seed=2))
    assert len(np.unique(pairs, axis=0)) == 17


def test_future_labels_and_structural_predictors_are_rejected():
    x, y, times = observations()
    args = dict(entities=np.arange(40), timestamps=times, origin="2020-02-02", horizon="30d")
    with pytest.raises(ValueError, match="matured"):
        FrozenOffsetRanker(iterations=2, thread_count=1).fit(x, y, np.full(40, .4), **args)
    args["origin"] = "2020-04-01"
    x["__event_entity"] = np.arange(40)
    with pytest.raises(ValueError, match="structural"):
        FrozenOffsetRanker(iterations=2, thread_count=1).fit(x, y, np.full(40, .4), **args)


def test_pool_baseline_is_not_added_twice_and_model_serializes():
    from catboost import Pool

    x, y, times = observations()
    base = np.linspace(.2, .7, 40)
    model = FrozenOffsetRanker(iterations=3, thread_count=1).fit(x, y, base,
        entities=np.arange(40), timestamps=times, origin="2020-04-01", horizon="30d")
    features = x[model.columns].to_numpy(dtype="float32")
    bare = model.model.predict(features)
    with_offset = model.model.predict(Pool(features, baseline=probability_logits(base)))
    np.testing.assert_allclose(with_offset, bare + probability_logits(base), rtol=0, atol=1e-7)
    expected = expit(probability_logits(base) + np.clip(bare, -1, 1))
    np.testing.assert_allclose(model.predict_values(x, base, strength=1.), expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(model.predict_values(x, base, strength=.25),
        pickle.loads(pickle.dumps(model)).predict_values(x, base, strength=.25))
    assert model.columns == ["state"]
    assert model.audit["pairs"] == 200


def test_correction_preserves_legacy_half_and_exact_unchanged_rows():
    legacy = np.array([.1, .3, .8])
    head = np.array([.2, .4, .7])
    control = .5 * (legacy + head)
    corrected = head.copy()
    corrected[1] += .01
    result = replace_strong_half(control, legacy, corrected)
    np.testing.assert_array_equal(result[[0, 2]], control[[0, 2]])
    np.testing.assert_allclose(result, .5 * (legacy + corrected), rtol=0, atol=1e-15)


@pytest.mark.parametrize("invalid", [[np.nan], [-.1], [1.1], [[.5]]])
def test_invalid_baseline_probabilities_fail_closed(invalid):
    with pytest.raises(ValueError):
        probability_logits(invalid)


def test_oof_source_uses_completed_result_format_and_rejects_missing_cases(tmp_path):
    from scripts.benchmark_engagement_offset_ranker import completed_oof_source

    report = dict(complete_grid=True, official_test_accessed=False,
        records=[dict(fold=fold, seed=seed) for fold in range(3) for seed in range(3)])
    path = tmp_path / "results.json"
    path.write_text(json.dumps(report))
    assert len(completed_oof_source(tmp_path)) == 64
    report["records"][-1] = report["records"][0]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Complete unique"):
        completed_oof_source(tmp_path)


def test_correction_input_excludes_keys_and_never_takes_validation_targets():
    from scripts.benchmark_engagement_offset_ranker import correction_frame

    state = pd.DataFrame({"observed_count": [0, 1], "label_free_category": ["a", "b"]})
    result = correction_frame(state, [.2, .3], [.4, .5])
    assert list(result) == ["observed_count", "offset_base_logit", "offset_text_logit", "offset_logit_difference"]
    state["__event_entity"] = [1, 2]
    with pytest.raises(ValueError, match="Routing"):
        correction_frame(state, [.2, .3], [.4, .5])
