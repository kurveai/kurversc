"""Classification exercises the same S2 candidate screen and safety vetoes."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kurversc.search import GraphConfig, Trial
from scripts.benchmark_production_finalists import compare_cases as compare_three
from scripts.benchmark_production_finalists_v2 import compare_cases, CONTRACT
from scripts.generic_finalist_helpers import finalist_case_score, finalist_loss, small_finalist


def trial(width, *, metric="roc_auc", scores=(.8, .8, .8)):
    return Trial(config=GraphConfig(depth=1, feature_families=("base", "temporal") if width > 500 else ("base",)),
        metric=metric, validation_score=scores[-1], objective_score=0., feature_count=width,
        train_rows=1000, validation_rows=200, feature_seconds=0., model_seconds=0.,
        stage="rerank", cutoff_labels=("2020-01-01", "2020-02-01", "2020-03-01"),
        cutoff_scores=scores, execution_plan={"frozen": True})


def cases(origins=3):
    truth = np.r_[np.ones(100), np.zeros(100)]
    prediction = np.r_[np.full(80, .8), np.full(20, .05), np.linspace(.2, .6, 100)]
    return {(fold, seed): dict(truth=truth.copy(), prediction=prediction.copy(), groups=np.arange(200),
        origin=f"2020-0{fold+1}-01", forecast_cutoffs=np.asarray([f"2020-0{fold+1}-01"] * 200))
        for fold in range(origins) for seed in range(3)}


@pytest.mark.parametrize("origins", [2, 3])
def test_classification_can_select_a_smaller_graph_at_production_fidelity(origins):
    large, small = trial(1200), trial(450)
    result = compare_cases(large, small, cases(origins), cases(origins))
    assert result["selected"] == "challenger"
    assert result["seed_decisions"] == {"0": "challenger", "1": "challenger", "2": "challenger"}
    assert result["all_seed_fold_practical_safety"]
    assert result["evidence"]["challenger_seed0"]["scores"] == [.8] * origins


def test_legacy_three_origin_comparator_uses_the_same_classification_rule():
    large, small = trial(1200), trial(450)
    first, second = cases(), cases()
    assert compare_three(large, small, first, second)["seed_decisions"] == compare_cases(large, small, first, second)["seed_decisions"]


def test_one_bad_classification_seed_preserves_the_incumbent():
    candidate = cases()
    candidate[2, 2]["prediction"][0] = .4  # Lose 0.005 AUC: >1% of 0.2 ranking error.
    result = compare_cases(trial(1200), trial(450), cases(), candidate)
    assert result["selected"] == "incumbent"
    assert not result["all_seed_fold_practical_safety"]


def test_an_older_classification_loss_cannot_be_hidden_by_recent_improvement():
    candidate = cases()
    for seed in range(3):
        candidate[0, seed]["prediction"][0] = .4
        candidate[2, seed]["prediction"][99] = .8
    result = compare_cases(trial(1200), trial(450), cases(), candidate)
    assert result["selected"] == "incumbent"
    assert not result["all_seed_fold_practical_safety"]


def test_a_small_classification_loss_has_the_same_relative_sensitivity():
    candidate = cases()
    for row in candidate.values():
        row["prediction"][0] = .58  # Lose only0.0005 AUC, within the shared1% error cap.
    result = compare_cases(trial(1200), trial(450), cases(), candidate)
    assert result["selected"] == "challenger"
    assert result["all_seed_fold_practical_safety"]
    for baseline, changed in [(2., 2.01), (.2, .201), (.01, .01005)]:
        regression_ratio = finalist_loss("mae", changed) / finalist_loss("mae", baseline)
        # The same loss ratio applies to every valid ranking-error scale.
        if baseline <= 1 and changed <= 1:
            classification_ratio = finalist_loss("roc_auc", 1-changed) / finalist_loss("roc_auc", 1-baseline)
            assert classification_ratio == pytest.approx(regression_ratio)


def test_recent_screen_uses_ranking_error_and_ignores_other_metrics():
    large = trial(1200, scores=(.98, .98, .98))
    close = trial(450, scores=(.98, .98, .9799))
    bad = replace(close, config=GraphConfig(depth=2), feature_count=300, cutoff_scores=(.98, .98, .979))
    wrong_metric = replace(close, config=GraphConfig(depth=3), metric="mae", feature_count=200, cutoff_scores=(0., 0., 0.))
    rows = [(Path("large"), large), (Path("close"), close), (Path("bad"), bad), (Path("wrong"), wrong_metric)]
    assert small_finalist(rows, large.config)[0] == Path("close")
    assert small_finalist(list(reversed(rows)), large.config)[0] == Path("close")
    # An absolute0.001 AUC concession would be5% more ranking error here.
    assert small_finalist([(Path("large"), large), (Path("bad"), bad)], large.config) is None


def test_native_auc_uncertainty_and_non_numeric_binary_labels_are_supported():
    row = cases()[0, 0]
    labels = np.where(row["truth"] == 1, "positive", "negative")
    score, error = finalist_case_score("roc_auc", labels, row["prediction"])
    assert score == .8
    assert 0 < error < .1
    assert set(CONTRACT["metrics"]) == {"mae", "roc_auc"}
    assert CONTRACT["comparison_loss"] == {"mae": "mae", "roc_auc": "1 - roc_auc"}


def test_classification_checks_require_both_classes_and_aligned_forecasts():
    row = cases()[0, 0]
    with pytest.raises(ValueError, match="Both binary classes"):
        finalist_case_score("roc_auc", np.zeros(200), row["prediction"])
    with pytest.raises(ValueError, match="finite"):
        finalist_case_score("roc_auc", row["truth"], np.full(200, np.nan))
    candidate = cases()
    candidate[0, 0]["forecast_cutoffs"][0] = "1990-01-01"
    with pytest.raises(AssertionError):
        compare_cases(trial(1200), trial(450), cases(), candidate)
