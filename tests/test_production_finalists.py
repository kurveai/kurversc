from dataclasses import asdict
import json
import pickle
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from kurversc.search import GraphConfig, Trial
from scripts.benchmark_production_finalists import choose_pair, compare_cases, read_cases
from scripts.run_production_finalist_cloud import finalized_graph_row, graph_command


def trial(width, config=None, metric="mae"):
    return Trial(config=config or GraphConfig(depth=1, feature_families=("base",)),
        metric=metric, validation_score=.1, objective_score=-.1, feature_count=width,
        train_rows=100, validation_rows=20, feature_seconds=0, model_seconds=0,
        stage="rerank", cutoff_labels=("a", "b", "c"), cutoff_scores=(.12, .11, .1),
        execution_plan={"frozen": True})


def cases(scores=(.12, .11, .1)):
    return {(fold, seed):dict(truth=np.zeros(50), prediction=np.full(50, score),
        groups=np.arange(50), origin=f"2020-0{fold+1}-01")
        for fold, score in enumerate(scores) for seed in range(3)}


def pair():
    return trial(1200, GraphConfig(depth=1, feature_families=("base", "temporal"))), trial(450)


def test_existing_selector_selects_unanimous_small_production_tie():
    result = compare_cases(*pair(), cases(), cases((.1198, .1101, .1001)))
    assert result["selected"] == "challenger"
    assert result["seed_decisions"] == {"0":"challenger", "1":"challenger", "2":"challenger"}
    assert result["all_seed_fold_practical_safety"]
    assert not result["official_targets_used_for_selection"]


def test_any_seed_practical_loss_preserves_incumbent():
    alt = cases((.1198, .1101, .1001))
    alt[2, 2]["prediction"][:] = .102
    result = compare_cases(*pair(), cases(), alt)
    assert result["selected"] == "incumbent"
    assert not result["all_seed_fold_practical_safety"]


def test_older_practical_loss_is_not_ignored_for_newest_improvement():
    result = compare_cases(*pair(), cases(), cases((.123, .105, .09)))
    assert result["selected"] == "incumbent"


def test_no_graph_replacement_without_drastic_complexity_reduction():
    current, alternate = pair()
    alternate.feature_count = 700
    assert compare_cases(current, alternate, cases(), cases())["selected"] == "incumbent"


@pytest.mark.parametrize("field", ["truth", "groups", "origin"])
def test_rejects_unmatched_prediction_cohorts(field):
    alt = cases()
    if field == "origin":
        alt[0, 0][field] = "2012-01-01"
    else:
        alt[0, 0][field][0] = 999
    with pytest.raises((ValueError, AssertionError)):
        compare_cases(*pair(), cases(), alt)


def test_rejects_missing_seed_or_nonfinite_prediction():
    alt = cases()
    del alt[2, 2]
    with pytest.raises(ValueError, match="Complete"):
        compare_cases(*pair(), cases(), alt)
    alt = cases()
    alt[0, 0]["prediction"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compare_cases(*pair(), cases(), alt)


def test_only_schema_generic_current_run_finalists_enter_pair(tmp_path):
    current, alternate = pair()
    root = tmp_path / "checkpoints/profile/dataset_task/run/trials"
    root.mkdir(parents=True)
    for name, value in [("current", current), ("alternate", alternate)]:
        (root / f"{name}.pkl").write_bytes(pickle.dumps({"trial":value}))
    row = tmp_path / "search.json"
    row.write_text(json.dumps(dict(metric="mae", config=json.dumps(asdict(current.config)))))
    selected = choose_pair(tmp_path, row, "dataset", "task")
    assert [value.feature_count for _, value in selected] == [1200, 450]
    row.write_text(json.dumps(dict(metric="roc_auc", config=json.dumps(asdict(current.config)))))
    assert choose_pair(tmp_path, row, "dataset", "task") is None


def test_changed_graph_never_inherits_its_predecessors_official_score():
    current, alternate = pair()
    original = dict(config=json.dumps(asdict(current.config)), test_score=.012,
        improvement=.9, beats_published_kurversc=True, plan_fingerprint="old",
        inner_validation_score=.2, inner_full_validation_score=.1, stage_counts="unchanged")
    row = finalized_graph_row(original, dict(selected="challenger", config=asdict(alternate.config)),
        validation_report="validation/results.json")
    assert row["test_score"] is None and row["plan_fingerprint"] is None
    assert row["graph_official_score_pending"]
    assert row["stage_counts"] == "unchanged" and original["test_score"] == .012
    fallback = finalized_graph_row(original, dict(selected="incumbent"), validation_report="v")
    assert fallback["test_score"] == original["test_score"]


def test_routes_only_graph_stage_without_rewriting_other_stage_arguments():
    original = ["python", "-u", "-m", "scripts.run_kurversc_selected_relarena", "--dataset", "d", "--seed", "0"]
    routed = graph_command(original)
    assert routed == ["python", "-u", "-m", "scripts.run_production_finalist_cloud", "--mode", "graph", "--",
        "--dataset", "d", "--seed", "0"]
    assert original[3] == "scripts.run_kurversc_selected_relarena"
    for command in (["python", "script.py"], ["python", "-m", "scripts.benchmark_recovered_readouts"]):
        assert graph_command(command) == command


def test_refuses_wrong_lifecycle_before_reading_any_vectors(tmp_path):
    root = tmp_path / "raw-audit"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(dict(folds=[{}, {}, {}], seeds=[0,1,2], observed_refit_frames=3)))
    with pytest.raises(ValueError, match="single-refit"):
        read_cases(tmp_path)


def test_cloud_profile_preserves_previous_recipe_and_graph_parser_arguments():
    program = '''
import sys
from scripts import run_staged_complete_cloud as staged
from scripts.run_recovered_staged_cloud import READOUT_CONTRACT
from scripts.run_count_challenger_cloud import COUNT_CONTRACT
from scripts import run_production_finalist_cloud as fidelity
from scripts.benchmark_production_finalists import CONTRACT
original = dict(staged.CONTRACT)
fidelity.install_recipe()
assert staged.CONTRACT == {**original, "recovered_readouts": READOUT_CONTRACT,
    "conditional_count_challenger": COUNT_CONTRACT, "production_finalist_check": CONTRACT}
assert staged.PROFILE == fidelity.PROFILE
assert staged.CONTRACT["initial_production_training_frames"] == 1
sys.argv = ["run", "--mode", "graph", "--", "--dataset", "generic", "--task", "unknown", "--output", "some.csv"]
seen = []
fidelity.graph_stage = lambda args: seen.extend(args)
fidelity.main()
assert seen == ["--dataset", "generic", "--task", "unknown", "--output", "some.csv"]
print("profile and parser verified")
'''
    completed = subprocess.run([sys.executable, "-c", program], cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "profile and parser verified" in completed.stdout
