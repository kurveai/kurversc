from dataclasses import asdict
import json

import numpy as np
import pytest

from kurversc.temporal_policy import TemporalFitPolicy
from scripts.benchmark_signal_challengers import dump
from scripts.select_activity_readout import choose_readout, collect_evidence


def panel(tmp_path, candidate="strong"):
    root, experiment = tmp_path / "task", tmp_path / "preset"
    metadata = dict(metric="roc_auc", checkpoint_sha256="frozen", fold="current",
        atomic_date_replay=True, feature_schema=dict(columns=["arbitrary_numeric"],
            categorical_columns=[], datetime_columns=[]))
    policy = asdict(TemporalFitPolicy("incumbent"))
    dump(root / "selection/manifest.json", dict(metadata=metadata))
    dump(root / "selection/results.json", dict(selected_policy=policy))
    manifest = dict(metadata={k:v for k,v in metadata.items() if k != "feature_schema"},
        representation="full", selected_incumbent=policy, expert_support=policy,
        covariates=metadata["feature_schema"]["columns"], folds=[0,1], seeds=[0,1],
        arms=["frozen_control"], iterations=300, depth=6, l2_leaf_reg=10,
        class_weight="Balanced", channels=[], provider_sha256="provider",
        enum_provider_sha256="enum")
    dump(experiment / "manifest.json", manifest)
    dump(experiment / "results.json", dict(completed=True))
    truth = np.tile([0,1], 100)
    control = np.where(truth, .6, .4)
    control[:100] = .5
    for fold in (0,1):
        origin = f"2020-0{fold+3}-01 00:00:00"
        dump(root / "snapshots" / f"fold{fold}" / "manifest.json",
            dict(origin=origin, snapshot_cutoff=origin))
        for seed in (0,1):
            audit = dict(initial_cutoffs=["2019-01-01 00:00:00"],
                         observed_refit_cutoffs=["2019-02-01 00:00:00"])
            raw = root / "raw-audit"
            raw.mkdir(exist_ok=True)
            arrays = dict(truth=truth, groups=np.arange(len(truth)), prediction=control)
            np.savez(raw / f"fold{fold}-incumbent-seed{seed}.npz", **arrays)
            dump(raw / f"fold{fold}-incumbent-seed{seed}.json", dict(audit=audit))
            prediction = np.where(truth, .9, .1)
            if candidate == "weak" or (candidate == "older_regression" and fold == 0):
                prediction = 1 - prediction
            np.savez(experiment / f"fold{fold}-frozen_control-seed{seed}.npz",
                **{**arrays, "prediction": prediction})
            dump(experiment / f"fold{fold}-frozen_control-seed{seed}.json",
                dict(origin=origin, seed=seed, arm="frozen_control", fit_audit=audit,
                     columns=manifest["covariates"]))
    return root, experiment


def test_generic_confirmed_menu_can_promote_without_task_names(tmp_path):
    root, experiment = panel(tmp_path)
    evidence, recipes, manifest = collect_evidence(root, [experiment])
    assert len(evidence) == 16 and len(recipes) == 3
    assert manifest["deployment_seed"] == 0
    result = choose_readout(root, [experiment], bootstrap_repeats=30)
    assert result["selected_recipe"]["model_params"]["auto_class_weights"] == "Balanced"
    assert result["decision"]["reason"] == "paired_recent_gain_with_temporal_veto"


@pytest.mark.parametrize("candidate", ["weak", "older_regression"])
def test_validation_fallback_keeps_exact_incumbent_on_bad_candidates(tmp_path, candidate):
    root, experiment = panel(tmp_path, candidate)
    result = choose_readout(root, [experiment], bootstrap_repeats=30)
    assert result["selected_recipe"] is None
    assert result["decision"]["selected"] == "incumbent"


def test_single_fold_scout_cannot_be_silently_promoted(tmp_path):
    root, experiment = panel(tmp_path)
    path = experiment / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["folds"] = [1]
    dump(path, manifest)
    with pytest.raises(ValueError, match="two origins"):
        collect_evidence(root, [experiment])


def test_refit_support_or_truth_order_mismatch_is_not_accepted(tmp_path):
    root, experiment = panel(tmp_path)
    path = experiment / "fold0-frozen_control-seed0.json"
    report = json.loads(path.read_text())
    report["fit_audit"]["observed_refit_cutoffs"] = []
    dump(path, report)
    with pytest.raises(ValueError, match="different support"):
        collect_evidence(root, [experiment])


def test_all_presets_need_matched_fold_seed_coverage(tmp_path):
    root, experiment = panel(tmp_path)
    import shutil
    other = tmp_path / "second"
    shutil.copytree(experiment, other)
    path = other / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["seeds"] = [0,2]
    dump(path, manifest)
    with pytest.raises(ValueError, match="exact same"):
        collect_evidence(root, [experiment, other])
