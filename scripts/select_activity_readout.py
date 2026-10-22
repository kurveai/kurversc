"""Freeze a generic optional readout from matched retrospective evidence.

This program never opens an official/test split. The incumbent, graph, support,
and deployment seed stay fixed. All supplied presets and blend weights share
one multiplicity-corrected promotion decision; unavailable/unstable recipes
cannot be silently promoted from a single favorable scout.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kurversc.temporal_policy import TemporalFitPolicy, select_temporal_policy
from scripts.benchmark_signal_challengers import dump, incumbent_evidence


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_locations(task_root):
    """Prefer colocated artifacts; legacy manifests may reference shared caches."""
    manifest = json.loads((task_root / "selection/manifest.json").read_text())
    cache = task_root / "cache"
    snapshots = task_root / "snapshots"
    if not (cache / "history.parquet").exists():
        cache = Path(manifest["cache"])
    if not snapshots.exists():
        snapshots = Path(manifest["phase_frozen_cache"])
    return cache, snapshots


def collect_evidence(task_root, experiments):
    metadata = json.loads((task_root / "selection/manifest.json").read_text())["metadata"]
    base_policy = TemporalFitPolicy(**json.loads(
        (task_root / "selection/results.json").read_text())["selected_policy"])
    evidence, recipes, identities = [], {}, []
    controls, cohort = {}, None
    snapshot_root = task_root / "snapshots"
    if not snapshot_root.exists():
        snapshot_root = Path(json.loads((task_root / "selection/manifest.json").read_text())["phase_frozen_cache"])
    for number, directory in enumerate(experiments):
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if not (directory / "results.json").exists():
            raise ValueError("A scout or incomplete experiment cannot enter the confirmed menu")
        if manifest["representation"] != "full":
            raise ValueError("Production readout promotion requires the full frozen graph")
        if manifest["selected_incumbent"] != asdict(base_policy):
            raise ValueError("Experiments use different incumbent recipes")
        if manifest["expert_support"] != asdict(TemporalFitPolicy("incumbent")):
            raise ValueError("Unexpected expert support; no silent support-budget change")
        if manifest["metadata"] != {k: v for k, v in metadata.items() if k != "feature_schema"}:
            raise ValueError("Experiments use a different frozen graph or task")
        if manifest["covariates"] != list(metadata["feature_schema"]["columns"]):
            raise ValueError("Candidate did not receive the incumbent feature schema")
        current = (tuple(manifest["folds"]), tuple(manifest["seeds"]))
        if len(set(current[0])) < 2 or len(set(current[1])) < 2 or 0 not in current[1]:
            raise ValueError("At least two origins, two seeds, and deployment seed zero required")
        if cohort is not None and current != cohort:
            raise ValueError("Presets need the exact same origin/seed coverage")
        cohort = current
        identity = dict(path=str(directory), manifest_sha256=digest(manifest_path),
                        results_sha256=digest(directory / "results.json"), vectors={})
        for fold in manifest["folds"]:
            fold_manifest = json.loads((snapshot_root / f"fold{fold}" / "manifest.json").read_text())
            origin = fold_manifest["origin"]
            boundary = pd.Timestamp(fold_manifest["snapshot_cutoff"])
            if (boundary > pd.Timestamp(origin) or
                pd.Timestamp(fold_manifest.get("latest_observed_label_complete_at", boundary)) > boundary):
                raise ValueError("Future state or labels reached the declared phase")
            for seed in manifest["seeds"]:
                control = incumbent_evidence(task_root / "raw-audit", fold, seed, base_policy)
                key = (origin, seed)
                if key not in controls:
                    controls[key] = control
                    evidence.append(dict(origin=origin, seed=seed, policy="incumbent", **control))
                else:
                    for field in ("truth", "groups", "prediction"):
                        np.testing.assert_array_equal(controls[key][field], control[field])
                for arm in manifest["arms"]:
                    vector_path = directory / f"fold{fold}-{arm}-seed{seed}.npz"
                    report_path = directory / f"fold{fold}-{arm}-seed{seed}.json"
                    identity["vectors"][vector_path.name] = digest(vector_path)
                    identity["vectors"][report_path.name] = digest(report_path)
                    with np.load(vector_path) as saved:
                        np.testing.assert_array_equal(control["truth"], saved["truth"])
                        np.testing.assert_array_equal(control["groups"], saved["groups"])
                        prediction = saved["prediction"].copy()
                    report = json.loads(report_path.read_text())
                    if (report["origin"], report["seed"], report["arm"]) != (origin, seed, arm):
                        raise ValueError("Prediction vector provenance mismatch")
                    if list(report["columns"][:len(manifest["covariates"])]) != manifest["covariates"]:
                        raise ValueError("Expert omitted/reordered frozen graph columns")
                    expected_support = json.loads((task_root / "raw-audit" /
                        f"fold{fold}-incumbent-seed{seed}.json").read_text())["audit"]
                    for field in ("initial_cutoffs", "observed_refit_cutoffs"):
                        if report["fit_audit"][field] != expected_support[field]:
                            raise ValueError("Expert used different support from validation incumbent")
                    for weight in (.25, .5, 1.0):
                        name = f"preset{number}_{arm}__blend_{weight:g}"
                        recipes[name] = dict(arm=arm, blend_weight=weight,
                            model_params=dict(iterations=manifest["iterations"], depth=manifest["depth"],
                                l2_leaf_reg=manifest["l2_leaf_reg"], auto_class_weights=manifest["class_weight"]),
                            expert_support=manifest["expert_support"],
                            covariates=manifest["covariates"], channels=manifest["channels"],
                            provider_sha256=manifest["provider_sha256"],
                            enum_provider_sha256=manifest["enum_provider_sha256"])
                        blended = (1 - weight) * control["prediction"] + weight * prediction
                        evidence.append(dict(origin=origin, seed=seed, policy=name,
                            truth=control["truth"], groups=control["groups"], prediction=blended))
        identities.append(identity)
    if not identities:
        raise ValueError("At least one confirmed preset required")
    return evidence, recipes, dict(metadata=metadata, selected_incumbent=asdict(base_policy),
                                  experiments=identities, deployment_seed=0)


def choose_readout(task_root, experiments, *, bootstrap_repeats=600):
    evidence, recipes, manifest = collect_evidence(task_root, experiments)
    decision = select_temporal_policy(evidence, task="classification", deployment_seed=0,
                                      bootstrap_repeats=bootstrap_repeats)
    return dict(decision=decision, manifest=manifest, menu=recipes,
        selected_recipe=recipes.get(decision["selected"]),
        scope="Validation-only optional readout; no benchmark-specific winners; unchanged incumbent fallback")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = choose_readout(args.task_root, args.experiments)
    result["manifest"]["selector_sha256"] = digest(__file__)
    if args.output.exists() and json.loads(args.output.read_text()) != result:
        raise ValueError("Refusing to overwrite a different immutable readout selection")
    dump(args.output, result)
    print(json.dumps(dict(selected=result["decision"]["selected"],
                          reason=result["decision"]["reason"])), flush=True)


if __name__ == "__main__":
    main()
