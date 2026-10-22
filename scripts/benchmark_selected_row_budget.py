"""Validation-only row-budget challenger against the selected strong policy.

Reuse the audited graph, support, learner and blend. Independently replay the
original pooled learner at its original ceiling before increasing that ceiling.
No official split or test scores are opened; no production defaults change.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from kurversc.temporal_policy import TemporalFitPolicy, _score
from scripts import benchmark_training_row_budget as fits
from scripts.benchmark_signal_challengers import dump, incumbent_evidence
from scripts.fast_matched_bootstrap import accelerated_select_temporal_policy
from scripts.matched_learner_challengers import sha256


WEIGHTS = (.125, .5, 1.)
MENU_ALLOWANCE = 80


def primitive_policy(selected, candidates):
    policy = TemporalFitPolicy(**selected)
    name = policy.name.split("__blend_")[0]
    primitive = next((TemporalFitPolicy(**p) for p in candidates if p["name"] == name), None)
    if (primitive is None or primitive.mode != "pooled" or primitive.readout != "direct"
            or primitive.blend_weight != 1 or not 0 < policy.blend_weight <= 1):
        raise ValueError("An audited direct pooled selected policy is required")
    return policy, primitive


def replacement_prediction(strong, old_primitive, new_primitive, blend_weight, weight):
    """Replace only the selected pooled component, keeping its existing blend."""
    arrays = [np.asarray(x) for x in (strong, old_primitive, new_primitive)]
    if not all(a.shape == arrays[0].shape and np.isfinite(a).all() for a in arrays):
        raise ValueError("Aligned finite matched predictions required")
    if not 0 < blend_weight <= 1 or not 0 < weight <= 1:
        raise ValueError("Bounded fixed blend weights required")
    return arrays[0] + weight * blend_weight * (arrays[2] - arrays[1])


def run(args):
    started = time.monotonic()
    args.task_root = args.task_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.task_root / "raw-audit"
    audit = json.loads((raw / "manifest.json").read_text())
    kind = "classification" if audit["metadata"]["metric"] == "roc_auc" else "regression"
    selected = json.loads((args.task_root / "selection/results.json").read_text())
    policy, primitive = primitive_policy(selected["selected_policy"], audit["policies"])
    folds, seeds = list(range(len(audit["folds"]))), audit["seeds"]
    if len(folds) != 3 or sorted(seeds) != [0, 1, 2]:
        raise ValueError("Complete three-origin, three-seed control required")
    budgets = (audit["row_limit"], audit["row_limit"] * 2, audit["row_limit"] * 4)
    identity = dict(
        selected_policy=asdict(policy), primitive_policy=asdict(primitive),
        budgets=budgets, weights=WEIGHTS, menu_allowance=MENU_ALLOWANCE,
        task_root=str(args.task_root),
        source_sha256={str(p): sha256(p) for p in [
            Path(__file__), Path("scripts/benchmark_training_row_budget.py"),
            Path("scripts/benchmark_signal_challengers.py"), Path("scripts/fast_matched_bootstrap.py"),
            Path("kurversc/temporal_policy.py"),
            raw / "manifest.json", args.task_root / "selection/results.json",
            args.task_root / "selection/manifest.json",
        ]},
        scope="Frozen graph; only pooled training-row ceiling changes; matched against selected strong blend; validation only",
    )
    identity = json.loads(json.dumps(identity))
    path = args.output / "selected-budget-identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Immutable experiment inputs or code changed")
    dump(path, identity)
    fit_root = args.output / "fits-audit"
    fit_root.mkdir(exist_ok=True)
    # Exact grouping acceleration only; same source selector and random draws.
    fits.select_temporal_policy = accelerated_select_temporal_policy
    fits.run(argparse.Namespace(
        source_audit=raw, output=fit_root, policy=primitive.name,
        budgets=",".join(map(str, budgets)), workers=args.workers,
        threads=args.threads, reserve_cpus=16, reserve_gib=args.reserve_gib,
        estimated_worker_gib=48,
    ))
    evidence, reports, eligible = [], [], {
        f"rows{b}__weight_{w:g}" for b in budgets[1:] for w in WEIGHTS
    }
    for fold in folds:
        for seed in seeds:
            strong = incumbent_evidence(raw, fold, seed, policy)
            with np.load(raw / f"fold{fold}-{primitive.name}-seed{seed}.npz") as a:
                for name in ("truth", "groups"):
                    np.testing.assert_array_equal(strong[name], a[name])
                old = a["prediction"].copy()
            arrays = dict(truth=strong["truth"], groups=strong["groups"], control=strong["prediction"])
            scores = dict(control=_score(arrays["truth"], arrays["control"], kind))
            origin = audit["folds"][fold]["origin"]
            evidence.append(dict(origin=origin, seed=seed, policy="incumbent", **strong))
            for budget in budgets[1:]:
                with np.load(fit_root / f"fits/fold{fold}-rows{budget}-seed{seed}.npz") as a:
                    for name in ("truth", "groups"):
                        np.testing.assert_array_equal(strong[name], a[name])
                    prediction = a["prediction"].copy()
                for weight in WEIGHTS:
                    name = f"rows{budget}__weight_{weight:g}"
                    arrays[name] = replacement_prediction(strong["prediction"], old, prediction,
                        policy.blend_weight, weight)
                    scores[name] = _score(arrays["truth"], arrays[name], kind)
                    gain = (scores[name] - scores["control"]) * (1 if kind == "classification" else -1)
                    if gain <= 0:
                        eligible.discard(name)
                    evidence.append(dict(origin=origin, seed=seed, policy=name,
                        truth=arrays["truth"], groups=arrays["groups"], prediction=arrays[name]))
            folder = args.output / f"fold{fold}"
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / f"seed{seed}.npz", **arrays)
            report = dict(fold=fold, seed=seed, origin=origin, scores=scores,
                          gains={k: (v - scores["control"]) * (1 if kind == "classification" else -1)
                                 for k, v in scores.items() if k != "control"})
            reports.append(report)
            dump(folder / f"seed{seed}.json", report)
    decision = (accelerated_select_temporal_policy(
        [e for e in evidence if e["policy"] == "incumbent" or e["policy"] in eligible],
        task=kind, incumbent="incumbent", minimum_origins=3, minimum_seeds=3,
        deployment_seed=0, bootstrap_repeats=1600, alpha=.05 * len(eligible) / MENU_ALLOWANCE,
    ) if eligible else dict(selected="incumbent", reason="no_all_nine_incremental_improvement"))
    result = dict(decision=decision, reports=reports, complete_grid=True,
                  surviving=sorted(eligible), baseline=asdict(policy),
                  identity_sha256=sha256(path), elapsed_seconds=time.monotonic() - started,
                  scope=identity["scope"], official_score=None, default_changed=False)
    dump(args.output / "selection.json", result)
    print(json.dumps(dict(event="strong_baseline_gate_complete", **result)), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reserve-gib", type=float, default=128)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1 or args.reserve_gib < 128:
        parser.error("Positive fit concurrency and >=128 GiB host reserve required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
