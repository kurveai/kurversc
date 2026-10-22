"""Bounded, production-faithful comparison of two already-surviving finalists.

The ordinary search remains unchanged. For MAE, compare its winner with at most
one smaller, recent-competitive completed finalist. Replay the same recent
origins, mature labels, initial fit plus observed-validation refit, and three
seeds. Use the existing graph selector unchanged, with an additional unanimous
seed decision and per-fold practical safety cap. Never inspect official scores.
This is an opt-in experiment, not a change to the registered defaults.
"""

import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from scripts.engagement_local_comparison import freeze
from scripts.run_gap_followup_queue import digest, dump
from scripts.benchmark_small_finalist_readout import small_finalist


CONTRACT = dict(maximum_additional_graphs=1, metric="mae", origins=3,
    seeds=[0, 1, 2], compare_native_production_lifecycle=True,
    initial_training_frames=1, observed_validation_refit=True,
    recent_relative_loss_cap=.01, older_relative_loss_cap=.01,
    selection="existing_core_temporal_graph_selector",
    require_unanimous_seed_selection=True, independent_recent_replay=True,
    historical_winner_lookup=False, official_targets_used_for_selection=False)


def load_trial(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)["trial"]  # trusted artifacts from this run


def choose_pair(graph_root, search_result, dataset, task):
    row = json.loads(Path(search_result).read_text())
    if row["metric"] != "mae":
        return None
    wanted = json.loads(row["config"])
    trials = [(p, load_trial(p)) for p in sorted((Path(graph_root) / "checkpoints").glob(
        f"*/{dataset}_{task}/*/trials/*.pkl"))]
    selected = [(p, t) for p, t in trials if t.stage == "rerank"
        and t.status == "completed" and json.loads(json.dumps(asdict(t.config))) == wanted]
    if len(selected) != 1:
        raise ValueError("Exactly one completed current-run rerank winner is required")
    alternative = small_finalist(trials, selected[0][1].config)
    if alternative is not None and alternative[1].feature_count * 2 > selected[0][1].feature_count:
        alternative = None
    return (selected[0], alternative) if alternative is not None else None


def compare_cases(incumbent, challenger, control, candidate):
    """Pure validation selection, separate from any official replay/artifact."""
    from kurversc.core import _rerank_objective, _select_temporal_rerank_trials

    expected = {(fold, seed) for fold in range(3) for seed in range(3)}
    if set(control) != expected or set(candidate) != expected:
        raise ValueError("Complete three-origin/three-seed evidence is required")
    if incumbent.metric != "mae" or challenger.metric != "mae":
        raise ValueError("This optional bounded comparison is MAE-only")
    if challenger.feature_count * 2 > incumbent.feature_count:
        return dict(selected="incumbent", reason="no_drastic_complexity_reduction")
    decisions, safety, summaries = {}, True, {}
    for seed in range(3):
        records = []
        for name, trial, cases in (("incumbent", incumbent, control),
                                    ("challenger", challenger, candidate)):
            scores, errors, labels = [], [], []
            for fold in range(3):
                key = (fold, seed)
                a, b = control[key], candidate[key]
                for field in ("truth", "groups"):
                    np.testing.assert_array_equal(a[field], b[field])
                if a["origin"] != b["origin"]:
                    raise ValueError("Finalists were evaluated at different origins")
                y, prediction = np.asarray(cases[key]["truth"]), np.asarray(cases[key]["prediction"])
                if not y.size or y.shape != prediction.shape or not np.isfinite(prediction).all():
                    raise ValueError("Complete finite aligned validation vectors required")
                loss = np.abs(y - prediction)
                if not np.isfinite(loss).all():
                    raise ValueError("Finite mature validation targets required")
                scores.append(float(loss.mean()))
                errors.append(float(loss.std(ddof=1) / np.sqrt(len(loss))) if len(loss) > 1 else 0.)
                labels.append(cases[key]["origin"])
            objective, se = _rerank_objective("mae", float(np.mean(scores)),
                float(np.std(scores)), len(scores), .25)
            records.append(replace(trial, model=None, stage="rerank", validation_score=float(np.mean(scores)),
                objective_score=objective, validation_standard_error=se,
                cutoff_labels=tuple(labels), cutoff_scores=tuple(scores),
                cutoff_standard_errors=tuple(errors)))
            summaries[f"{name}_seed{seed}"] = dict(scores=scores, objective=objective,
                features=trial.feature_count)
        # Preserve the existing practical-loss/uncertainty thresholds. This is
        # not a claim of a statistically significant accuracy gain; a smaller
        # practical tie is an explicitly supported graph-selection outcome.
        chosen = _select_temporal_rerank_trials(records,
            classification_negligible_gain=.002, regression_negligible_relative_gain=.005,
            complexity_uncertainty_multiplier=1., complexity_selector=True,
            practical_recent_ties=True)[-1]
        decisions[str(seed)] = "challenger" if chosen.config == challenger.config else "incumbent"
        safety = safety and all(b <= a * 1.01 + 1e-12
            for a, b in zip(records[0].cutoff_scores, records[1].cutoff_scores))
    admitted = safety and all(v == "challenger" for v in decisions.values())
    return dict(selected="challenger" if admitted else "incumbent",
        reason="unanimous_production_fidelity_complexity_tie" if admitted else "incumbent_retained",
        seed_decisions=decisions, all_seed_fold_practical_safety=safety, evidence=summaries,
        official_targets_used_for_selection=False)


def read_cases(root):
    root = Path(root)
    raw = json.loads((root / "raw-audit/manifest.json").read_text())
    if len(raw["folds"]) != 3 or raw["seeds"] != [0, 1, 2] or raw["observed_refit_frames"] != 1:
        raise ValueError("Matched three-origin, three-seed, single-refit procedure required")
    if not raw["metadata"]["atomic_date_replay"]:
        raise ValueError("Atomic point-in-time features required")
    result = {}
    for fold in range(3):
        for seed in range(3):
            path = root / "raw-audit" / f"fold{fold}-incumbent-seed{seed}.npz"
            with np.load(path) as a:
                result[fold, seed] = {k:a[k].copy() for k in ("truth", "groups", "prediction")}
            result[fold, seed]["origin"] = raw["folds"][fold]["origin"]
    return result


def independent_repeat(root, *, output=None):
    """Repeat recent seed zero through the ordinary estimator and serialization."""
    from kurversc.temporal_policy import TemporalFitPolicy
    from scripts.benchmark_recovered_readouts import phase_frames, prepared, fit_prediction
    import pandas as pd

    root = Path(root)
    raw = json.loads((root / "raw-audit/manifest.json").read_text())
    policy = TemporalFitPolicy("incumbent")
    train, held, phase = phase_frames(root, raw, policy, 2, maximum_feature_cells=1_000_000_000)
    if not phase["optional_fit_available"]:
        raise ValueError("Production finalist exceeds the declared replay capacity")
    schema, meta = raw["metadata"]["feature_schema"], raw["metadata"]
    prediction, audit = fit_prediction(prepared(train, schema), train[meta["target"]],
        pd.to_datetime(train.__event_cutoff), prepared(held, schema), raw=raw,
        meta=meta, policy=policy, phase=phase, seed=0)
    with np.load(root / "raw-audit/fold2-incumbent-seed0.npz") as a:
        np.testing.assert_array_equal(a["truth"], held[meta["target"]])
        np.testing.assert_array_equal(a["groups"], held.__event_entity)
        np.testing.assert_allclose(prediction, a["prediction"], rtol=0, atol=1e-9)
        error = float(np.max(np.abs(prediction-a["prediction"])))
    dump(Path(output) if output is not None else root / "independent-repeat.json", dict(maximum_error=error,
        serialized_predictions_exact=True, audit=audit))
    del train, held, prediction
    gc.collect()


def arm_stage(args):
    from scripts import benchmark_small_finalist_readout as small
    arm = args.output / "arms" / args.arm
    values = argparse.Namespace(output=arm, dataset=args.dataset, task=args.task)
    if args.stage == "materialize":
        small.materialize(values)
    elif args.stage == "snapshots":
        from scripts import materialize_policy_folds as snapshots
        sys.argv = [snapshots.__file__, "--cache", str(arm / "task/cache"),
            "--output", str(arm / "task/snapshots"), "--origins", "3",
            "--duckdb-memory-limit", "none", "--duckdb-temp-limit", "none"]
        with small.normalized_engine_limits():
            snapshots.main()
    elif args.stage == "native":
        small.native_audit(values)
    else:
        independent_repeat(arm / "task")


def run(args):
    from scripts.run_kurversc_local_campaign import _run_guarded
    from scripts.run_fresh_cloud_campaign import ROOT
    started = time.monotonic()
    pair = choose_pair(args.graph_root, args.search_result, args.dataset, args.task)
    manifest = dict(contract=CONTRACT, dataset=args.dataset, task=args.task,
        search_result=str(args.search_result), search_result_sha256=digest(args.search_result),
        candidates=[] if pair is None else [dict(arm=name, checkpoint=str(path),
            checkpoint_sha256=digest(path), config=asdict(trial.config), feature_count=trial.feature_count)
            for name, (path, trial) in zip(("incumbent", "challenger"), pair)],
        source_sha256=digest(__file__))
    freeze(args.output / "manifest.json", manifest)
    if pair is None:
        dump(args.output / "results.json", dict(selected="incumbent", reason="no_bounded_alternative",
            official_targets_used_for_selection=False))
        return
    for candidate in manifest["candidates"]:
        arm = args.output / "arms" / candidate["arm"]
        freeze(arm / "plan.json", candidate)
        stages = (("materialize", arm / "task/cache/materialization.json"),
            ("snapshots", arm / "task/snapshots/manifest.json"),
            ("native", arm / "task/raw-audit/results.json"),
            ("repeat", arm / "task/independent-repeat.json"))
        for name, completion in stages:
            if completion.exists():
                continue
            if digest(candidate["checkpoint"]) != candidate["checkpoint_sha256"]:
                raise ValueError("Fresh finalist checkpoint changed")
            dump(args.output / "progress.json", dict(active=candidate["arm"], stage=name))
            command = [sys.executable, "-u", "-m", "scripts.benchmark_production_finalists",
                "--mode", "arm", "--stage", name, "--arm", candidate["arm"],
                "--graph-root", str(args.graph_root), "--search-result", str(args.search_result),
                "--output", str(args.output), "--dataset", args.dataset, "--task", args.task]
            with (arm / f"{name}.log").open("a", buffering=1) as log:
                code, usage = _run_guarded(command, cwd=ROOT, environment=__import__("os").environ.copy(),
                    log=log, minimum_available_memory_gb=128, poll_seconds=5,
                    task_time_limit_seconds=max(1, args.hours*3600-(time.monotonic()-started)))
            dump(arm / f"{name}-resources.json", usage)
            if code or not completion.exists():
                raise RuntimeError(f"Production finalist {candidate['arm']}/{name} incomplete: {code}")
    result = compare_cases(pair[0][1], pair[1][1],
        read_cases(args.output / "arms/incumbent/task"), read_cases(args.output / "arms/challenger/task"))
    chosen = manifest["candidates"][result["selected"] == "challenger"]
    dump(args.output / "results.json", dict(**result, checkpoint=chosen["checkpoint"],
        checkpoint_sha256=chosen["checkpoint_sha256"], config=chosen["config"],
        elapsed_seconds=time.monotonic()-started, manifest_sha256=digest(args.output / "manifest.json")))


def verify_existing(args):
    """Qualify the correction on matched current-run audits, not old winners.

    The optional operational launch starts a NEW search; these graphs, caches,
    predictions and scores are never imported into that fresh reproduction.
    """
    from scripts.run_fresh_cloud_campaign import ROOT
    pair = choose_pair(args.graph_root, args.search_result, args.dataset, args.task)
    if pair is None or args.incumbent_task_root is None or args.challenger_task_root is None:
        raise ValueError("Two current-run finalists and their complete native audits required")
    identity = dict(contract=CONTRACT, source_sha256=digest(__file__),
        search_result_sha256=digest(args.search_result), cases={})
    roots = (args.incumbent_task_root, args.challenger_task_root)
    for name, root, (checkpoint, _) in zip(("incumbent", "challenger"), roots, pair):
        raw_path = root / "raw-audit/manifest.json"
        raw = json.loads(raw_path.read_text())
        if raw["metadata"]["checkpoint_sha256"] != digest(checkpoint):
            raise ValueError("Evidence does not belong to the automatically selected fresh finalist")
        identity["cases"][name] = {str(p):digest(p) for p in sorted((root / "raw-audit").glob("fold*-incumbent-seed*.npz"))}
        identity["cases"][name][str(raw_path)] = digest(raw_path)
    freeze(args.output / "manifest.json", identity)
    result = compare_cases(pair[0][1], pair[1][1], *(read_cases(root) for root in roots))
    if result["selected"] == "challenger":
        for name, root in zip(("incumbent", "challenger"), roots):
            output = args.output / f"{name}-independent-repeat.json"
            if not output.exists():
                independent_repeat(root, output=output)
            if json.loads(output.read_text())["maximum_error"] > 1e-9:
                raise ValueError("Independent production replay failed")
    dump(args.output / "results.json", dict(**result, independent_replay_completed=result["selected"] == "challenger",
        official_test_accessed=False, starts_new_search_on_launch=True))
    if args.launch_qualified_output is not None and result["selected"] == "challenger":
        command = [sys.executable, "-u", "-m", "scripts.run_production_finalist_cloud",
            "--dataset", args.dataset, "--task", args.task, "--output", str(args.launch_qualified_output)]
        with (args.output / "launch.log").open("a") as log:
            code = subprocess.call(command, cwd=ROOT, env=__import__("os").environ.copy(),
                stdout=log, stderr=subprocess.STDOUT)
        dump(args.output / "launch-status.json", dict(status="launched" if code == 0 else "launch_failed",
            returncode=code, output=str(args.launch_qualified_output)))
        if code:
            raise RuntimeError(f"Fresh qualified reproduction launch failed: {code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "arm", "verify"), default="run")
    parser.add_argument("--stage", choices=("materialize", "snapshots", "native", "repeat"))
    parser.add_argument("--arm", choices=("incumbent", "challenger"))
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--search-result", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=4.)
    parser.add_argument("--incumbent-task-root", type=Path)
    parser.add_argument("--challenger-task-root", type=Path)
    parser.add_argument("--launch-qualified-output", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.hours <= 0 or args.hours >= 24:
        parser.error("Positive bounded finalist audit under 24 hours required")
    if args.mode == "arm":
        if args.arm is None or args.stage is None:
            parser.error("Arm and stage required")
        arm_stage(args)
    elif args.mode == "verify":
        verify_existing(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
