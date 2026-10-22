"""Refit precisely one validation-locked sequence procedure before scoring.

Incumbent forecasts must reproduce first. A repeated process compares ordered
keyed vectors and never calls the official scorer. No holdout-based fallback.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from kurversc.prediction_provenance import align_prediction, artifact_keys, key_arrays, key_fingerprint, masked_query_keys
from kurversc.temporal_policy import TemporalFitPolicy
from scripts.benchmark_large_gap_challengers import fit_prediction
from scripts.benchmark_production_sequence import CONTRACT, payloads, verify
from scripts.engagement_local_comparison import freeze
from scripts.materialize_sequence_incumbent import materialize
from scripts.run_gap_followup_queue import digest, dump
from scripts.sequence_recovery_protocol import choose, combine


def admission(root):
    header = json.loads((root / "manifest.json").read_text())
    choice = json.loads((root / "selection.json").read_text())
    if (header.get("contract") != CONTRACT or choice.get("selected") == "incumbent"
            or not choice.get("complete_grid") or choice.get("official_targets_used_for_selection") is not False
            or choice.get("manifest_sha256") != digest(root / "manifest.json")):
        raise ValueError("Complete target-unseen production sequence admission required")
    verify(header["sources"])
    verify(header["inputs"])
    expected = {(f, s) for f in header["folds"] for s in range(3)}
    rows = choice["records"]
    if len(rows) != len(expected) or {(r["fold"], r["seed"]) for r in rows} != expected:
        raise ValueError("The declared full origin/seed grid must be present")
    evidence = []
    for row in rows:
        path = root / f"fold{row['fold']}/seed{row['seed']}.npz"
        if row["vectors_sha256"] != digest(path) or row["manifest_sha256"] != digest(root / "manifest.json"):
            raise ValueError("Production sequence vectors changed")
        with np.load(path) as a:
            evidence.append(dict(origin=row["origin"], seed=row["seed"], **{n: a[n].copy()
                for n in ("truth", "groups", "native", "basis", "exact")}))
    replay = choose(evidence)
    if any(choice[k] != v for k, v in replay.items()):
        raise ValueError("The same validation-only choice must replay exactly")
    return header, choice


def run(args):
    began = time.monotonic()
    header, choice = admission(args.validation)
    meta = header["metadata"]
    policy = TemporalFitPolicy(**header["policy"])
    if policy.name != "incumbent" or policy.mode != "frame_ensemble":
        raise ValueError("A pooled lifecycle requires a separate faithful final-fit replay")
    inputs = [args.validation / n for n in ("manifest.json", "selection.json")]
    inputs += [args.task_root / "recovered-readouts" / n for n in ("predictions.npz", "selection.json")]
    if args.expected_predictions:
        inputs.append(args.expected_predictions)
    modules = {Path(__file__), *[Path(sys.modules[f.__module__].__file__) for f in
        (materialize, payloads, fit_prediction, choose, combine)]}
    selected = dict(profile="production_sequence_confirmation_v1", selected=choice["selected"],
        coefficients=choice["coefficients"], policy=header["policy"], model_params=header["model_params"],
        metadata=meta, sources={str(p): digest(p) for p in modules},
        inputs={str(p): digest(p) for p in inputs}, official_targets_used_for_selection=False)
    args.output.mkdir(parents=True, exist_ok=True)
    freeze(args.output / "selection.json", selected)
    identity = digest(args.output / "selection.json")
    if (args.output / "results.json").exists():
        raise ValueError("Never rerun a completed official confirmation in place")
    source, outer, train, forecast, actual_native, lifecycle = materialize(args, meta, policy, header["model_params"])
    keys = masked_query_keys(forecast, entity="__event_entity", timestamp="__event_cutoff", target=meta["target"])
    with np.load(args.task_root / "recovered-readouts/predictions.npz") as a:
        expected = align_prediction(a["candidate"], source_keys=artifact_keys(a), query_keys=keys)
    np.testing.assert_array_equal(actual_native, expected)
    raw = json.loads((args.task_root / "raw-audit/manifest.json").read_text())
    raw = {**raw, "model_params": header["model_params"]}
    blocks = payloads(train, forecast, meta=meta, source_root=args.source_root,
        phase=outer.cutoff, temporary=args.output / "spill")
    components = {}
    for name, (x, vx) in zip(("native", "basis", "exact"), blocks):
        prediction, _ = fit_prediction(x, train[meta["target"]], pd.to_datetime(train.__event_cutoff), vx,
            raw=raw, meta=meta, policy=policy, phase=dict(fit_origin=str(outer.cutoff)), seed=0)
        components[name] = prediction
    np.testing.assert_array_equal(components["native"], expected)
    candidate = combine(components["native"], components["basis"], components["exact"], choice["coefficients"])
    arrays = dict(control=expected, candidate=candidate, **components, **key_arrays(keys))
    verify(selected["sources"])
    verify(selected["inputs"])
    if identity != digest(args.output / "selection.json"):
        raise ValueError("Selection changed after prediction began")
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    if args.expected_predictions:
        with np.load(args.expected_predictions) as expected_vectors:
            prior = align_prediction(expected_vectors["candidate"], source_keys=artifact_keys(expected_vectors), query_keys=keys)
            np.testing.assert_array_equal(candidate, prior)
    if args.repeat_from:
        if json.loads((args.repeat_from / "selection.json").read_text()) != selected:
            raise ValueError("An independent repeat requires an identical locked procedure")
        with np.load(args.repeat_from / "predictions.npz") as a:
            for n, v in arrays.items():
                np.testing.assert_array_equal(a[n], v)
        result = dict(independent_repeat_exact=True, scorer_called=False)
    elif args.expected_predictions:
        result = dict(previous_locked_prediction_exact=True, scorer_called=False)
    elif getattr(source.task, "has_hidden_targets", True):
        result = dict(scorer_called=True, official_test_scores={n: float(source.task.evaluate(v,
            target_table=None)[meta["metric"]]) for n, v in (("control", expected), ("candidate", candidate))})
    else:
        result = dict(scorer_called=False, test_score=None, portable_unscored=True)
    result.update(dataset=meta["dataset"], task=meta["task"], metric=meta["metric"],
        selected=choice["selected"], deployment_seed=0, native_exact=True, lifecycle=lifecycle,
        query_keys_sha256=key_fingerprint(keys), prediction_sha256=digest(args.output / "predictions.npz"),
        selection_sha256=identity, official_targets_used_for_selection=False, elapsed_seconds=time.monotonic()-began)
    dump(args.output / "results.json", result)
    # Native learner lifecycle audits can contain numpy scalar parameters.
    # They are already persisted by dump(); keep console output lightweight.
    print(json.dumps({k: v for k, v in result.items() if k != "lifecycle"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root", "source-root", "validation", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--repeat-from", type=Path)
    p.add_argument("--expected-predictions", type=Path,
        help="Compare to an already locked keyed vector without invoking a scorer")
    run(p.parse_args())
