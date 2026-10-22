"""Refit one validation-locked bounded-view mean on a fresh owned graph.

Replay the exact native production path first; no historical model/score fill.
All four keyed vectors are frozen before official scoring. Independent repeat
mode compares complete fresh fits and never invokes the hidden-target scorer.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

import kurversc
from kurversc.core import _normalize_tables
from kurversc.prediction_provenance import artifact_keys, align_prediction, key_arrays, key_fingerprint, masked_query_keys
from kurversc.temporal_policy import TemporalFitPolicy
from scripts.association_state_compression_20261020 import AssociationState
from scripts.benchmark_large_gap_challengers import fit_prediction
from scripts.benchmark_recovered_readouts import matched_gate
from scripts.benchmark_sequence_recovery_20261020 import verify
from scripts.benchmark_signal_challengers import prepared
from scripts.capture_owned_pooled_native_20261020 import materialize
from scripts.engagement_local_comparison import freeze
from scripts.latest_completed_peers_20261020 import LatestCompletedPeers
from scripts.qualify_bounded_view_consensus_20261020 import compose, RECIPES
from scripts.run_gap_followup_queue import digest, dump
from scripts.source_response_annotation_20261020 import SourceResponseAnnotation


def admission(root, metadata):
    folder = root / "consensus"
    m = json.loads((folder / "manifest.json").read_text())
    r = json.loads((folder / "results.json").read_text())
    if (not r["complete_grid"] or r["official_test_accessed"] is not False or m["metadata"] != metadata
            or r["selected"] not in RECIPES or m["allowance"] != 1024):
        raise ValueError("Complete matched target-unseen bounded-view admission required")
    verify(m["sources"])
    verify(m["inputs"])
    for component in m["component_headers"]:
        verify(component["sources"])
        verify(component["inputs"])
    expected = {(f, s) for f in m["folds"] for s in range(3)}
    if len(r["records"]) != len(expected) or {(x["fold"], x["seed"]) for x in r["records"]} != expected:
        raise ValueError("Every declared origin/seed case must exist exactly once")
    paired = []
    for row in r["records"]:
        path = folder / f"fold{row['fold']}/seed{row['seed']}.npz"
        if digest(path) != row["vectors_sha256"]:
            raise ValueError("Validation-locked consensus vectors changed")
        with np.load(path) as a:
            paired.append(dict(origin=row["origin"], seed=row["seed"], truth=a["truth"].copy(),
                groups=a["groups"].copy(), control=a["control"].copy(), vectors={n: a[n].copy() for n in RECIPES}))
    replay = matched_gate(paired, metric="roc_auc", recipes=RECIPES, allowance=1024)
    if replay != r["gate"] or replay["selected"] != r["selected"]:
        raise ValueError("The unchanged full validation-only gate did not replay exactly")
    return m, r


def views(train, forecast, *, source, outer, metadata, source_root, temporary):
    task = source.task
    problem = kurversc.relbench_problem_from_objects(task, outer.db_state, outer.train_table, outer.val_table,
        dataset_name=metadata["dataset"], task_name=metadata["task"], sample_rows=1000,
        search_training_frames=1, max_train_timestamps=3, schema_depth=4,
        max_base_table_occurrences=1, include_target_history=False, random_state=0)
    root, tables = _normalize_tables(problem.parent_node, problem.tables)
    raw = pd.concat([t.df for t in (outer.train_table, outer.val_table)], ignore_index=True)
    fields = [task.entity_col, task.time_col, task.target_col]
    history = raw[fields].rename(columns=dict(zip(fields, ("__event_entity", "__event_cutoff", "__outcome"))))
    actual_availability = getattr(task, "label_available_at", None)
    history["__available_at"] = (pd.to_datetime(raw[actual_availability]).to_numpy() if actual_availability
        else pd.to_datetime(history.__event_cutoff)+pd.Timedelta(metadata["label_horizon"]))
    if len(history)>500_000 or max(len(train), len(forecast))>250_000:
        raise ValueError("Final complete bounded-view payload exceeds its qualified capacity")
    latest = LatestCompletedPeers().fit(history, tables=tables, relationships=problem.relationships,
        root_name=root, phase_boundary=outer.cutoff)
    peer = [latest.transform(frame) for frame in (train, forecast)]
    association = AssociationState()
    incidence, supported = association.fit_transform(train, forecast, tables=tables,
        relationships=problem.relationships, root_name=root, phase_boundary=outer.cutoff)
    annotation = SourceResponseAnnotation(source_root / "db", root,
        training_cutoff=pd.to_datetime(train.__event_cutoff).min(), horizon=metadata["label_horizon"],
        temporary=temporary / "source-annotation")
    try:
        response = [annotation.transform(f[["__event_entity", "__event_cutoff"]],
            phase_boundary=outer.cutoff) for f in (train, forecast)]
        response_support = annotation.support.copy()
        annotation_audit = annotation.audit
    finally:
        annotation.close()
    return dict(latest=peer, association=incidence, annotation=response), dict(
        association=supported[1], annotation=response_support), dict(
        latest=latest.audit, association=association.audit, annotation=annotation_audit)


def run(args):
    began = time.monotonic()
    raw = json.loads((args.task_root / "raw-audit/manifest.json").read_text())
    meta = raw["metadata"]
    overview = json.loads((args.validation / "results.json").read_text())
    if overview["selected"] == "incumbent":
        if overview["official_test_accessed"] is not False or not args.task_root.resolve().is_relative_to(args.run_root.resolve()):
            raise ValueError("Only a target-unseen own-run validation fallback is allowed")
        args.output.mkdir(parents=True, exist_ok=True)
        marker = dict(selected="incumbent", incumbent_retained=True, scorer_called=False,
            fallback=overview.get("fallback", overview.get("gate", {}).get("reason")),
            validation_sha256=digest(args.validation / "results.json"),
            independent_repeat_exact=None, repeat_not_applicable="no admitted challenger",
            official_targets_used_for_selection=False, predictive_defaults_changed=False)
        freeze(args.output / "selection.json", marker)
        dump(args.output / "results.json", marker)
        return
    m, admitted = admission(args.validation, meta)
    policy = TemporalFitPolicy(**m["policy"])
    if (meta["metric"] != "roc_auc" or policy.mode != "pooled" or policy.readout != "direct"
            or policy.blend_weight != 1):
        raise ValueError("Faithful unblended binary pooled final-fit adapter required")
    for name in ("signal-challengers", "recovered-readouts"):
        if json.loads((args.task_root / name / "selection.json").read_text())["selected"] != "incumbent":
            raise ValueError("An already admitted stronger readout must not be overwritten")
    sequence = args.task_root / "production-sequence/admission.json"
    if sequence.exists() and json.loads(sequence.read_text())["selected"] != "incumbent":
        raise ValueError("An already admitted sequence readout must remain the comparator")
    execution = args.task_root / "official/cloud-execution.json"
    if json.loads(execution.read_text())["catboost_threads"] != 8:
        raise ValueError("Final native capture must reproduce the recorded production learner")
    if m["model_params"] != {**raw["model_params"], "thread_count": 8}:
        raise ValueError("Native/candidate learner parameters differ from production qualification")
    identity = json.loads((args.run_root / "source-identity.json").read_text())
    verify(identity["source_sha256"])
    functions = (materialize, views, fit_prediction, compose, LatestCompletedPeers, AssociationState, SourceResponseAnnotation)
    sources = {str(p): digest(p) for p in {Path(__file__), *[Path(sys.modules[f.__module__].__file__) for f in functions]}}
    files = [args.validation / "consensus" / n for n in ("manifest.json", "results.json")]
    files += [args.task_root / "raw-audit/manifest.json", args.task_root / "selection/results.json", execution,
        args.task_root / "recovered-readouts/predictions.npz", args.run_root / "source-identity.json"]
    files += [*sorted((args.source_root / "db").glob("*.parquet")),
        *[args.source_root / "tasks" / meta["task"] / f"{split}.parquet" for split in ("train", "val")]]
    selection = dict(profile="bounded_view_owned_fresh_confirmation_v1", selected=admitted["selected"],
        metadata=meta, policy=m["policy"], model_params=m["model_params"], deployment_seed=0,
        sources=sources, inputs={str(p): digest(p) for p in files}, official_targets_used_for_selection=False)
    args.output.mkdir(parents=True, exist_ok=True)
    freeze(args.output / "selection.json", selection)
    if (args.output / "results.json").exists():
        raise ValueError("Completed confirmation cannot be overwritten")
    print(json.dumps(dict(phase="fresh_owned_native_capture", selected=admitted["selected"])), flush=True)
    source, outer, train, forecast, actual_native = materialize(args, meta)
    keys = masked_query_keys(forecast, entity="__event_entity", timestamp="__event_cutoff", target=meta["target"])
    with np.load(args.task_root / "recovered-readouts/predictions.npz") as a:
        expected = align_prediction(a["candidate"], source_keys=artifact_keys(a), query_keys=keys)
    np.testing.assert_array_equal(actual_native, expected)
    train.to_parquet(args.output / "mature-training.parquet", index=False)
    forecast.to_parquet(args.output / "masked-forecast.parquet", index=False)
    x, vx = prepared(train, meta["feature_schema"]), prepared(forecast, meta["feature_schema"])
    kwargs = dict(raw={**raw, "model_params": m["model_params"]}, meta=meta, policy=policy,
        phase=dict(fit_origin=str(outer.cutoff)), seed=0)
    native, lifecycle = fit_prediction(x, train[meta["target"]], pd.to_datetime(train.__event_cutoff), vx, **kwargs)
    np.testing.assert_array_equal(native, expected)
    extra, supports, audits = views(train, forecast, source=source, outer=outer, metadata=meta,
        source_root=args.source_root, temporary=args.output / "feature-state")
    heads = {}
    for name in ("latest", "association", "annotation"):
        tx, tv = extra[name]
        if set(tx)&set(x) or len(tx)!=len(x) or len(tv)!=len(vx) or list(tx)!=list(tv):
            raise ValueError("Qualified bounded view overwrites native fields or loses request alignment")
        xx, vxx = [pd.concat([a.reset_index(drop=True), b], axis=1) for a, b in ((x, tx), (vx, tv))]
        print(json.dumps(dict(phase="fresh_bounded_view_fit", view=name, columns=len(tx.columns))), flush=True)
        head, _ = fit_prediction(xx, train[meta["target"]], pd.to_datetime(train.__event_cutoff), vxx, **kwargs)
        if name in supports:
            head[~supports[name]] = native[~supports[name]]
        heads[name] = head
    candidate = compose(native, [heads[n] for n in ("latest", "association", "annotation")])[admitted["selected"]]
    arrays = dict(control=native, candidate=candidate, **heads, **key_arrays(keys))
    verify(selection["sources"])
    verify(selection["inputs"])
    verify(m["sources"])
    verify(m["inputs"])
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    if args.repeat_from:
        if json.loads((args.repeat_from / "selection.json").read_text()) != selection:
            raise ValueError("Independent confirmation must preserve every locked procedure choice")
        with np.load(args.repeat_from / "predictions.npz") as a:
            for key, values in arrays.items():
                np.testing.assert_array_equal(a[key], values)
        result = dict(independent_repeat_exact=True, scorer_called=False)
    elif getattr(source.task, "has_hidden_targets", True):
        result = dict(scorer_called=True, official_test_scores={name: float(source.task.evaluate(values,
            target_table=None)[meta["metric"]]) for name, values in (("control", native), ("candidate", candidate))})
    else:
        result = dict(scorer_called=False, test_score=None, portable_unscored=True)
    dump(args.output / "results.json", dict(result, selected=admitted["selected"], dataset=meta["dataset"], task=meta["task"],
        metric=meta["metric"], native_exact=True, lifecycle=lifecycle, feature_audits=audits,
        query_keys_sha256=key_fingerprint(keys), predictions_sha256=digest(args.output / "predictions.npz"),
        selection_sha256=digest(args.output / "selection.json"), deployment_seed=0,
        official_targets_used_for_selection=False, predictive_defaults_changed=False,
        elapsed_seconds=time.monotonic()-began))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root", "run-root", "source-root", "validation", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--repeat-from", type=Path)
    run(p.parse_args())
