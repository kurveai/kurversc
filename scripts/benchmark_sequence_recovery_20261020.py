"""Full-grid, incumbent-preserving qualification of existing sequence algebra.

Two fixed procedures reconstruct the previously demonstrated sequence inputs.
They are discovered from schema metadata, not task names or historical scores.
The native graph, mature support, learner parameters and confidence rule stay
unchanged. No official evaluation data or score is opened by this module.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from kurversc.temporal_policy import TemporalFitPolicy, _score
from scripts.activity_timeline_challenger import discover_channels
from scripts.benchmark_large_gap_challengers import fit_prediction
from scripts.benchmark_recovered_readouts import phase_frames
from scripts.benchmark_signal_challengers import incumbent_evidence, prepared
from scripts.engagement_local_comparison import freeze
from scripts.frozen_auc_bootstrap import frozen_auc_select_temporal_policy
from scripts.run_gap_followup_queue import digest, dump
from scripts.schema_semantic_recurrence import declared_root
from scripts.select_activity_readout import cache_locations
from scripts.sequence_basis import SequenceBasis
from scripts.subday_sequence import SubdaySequence


RECIPES = ("sequence_basis_eighth", "sequence_basis_and_exact_time")


def candidate_vectors(native, basis, exact_time):
    native, basis, exact_time = [np.asarray(v, dtype=float) for v in (native, basis, exact_time)]
    if (native.ndim != 1 or not len(native) or any(v.shape != native.shape for v in (basis, exact_time))
            or not all(np.isfinite(v).all() for v in (native, basis, exact_time))):
        raise ValueError("Aligned finite sequence components required")
    return {
        RECIPES[0]: .875 * native + .125 * basis,
        RECIPES[1]: .765625 * native + .125 * basis + .109375 * exact_time,
    }


def choose(evidence, *, origins):
    if origins < 2:
        raise ValueError("Complete multi-origin validation required")
    result = frozen_auc_select_temporal_policy(evidence, task="classification",
        minimum_origins=origins, minimum_seeds=3, deployment_seed=0,
        bootstrap_repeats=1600, alpha=.05 * len(RECIPES) / 1024)
    eligible = []
    for name in RECIPES:
        row = result.get("candidates", {}).get(name, {})
        folds = row.get("folds", [])
        if (row.get("reason") == "eligible" and len(folds) == origins
                and folds[-1].get("recent_lower_bound", -1) > 0
                and folds[-1].get("temporal_gain_lower_bound", -1) > 0
                and min(folds[-1]["seed_gains"]) > 0
                and all(min(f["seed_gains"]) >= -f["practical_loss_cap"] for f in folds[:-1])):
            eligible.append((folds[-1]["gain"], RECIPES.index(name), name))
    return (min(eligible, key=lambda r: (-r[0], r[1]))[-1] if eligible else "incumbent"), result


def verify(files):
    if any(digest(p) != sha for p, sha in files.items()):
        raise ValueError("Frozen sequence qualification changed")


def case(task_root, output, fold, seed):
    task_root, output = Path(task_root), Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    folder = output / f"fold{fold}"
    folder.mkdir(exist_ok=True)
    path = folder / f"seed{seed}.npz"
    report = path.with_suffix(".json")
    if report.exists():
        record = json.loads(report.read_text())
        if record["vectors_sha256"] != digest(path) or record["manifest_sha256"] != digest(output / "manifest.json"):
            raise ValueError("Sequence case checkpoint changed")
        return record
    raw = json.loads((task_root / "raw-audit/manifest.json").read_text())
    meta, schema = raw["metadata"], raw["metadata"]["feature_schema"]
    policy = TemporalFitPolicy(**manifest["policy"])
    train, held, phase = phase_frames(task_root, raw, policy, fold)
    if not phase["optional_fit_available"]:
        raise ValueError("Complete sequence learner exceeds its declared feature capacity")
    x, vx = prepared(train, schema), prepared(held, schema)
    y, times = train[meta["target"]], pd.to_datetime(train.__event_cutoff)
    reference = incumbent_evidence(task_root / "raw-audit", fold, seed, policy)
    np.testing.assert_array_equal(reference["truth"], held[meta["target"]])
    np.testing.assert_array_equal(reference["groups"], held.__event_entity)
    kwargs = dict(raw=raw, meta=meta, policy=policy, phase=phase, seed=seed)
    print(json.dumps(dict(phase="native_sequence_replay", fold=fold, seed=seed)), flush=True)
    native, native_audit = fit_prediction(x, y, times, vx, **kwargs)
    np.testing.assert_allclose(native, reference["prediction"], rtol=0, atol=1e-9)
    basis = SequenceBasis().fit(schema)
    if not basis.columns:
        raise ValueError("No schema-derived sequence algebra applies")
    bx = pd.concat([x.reset_index(drop=True), basis.transform(train)], axis=1)
    bvx = pd.concat([vx.reset_index(drop=True), basis.transform(held)], axis=1)
    print(json.dumps(dict(phase="sequence_basis_fit", fold=fold, seed=seed)), flush=True)
    basis_prediction, _ = fit_prediction(bx, y, times, bvx, **kwargs)
    del bx, bvx
    gc.collect()
    source_root = Path(manifest["source_root"])
    root, _, _ = declared_root(source_root / "tasks" / meta["task"] / "train.parquet", meta)
    channels = discover_channels(source_root / "db", root)
    provider = SubdaySequence(channels, horizon=meta["label_horizon"],
        temporary=folder / f"seed{seed}-spill")
    try:
        extra = provider.transform(train[["__event_entity", "__event_cutoff"]], phase_boundary=phase["fit_origin"])
        held_extra = provider.transform(held[["__event_entity", "__event_cutoff"]], phase_boundary=phase["fit_origin"])
    finally:
        provider.close()
    if list(extra) != list(held_extra) or len(extra.columns) > SubdaySequence.maximum_columns:
        raise ValueError("Complete bounded exact-time schema changed")
    xx, vxx = [pd.concat([a.reset_index(drop=True), b], axis=1) for a, b in ((x, extra), (vx, held_extra))]
    print(json.dumps(dict(phase="exact_time_fit", fold=fold, seed=seed)), flush=True)
    exact_time, _ = fit_prediction(xx, y, times, vxx, **kwargs)
    repeated = None
    if fold == max(manifest["folds"]) and seed == 0:
        repeated, _ = fit_prediction(xx, y, times, vxx, **kwargs)
        np.testing.assert_allclose(exact_time, repeated, rtol=0, atol=1e-9)
    arrays = dict(truth=reference["truth"], groups=reference["groups"], control=reference["prediction"],
        native=native, basis=basis_prediction, exact_time=exact_time,
        **candidate_vectors(native, basis_prediction, exact_time))
    np.savez_compressed(path, **arrays)
    record = dict(fold=fold, seed=seed, origin=phase["origin"], phase=phase,
        native_archive_max_error=float(np.max(np.abs(native-reference["prediction"]))),
        exact_time_repeat_max_error=(float(np.max(np.abs(exact_time-repeated))) if repeated is not None else None),
        basis_columns=len(basis.columns), exact_time_columns=len(extra.columns), native_fit=native_audit,
        scores={n:_score(arrays["truth"], arrays[n], "classification") for n in ("control", *RECIPES)},
        manifest_sha256=digest(output / "manifest.json"), vectors_sha256=digest(path))
    dump(report, record)
    print(json.dumps(dict(phase="sequence_case_complete", fold=fold, seed=seed, scores=record["scores"])), flush=True)
    return record


def run(args):
    started = time.monotonic()
    raw_path = args.task_root / "raw-audit/manifest.json"
    raw = json.loads(raw_path.read_text())
    meta = raw["metadata"]
    if meta["metric"] != "roc_auc" or not meta["atomic_date_replay"] or meta["fold"] != "current":
        raise ValueError("Atomic phase-frozen binary task required")
    # This qualification is additive to an unchanged native procedure. Refuse
    # to overwrite any separately admitted signal or production readout.
    selections = [args.task_root / p / "selection.json" for p in ("signal-challengers", "recovered-readouts")]
    if any(json.loads(p.read_text())["selected"] != "incumbent" for p in selections):
        raise ValueError("A stronger admitted production component requires its own matched composition")
    policy_path = args.task_root / "selection/results.json"
    policy = json.loads(policy_path.read_text())["selected_policy"]
    folds = list(range(len(raw["folds"])))
    if not 2 <= len(folds) <= 3 or raw["seeds"] != [0, 1, 2]:
        raise ValueError("Complete available origins and three seeds required")
    cache, snapshots = cache_locations(args.task_root)
    root, _, _ = declared_root(args.source_root / "tasks" / meta["task"] / "train.parquet", meta)
    channels = discover_channels(args.source_root / "db", root)
    if not channels or any(getattr(c, "available_at", None) for c in channels):
        raise ValueError("Ordinary event-clock channels without a later availability clock required")
    sources = {Path(__file__), *[Path(sys.modules[f.__module__].__file__) for f in
        (fit_prediction, SequenceBasis, SubdaySequence, phase_frames, prepared, incumbent_evidence,
         frozen_auc_select_temporal_policy, discover_channels)]}
    inputs = [raw_path, policy_path, *selections, cache / "history.parquet",
        args.source_root / "tasks" / meta["task"] / "train.parquet", *[c.path for c in channels],
        *[snapshots / f"fold{f}" / name for f in folds for name in ("manifest.json", "validation.parquet")],
        *[args.task_root / f"raw-audit/fold{f}-{policy['name']}-seed{s}.npz" for f in folds for s in range(3)]]
    manifest = dict(profile="existing_sequence_recovery_validation_v1", metadata=meta, policy=policy,
        folds=folds, seeds=[0,1,2], recipes=RECIPES, workers=args.workers, allowance=1024,
        source_root=str(args.source_root), model_params=raw["model_params"], row_limit=raw["row_limit"],
        sources={str(p):digest(p) for p in sources}, inputs={str(p):digest(p) for p in inputs},
        graph_search_changed=False, default_changed=False, official_test_accessed=False,
        scope="Schema-derived composition; complete matched validation only, no historical-winner lookup")
    args.output.mkdir(parents=True, exist_ok=True)
    freeze(args.output / "manifest.json", manifest)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        jobs = [pool.submit(case, str(args.task_root), str(args.output), f, s)
            for f in reversed(folds) for s in range(3)]
        for job in as_completed(jobs):
            records.append(job.result())
            dump(args.output / "progress.json", dict(cases=len(records), required_cases=len(jobs)))
    records.sort(key=lambda r:(r["fold"], r["seed"]))
    evidence = []
    for record in records:
        with np.load(args.output / f"fold{record['fold']}/seed{record['seed']}.npz") as saved:
            for name in ("incumbent", *RECIPES):
                evidence.append(dict(origin=record["origin"], seed=record["seed"], policy=name,
                    truth=saved["truth"].copy(), groups=saved["groups"].copy(),
                    prediction=saved["control" if name == "incumbent" else name].copy()))
    selected, gate = choose(evidence, origins=len(folds))
    verify(manifest["sources"])
    verify(manifest["inputs"])
    result = dict(selected=selected, gate=gate, records=records, complete_grid=True,
        elapsed_seconds=time.monotonic()-started, default_changed=False, official_test_accessed=False)
    dump(args.output / "results.json", result)
    print(json.dumps(dict(selected=selected, cases=len(records))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root", "source-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("One to three independent case workers required")
    run(args)
