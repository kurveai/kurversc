"""Production-matched optional sequence readout, selected without test answers.

One frozen schema-derived graph; three seeds at every available origin. This
module adds no graph configurations and preserves already-admitted readouts.
Unsupported optional domains/capacities are explicit exact-incumbent fallbacks.
Numerical/control failures are errors, never silently weaker predictions.
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
import pyarrow.parquet as pq

from kurversc.temporal_policy import TemporalFitPolicy, _score
from scripts.activity_timeline_challenger import discover_channels
from scripts.benchmark_large_gap_challengers import fit_prediction
from scripts.benchmark_recovered_readouts import phase_frames
from scripts.benchmark_signal_challengers import prepared
from scripts.engagement_local_comparison import freeze
from scripts.schema_semantic_recurrence import declared_root
from scripts.select_activity_readout import cache_locations
from scripts.sequence_basis import SequenceBasis
from scripts.sequence_recovery_protocol import choose
from scripts.subday_sequence import SubdaySequence
from scripts.run_gap_followup_queue import digest, dump


CONTRACT = dict(optional=True, graph_search_expanded=False, historical_winner_lookup=False,
    production_matched_native_and_heads=True, complete_available_origins=True,
    seeds=[0, 1, 2], deployment_seed=0, allowance=1024,
    maximum_feature_cells=125_000_000, maximum_rows=250_000,
    exact_incumbent_fallback=True, native_official_parity_required=True)


def applicability(raw, policy, selections):
    if raw.get("metadata", {}).get("metric") != "roc_auc":
        return "not_binary_auc"
    if not raw.get("metadata", {}).get("atomic_date_replay"):
        return "not_atomic_phase_replay"
    if not 2 <= len(raw.get("folds", [])) <= 3 or raw.get("seeds") != [0, 1, 2]:
        return "insufficient_available_temporal_evidence"
    if any(s.get("selected") != "incumbent" for s in selections):
        return "preserve_already_admitted_signal_or_readout"
    if (policy.get("blend_weight") != 1 or policy.get("readout") != "direct"
            or policy.get("mode") != "frame_ensemble" or policy.get("name") != "incumbent"):
        return "preserve_composite_native_procedure"
    return None


def sources():
    functions = (fit_prediction, phase_frames, prepared, declared_root, choose, SequenceBasis,
        SubdaySequence, discover_channels)
    return {str(p): digest(p) for p in {Path(__file__), *[
        Path(sys.modules[f.__module__].__file__) for f in functions]}}


def verify(mapping):
    if any(digest(path) != sha for path, sha in mapping.items()):
        raise ValueError("Production sequence inputs or implementation changed")


def channel_availability(channel):
    """Inspect physical metadata even with an older EventChannel adapter."""
    metadata = pq.ParquetFile(channel.path).schema_arrow.metadata or {}
    return json.loads(metadata.get(b"available_at", b"null"))


def delayed_channel(channel):
    available = channel_availability(channel)
    # Explicit-context exports may declare the event clock itself as its
    # availability clock. That is not a distinct late-arrival constraint.
    return bool(available and available != channel.timestamp)


def payloads(train, held, *, meta, source_root, phase, temporary):
    x, vx = prepared(train, meta["feature_schema"]), prepared(held, meta["feature_schema"])
    basis = SequenceBasis().fit(meta["feature_schema"])
    if not basis.columns:
        raise ValueError("Declared sequence algebra disappeared after applicability check")
    bx, bvx = [pd.concat([a.reset_index(drop=True), basis.transform(f)], axis=1)
        for a, f in ((x, train), (vx, held))]
    root, _, _ = declared_root(source_root / "tasks" / meta["task"] / "train.parquet", meta)
    channels = discover_channels(source_root / "db", root)
    provider = SubdaySequence(channels, horizon=meta["label_horizon"], temporary=temporary)
    try:
        extra, held_extra = [provider.transform(f[["__event_entity", "__event_cutoff"]],
            phase_boundary=phase) for f in (train, held)]
    finally:
        provider.close()
    if list(extra) != list(held_extra) or len(extra.columns) > SubdaySequence.maximum_columns:
        raise ValueError("Bounded sequence payload changed")
    xx, vxx = [pd.concat([a.reset_index(drop=True), b], axis=1)
        for a, b in ((x, extra), (vx, held_extra))]
    return ((x, vx), (bx, bvx), (xx, vxx))


def case(task_root, source_root, output, fold, seed):
    task_root, source_root, output = map(Path, (task_root, source_root, output))
    m = json.loads((output / "manifest.json").read_text())
    path = output / f"fold{fold}/seed{seed}.npz"
    path.parent.mkdir(exist_ok=True)
    if path.with_suffix(".json").exists():
        saved = json.loads(path.with_suffix(".json").read_text())
        if saved["vectors_sha256"] != digest(path) or saved["manifest_sha256"] != digest(output / "manifest.json"):
            raise ValueError("Production sequence checkpoint changed")
        return saved
    raw = json.loads((task_root / "raw-audit/manifest.json").read_text())
    raw = {**raw, "model_params": m["model_params"]}
    meta, policy = raw["metadata"], TemporalFitPolicy(**m["policy"])
    train, held, phase = phase_frames(task_root, raw, policy, fold,
        maximum_feature_cells=CONTRACT["maximum_feature_cells"], maximum_rows=CONTRACT["maximum_rows"])
    if not phase["optional_fit_available"]:
        raise ValueError("Production sequence capacity differed from preflight")
    blocks = payloads(train, held, meta=meta, source_root=source_root,
        phase=phase["fit_origin"], temporary=path.parent / f"seed{seed}-spill")
    kwargs = dict(raw=raw, meta=meta, policy=policy, phase=phase, seed=seed)
    arrays = dict(truth=held[meta["target"]].to_numpy(), groups=held.__event_entity.to_numpy())
    for name, (x, vx) in zip(("native", "basis", "exact"), blocks):
        print(json.dumps(dict(phase="production_sequence_fit", fold=fold, seed=seed, head=name)), flush=True)
        prediction, _ = fit_prediction(x, train[meta["target"]], pd.to_datetime(train.__event_cutoff), vx, **kwargs)
        arrays[name] = prediction
        if fold == max(m["folds"]) and seed == 0:
            repeated, _ = fit_prediction(x, train[meta["target"]], pd.to_datetime(train.__event_cutoff), vx, **kwargs)
            np.testing.assert_array_equal(prediction, repeated)
    np.savez_compressed(path, **arrays)
    row = dict(fold=fold, seed=seed, origin=phase["origin"],
        scores={n: _score(arrays["truth"], arrays[n], "classification") for n in ("native", "basis", "exact")},
        manifest_sha256=digest(output / "manifest.json"), vectors_sha256=digest(path))
    dump(path.with_suffix(".json"), row)
    del blocks, train, held
    gc.collect()
    return row


def validate(args):
    began = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.task_root / "raw-audit/manifest.json"
    if not raw_path.exists():
        result = dict(selected="incumbent", coefficients=[1., 0., 0.], fallback="no_atomic_validation_audit",
            complete_grid=False, official_targets_used_for_selection=False)
        freeze(args.output / "selection.json", result)
        return result
    raw = json.loads(raw_path.read_text())
    policy_path = args.task_root / "selection/results.json"
    policy = json.loads(policy_path.read_text())["selected_policy"]
    selection_paths = [args.task_root / name / "selection.json" for name in ("signal-challengers", "recovered-readouts")]
    selections = [json.loads(p.read_text()) for p in selection_paths]
    why = applicability(raw, policy, selections)
    folds = list(range(len(raw["folds"])))
    meta = raw["metadata"]
    input_paths = [raw_path, policy_path, *selection_paths]
    channels = ()
    if why is None:
        root, _, _ = declared_root(args.source_root / "tasks" / meta["task"] / "train.parquet", meta)
        channels = discover_channels(args.source_root / "db", root)
        if not channels or any(delayed_channel(c) for c in channels):
            why = "unsupported_or_unavailable_ordinary_event_channels"
        elif not SequenceBasis().fit(meta["feature_schema"]).columns:
            why = "no_schema_sequence_algebra"
    if why is None:
        cache, snapshots = cache_locations(args.task_root)
        # Conservative whole-cache preflight: no huge Parquet payload is loaded
        # just to decide that an optional learner cannot fit its declared bound.
        cells = pq.ParquetFile(cache / "history.parquet").metadata.num_rows * (len(meta["feature_schema"]["columns"])+224)
        if cells > CONTRACT["maximum_feature_cells"]:
            why = "optional_sequence_feature_capacity"
        input_paths += [cache / "history.parquet", *[snapshots / f"fold{f}" / n
            for f in folds for n in ("manifest.json", "validation.parquet")],
            args.source_root / "tasks" / meta["task"] / "train.parquet", *[c.path for c in channels]]
    execution_path = args.task_root / "official/cloud-execution.json"
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else {}
    threads = execution.get("catboost_threads", args.threads)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("Explicit positive production learner threads required")
    params = {**raw["model_params"], "thread_count": threads}
    if execution_path.exists():
        input_paths.append(execution_path)
    manifest = dict(profile="production_sequence_readout_v1", contract=CONTRACT, metadata=meta,
        policy=policy, model_params=params, folds=folds, seeds=[0, 1, 2], workers=args.workers,
        applicability_reason=why, sources=sources(), inputs={str(p): digest(p) for p in input_paths},
        official_targets_used_for_selection=False)
    freeze(args.output / "manifest.json", manifest)
    if why is not None:
        result = dict(selected="incumbent", coefficients=[1., 0., 0.], fallback=why,
            complete_grid=False, official_targets_used_for_selection=False)
    else:
        reports = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as pool:
            jobs = [pool.submit(case, str(args.task_root), str(args.source_root), str(args.output), f, s)
                for f in reversed(folds) for s in range(3)]
            for future in as_completed(jobs):
                reports.append(future.result())
                dump(args.output / "progress.json", dict(cases=len(reports), required_cases=len(jobs)))
        reports.sort(key=lambda r: (r["fold"], r["seed"]))
        evidence = []
        for r in reports:
            path = args.output / f"fold{r['fold']}/seed{r['seed']}.npz"
            if r["vectors_sha256"] != digest(path):
                raise ValueError("Sequence evidence changed before selection")
            with np.load(path) as a:
                evidence.append(dict(origin=r["origin"], seed=r["seed"], **{n: a[n].copy()
                    for n in ("truth", "groups", "native", "basis", "exact")}))
        result = dict(**choose(evidence), complete_grid=True, records=reports)
    verify(manifest["sources"])
    verify(manifest["inputs"])
    freeze(args.output / "selection.json", dict(result, manifest_sha256=digest(args.output / "manifest.json"),
        elapsed_seconds=time.monotonic()-began))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root", "source-root", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    if not 1 <= args.workers <= 3 or args.threads < 1:
        p.error("One to three workers and positive production threads required")
    print(json.dumps(validate(args)), flush=True)


if __name__ == "__main__":
    main()
