"""Rebuild a bounded view challenger from this run's own frozen graph cache.

No old winners, scores or model binaries are loaded. Generate an explicitly
production-thread native audit, then recompute the three fixed views and let
their fixed consensus pass/fail the existing validation-only gate. This is an
isolated qualification entry point, not a change to public fit defaults.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from kurversc.signal_challengers import ChallengerUnavailable
from kurversc.temporal_policy import TemporalFitPolicy
from scripts.benchmark_large_gap_challengers import fit_prediction
from scripts.benchmark_recovered_readouts import phase_frames
from scripts.benchmark_sequence_recovery_20261020 import verify
from scripts.benchmark_signal_challengers import prepared
from scripts.engagement_local_comparison import freeze
from scripts.enum_event_history_20261020 import source_metadata
from scripts.marked_response_history import discover_response_path, ResponsePathUnavailable
from scripts.run_gap_followup_queue import digest, dump
from scripts.schema_semantic_recurrence import declared_root
from scripts.select_activity_readout import cache_locations


def eligibility(raw, policy, selections):
    if raw.get("metadata", {}).get("metric") != "roc_auc":
        return "not_binary_auc"
    if not raw["metadata"].get("atomic_date_replay"):
        return "not_atomic_temporal_cache"
    if not 2 <= len(raw.get("folds", [])) <= 3 or raw.get("seeds") != [0, 1, 2]:
        return "insufficient_complete_origin_seed_grid"
    if any(x.get("selected") != "incumbent" for x in selections):
        return "preserve_already_admitted_procedure"
    if policy.get("blend_weight") != 1 or policy.get("readout") != "direct":
        return "preserve_composite_native_procedure"
    if policy.get("mode") != "pooled":
        # The owned final-fit confirmation currently qualifies only pooled
        # native fits. Do not admit a procedure that cannot be replayed by the
        # same final-fit adapter on another schema.
        return "unqualified_native_lifecycle"
    return None


def validation_source(dataset, task):
    # Public bundles contain no held-out answers and must not use a registry.
    from kurversc.task_source import CONTEXT_ENV, load_task_source
    if os.environ.get(CONTEXT_ENV):
        return load_task_source(dataset, task)
    from scripts.materialize_validation_origins import ValidationOnlySource
    return ValidationOnlySource(dataset, task)


def source_preflight(task_root, source_root, metadata):
    """Target-unseen applicability only: no learner fit or model checkpoint.

    Share this exact boundary with the cross-task surface audit, rather than
    letting a diagnostic approximation claim a different deployment surface.
    """
    import duckdb
    task_root, source_root = Path(task_root), Path(source_root)
    sources = [source_metadata(p) for p in sorted((source_root / "db").glob("*.parquet"))]
    if not sources:
        raise ValueError("Declared ordinary source tables are required")
    if any(m["available_at"] and m["available_at"] != m["time_col"] for m in sources):
        return "unsupported_delayed_source_arrival", {}
    root, _, _ = declared_root(source_root / "tasks" / metadata["task"] / "train.parquet", metadata)
    cache, _ = cache_locations(task_root)
    clocks = pd.read_parquet(cache / "history.parquet", columns=["__event_cutoff"])
    validation_rows = pq.ParquetFile(cache / "validation.parquet").metadata.num_rows
    columns = len(metadata["feature_schema"]["columns"])
    cells = (len(clocks)+validation_rows)*(columns+112)
    audit = dict(history_rows=len(clocks), validation_rows=validation_rows,
        native_columns=columns, maximum_added_columns=112, feature_cells=cells,
        root=root, learner_fits=0, model_checkpoints_loaded=0, official_test_accessed=False)
    if len(clocks)>250_000 or validation_rows>250_000 or cells>125_000_000:
        return "optional_bounded_view_capacity", audit
    if clocks.empty or clocks.__event_cutoff.isna().any():
        raise ValueError("Complete training-only temporal headers required")
    try:
        with duckdb.connect(config=dict(memory_limit="2GB", threads=1)) as con:
            path = discover_response_path(source_root / "db", root, pd.to_datetime(clocks.__event_cutoff).min(), con)
    except ResponsePathUnavailable:
        return "no_declared_ordinary_binary_response", audit
    from dataclasses import asdict
    audit["ordinary_response_path"] = asdict(path)
    return None, audit


def native_case(task_root, output, fold, seed):
    task_root, output = Path(task_root), Path(output)
    m = json.loads((output / "manifest.json").read_text())
    path = output / f"fold{fold}/seed{seed}.npz"
    if path.with_suffix(".json").exists():
        saved = json.loads(path.with_suffix(".json").read_text())
        if saved["vectors_sha256"] != digest(path) or saved["manifest_sha256"] != digest(output / "manifest.json"):
            raise ValueError("Completed native contract case changed")
        return saved
    raw = json.loads((task_root / "raw-audit/manifest.json").read_text())
    raw = {**raw, "model_params": m["model_params"]}
    meta, policy = raw["metadata"], TemporalFitPolicy(**m["policy"])
    train, held, phase = phase_frames(task_root, raw, policy, fold,
        maximum_feature_cells=125_000_000, maximum_rows=250_000)
    if not phase["optional_fit_available"]:
        raise ValueError("Complete native payload exceeds the optional view budget")
    x, vx = prepared(train, meta["feature_schema"]), prepared(held, meta["feature_schema"])
    kwargs = dict(raw=raw, meta=meta, policy=policy, phase=phase, seed=seed)
    control, _ = fit_prediction(x, train[meta["target"]], pd.to_datetime(train.__event_cutoff), vx, **kwargs)
    if fold == max(m["folds"]) and seed == 0:
        again, _ = fit_prediction(x, train[meta["target"]], pd.to_datetime(train.__event_cutoff), vx, **kwargs)
        np.testing.assert_array_equal(control, again)
    p = path
    p.parent.mkdir(exist_ok=True)
    np.savez_compressed(p, truth=held[meta["target"]].to_numpy(), groups=held.__event_entity.to_numpy(), control=control)
    row = dict(fold=fold, seed=seed, origin=phase["origin"], vectors_sha256=digest(p),
        manifest_sha256=digest(output / "manifest.json"))
    dump(p.with_suffix(".json"), row)
    return row


def native_audit(args, raw, policy):
    cache, snapshots = cache_locations(args.task_root)
    execution = args.task_root / "official/cloud-execution.json"
    threads = json.loads(execution.read_text())["catboost_threads"]
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("Explicit recorded production thread count required")
    params = {**raw["model_params"], "thread_count": threads}
    files = [args.task_root / "raw-audit/manifest.json", args.task_root / "selection/results.json", execution,
        cache / "history.parquet", cache / "validation.parquet",
        *[snapshots / f"fold{f}" / n for f in range(len(raw["folds"])) for n in ("manifest.json", "validation.parquet")]]
    out = args.output / "native"
    out.mkdir(exist_ok=True)
    functions = (native_case, fit_prediction, phase_frames, prepared)
    m = dict(profile="bounded_view_production_native_contract_v1", metadata=raw["metadata"], policy=policy,
        model_params=params, folds=list(range(len(raw["folds"]))), seeds=[0, 1, 2],
        sources={str(p): digest(p) for p in {Path(sys.modules[f.__module__].__file__) for f in functions}},
        inputs={str(p): digest(p) for p in files}, official_test_accessed=False)
    freeze(out / "manifest.json", m)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as pool:
        jobs = [pool.submit(native_case, str(args.task_root), str(out), f, s) for f in m["folds"] for s in range(3)]
        for job in as_completed(jobs):
            rows.append(job.result())
    rows.sort(key=lambda r: (r["fold"], r["seed"]))
    verify(m["sources"])
    verify(m["inputs"])
    freeze(out / "results.json", dict(complete_grid=True, records=rows, official_test_accessed=False))
    proxy = args.output / "production-contract-task"
    proxy.mkdir(exist_ok=True)
    for name, path in (("cache", cache), ("snapshots", snapshots)):
        link = proxy / name
        if not link.exists():
            link.symlink_to(path.resolve(), target_is_directory=True)
        if link.resolve() != path.resolve():
            raise ValueError("Optional validation view points outside its own campaign cache")
    freeze(proxy / "raw-audit/manifest.json", dict(raw, model_params=params, cache=str(cache), phase_frozen_cache=str(snapshots)))
    freeze(proxy / "selection/manifest.json", dict(metadata=raw["metadata"], cache=str(cache), phase_frozen_cache=str(snapshots)))
    freeze(proxy / "selection/results.json", json.loads((args.task_root / "selection/results.json").read_text()))
    return proxy, out


def run(args):
    began = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        raise ValueError("Completed qualification must not be overwritten")
    raw = json.loads((args.task_root / "raw-audit/manifest.json").read_text())
    if getattr(args, "run_root", None) is not None:
        if not args.task_root.resolve().is_relative_to(args.run_root.resolve()):
            raise ValueError("Qualification must consume this fresh run's own task artifacts")
        key = f"{raw['metadata']['dataset']}_{raw['metadata']['task']}"
        task_report = json.loads((args.run_root / "tasks" / key / "results.json").read_text())
        if task_report.get("status") != "completed" or not 0 < task_report["time_total"] < 24*3600:
            raise ValueError("A completed, in-budget fresh native pipeline is required")
        verify(json.loads((args.run_root / "source-identity.json").read_text())["source_sha256"])
    policy = json.loads((args.task_root / "selection/results.json").read_text())["selected_policy"]
    paths = [args.task_root / stage / "selection.json" for stage in ("signal-challengers", "recovered-readouts")]
    optional = args.task_root / "production-sequence/admission.json"
    if optional.exists():
        paths.append(optional)
    why = eligibility(raw, policy, [json.loads(p.read_text()) for p in paths])
    if why is None:
        why, _ = source_preflight(args.task_root, args.source_root, raw["metadata"])
    if why:
        dump(args.output / "results.json", dict(selected="incumbent", fallback=why,
            complete_grid=False, official_test_accessed=False, predictive_defaults_changed=False))
        return
    proxy, production = native_audit(args, raw, policy)
    from scripts import benchmark_latest_completed_peers_20261020 as latest
    from scripts import benchmark_association_state_20261020 as association
    from scripts import benchmark_source_response_annotation_20261020 as annotation
    from scripts import qualify_bounded_view_consensus_20261020 as consensus
    previous_source = latest.ValidationOnlySource
    latest.ValidationOnlySource = validation_source
    domain_fallback = None
    try:
        for name, runner in (("latest", latest), ("association", association), ("annotation", annotation)):
            runner.run(argparse.Namespace(task_root=proxy, source_root=args.source_root, production=production,
                output=args.output / name, workers=args.workers))
    except ChallengerUnavailable as error:
        domain_fallback = str(error)
    except ValueError as error:
        if str(error) not in {"No nonstructural event-context payload",
                "No reusable event context after bounded profiling", "Insufficient past ordinary response supervision"}:
            raise
        domain_fallback = str(error)
    finally:
        latest.ValidationOnlySource = previous_source
    if domain_fallback is not None:
        dump(args.output / "results.json", dict(selected="incumbent", fallback="unsupported_bounded_view_domain",
            detail=domain_fallback, complete_grid=False, official_test_accessed=False,
            predictive_defaults_changed=False))
        return
    consensus.run(argparse.Namespace(component=[args.output / name for name in ("latest", "association", "annotation")],
        output=args.output / "consensus"))
    report = json.loads((args.output / "consensus/results.json").read_text())
    dump(args.output / "results.json", dict(report, original_task_root=str(args.task_root),
        current_run_only=True, native_parameters=json.loads((production / "manifest.json").read_text())["model_params"],
        predictive_defaults_changed=False, elapsed_seconds=time.monotonic()-began))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root", "source-root", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--run-root", type=Path, help="Require a completed fresh pipeline and verify its complete source identity")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if not 1 <= args.workers <= 3:
        p.error("One to three case workers required")
    run(args)
