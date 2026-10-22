"""Matched learner-row-budget ablation on a completed frozen temporal audit.

Only the pooled learner row ceiling changes. Graph, initial/refit cutoffs,
features, objective, tree budget, and chronological validation cohorts stay
fixed. No outer/test dataset or scorer is imported. Independent fit processes
load only their required cohorts, and selection waits for every fit.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import gc
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd

from kurversc.modeling import prepare_prediction_features
from kurversc.temporal_policy import (
    TemporalFitPolicy,
    _score,
    fit_temporal_policy,
    policy_support,
    select_temporal_policy,
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def parallelism(
    *,
    jobs,
    requested,
    cpus,
    available_gib,
    threads,
    reserve_cpus=24,
    reserve_gib=256,
    worker_gib=48,
):
    """Admission estimate, not a guaranteed worker peak; live guard is separate."""
    if min(jobs, requested, cpus, threads, worker_gib) <= 0:
        raise ValueError("Positive execution budgets required")
    cpu_workers = (cpus - reserve_cpus) // threads
    memory_workers = int((available_gib - reserve_gib - 16) // worker_gib)
    if min(cpu_workers, memory_workers) < 1:
        raise ValueError("Insufficient resources after reserving node headroom")
    return min(jobs, requested, cpu_workers, memory_workers)


def canonical_jobs(folds, seeds, budgets):
    return [
        (fold, seed, budget)
        for fold in range(folds)
        for seed in seeds
        for budget in budgets
    ]


def verify_reference(prediction, truth, groups, reference):
    with np.load(reference) as saved:
        np.testing.assert_array_equal(saved["truth"], truth)
        np.testing.assert_array_equal(saved["groups"], groups)
        # Thread allocation can change floating-point reduction order only.
        np.testing.assert_allclose(saved["prediction"], prediction, rtol=0, atol=1e-10)


def stage_fold(root, audit, headers, fold_number, policy):
    metadata, fold = audit["metadata"], audit["folds"][fold_number]
    snapshot_root = audit.get("phase_frozen_cache")
    if not snapshot_root:
        raise ValueError("An audited phase-frozen validation snapshot is required")
    snapshot = Path(snapshot_root) / f"fold{fold_number}"
    boundary = json.loads((snapshot / "manifest.json").read_text())
    if (
        boundary["origin"] != fold["origin"]
        or boundary["checkpoint_sha256"] != metadata["checkpoint_sha256"]
    ):
        raise ValueError("Snapshot identity differs from the original audit")
    fit_origin = boundary["snapshot_cutoff"]
    if pd.Timestamp(fit_origin) > pd.Timestamp(fold["origin"]) or pd.Timestamp(
        boundary["latest_observed_label_complete_at"]
    ) > pd.Timestamp(fit_origin):
        raise ValueError("Future observations reached the validation snapshot")
    support = policy_support(
        headers,
        timestamp="__event_cutoff",
        origin=fit_origin,
        label_horizon=metadata["label_horizon"],
        refit_frames=audit["observed_refit_frames"],
        policy=policy,
    )
    mask = pd.to_datetime(headers["__event_cutoff"]).isin(support["cutoffs"])
    # Preserve the original stored timestamp values and row ordering.
    cutoff_values = headers.loc[mask, "__event_cutoff"].drop_duplicates().tolist()
    schema = metadata["feature_schema"]
    columns = list(
        dict.fromkeys(
            [
                *schema["columns"],
                metadata["target"],
                "__event_cutoff",
                "__event_entity",
                *([metadata["label_available_at"]] if metadata.get("label_available_at") else []),
            ]
        )
    )
    train = pd.concat(
        [
            pd.read_parquet(
                Path(audit["cache"]) / name,
                columns=columns,
                filters=[("__event_cutoff", "in", cutoff_values)],
            )
            for name in ("history.parquet", "validation.parquet")
        ],
        ignore_index=True,
    )
    np.testing.assert_array_equal(
        train[[metadata["target"], "__event_cutoff", "__event_entity"]].to_numpy(),
        headers.loc[
            mask, [metadata["target"], "__event_cutoff", "__event_entity"]
        ].to_numpy(),
    )
    query = pd.read_parquet(snapshot / "validation.parquet")
    positions = query["__position"].to_numpy(dtype=int)
    expected = np.flatnonzero(
        pd.to_datetime(headers["__event_cutoff"]).isin(pd.to_datetime(fold["cutoffs"]))
    )
    if not np.isin(positions, expected).all() or len(np.unique(positions)) != len(
        positions
    ):
        raise ValueError("Unknown or duplicate phase validation rows")
    np.testing.assert_array_equal(
        query[metadata["target"]], headers.iloc[positions][metadata["target"]]
    )
    query["__event_entity"] = headers.iloc[positions]["__event_entity"].to_numpy()
    query["__event_cutoff"] = headers.iloc[positions]["__event_cutoff"].to_numpy()
    folder = root / "inputs" / f"fold{fold_number}"
    folder.mkdir(parents=True, exist_ok=True)
    train.to_parquet(folder / "train.parquet", index=False)
    query[columns].to_parquet(folder / "validation.parquet", index=False)
    phase = dict(
        origin=fold["origin"],
        fit_origin=fit_origin,
        initial_cutoffs=list(map(str, support["initial"])),
        observed_refit_cutoffs=list(map(str, support["refit"])),
        eligible_training_rows=len(train),
        validation_rows=len(query),
    )
    write_json(folder / "phase.json", phase)
    del train, query
    gc.collect()
    return phase


def configure_worker():
    import pyarrow as pa

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def fit_worker(root, manifest, job):
    fold, seed, budget = job
    root = Path(root)
    basename = root / "fits" / f"fold{fold}-rows{budget}-seed{seed}"
    record_path, prediction_path = (
        basename.with_suffix(".json"),
        basename.with_suffix(".npz"),
    )
    if record_path.exists() and prediction_path.exists():
        record = json.loads(record_path.read_text())
        if record["experiment_sha256"] != manifest["experiment_sha256"]:
            raise ValueError("Fit checkpoint belongs to a different experiment")
        return record
    started = time.perf_counter()
    metadata, schema = manifest["metadata"], manifest["metadata"]["feature_schema"]
    folder = root / "inputs" / f"fold{fold}"
    train, query = [
        pd.read_parquet(folder / name)
        for name in ("train.parquet", "validation.parquet")
    ]
    kwargs = dict(
        columns=schema["columns"],
        categorical_columns=set(schema["categorical_columns"]),
        datetime_columns=set(schema["datetime_columns"]),
    )
    x, vx = (
        prepare_prediction_features(train, **kwargs),
        prepare_prediction_features(query, **kwargs),
    )
    y, times = train[metadata["target"]], pd.to_datetime(train["__event_cutoff"])
    truth, groups = (
        query[metadata["target"]].to_numpy(),
        query["__event_entity"].to_numpy(),
    )
    del train, query
    gc.collect()
    phase = manifest["phases"][fold]
    policy = TemporalFitPolicy(**manifest["fixed_policy"])
    kind = "classification" if metadata["metric"] == "roc_auc" else "regression"
    categorical = [
        i for i, name in enumerate(x) if name in schema["categorical_columns"]
    ]
    model, audit = fit_temporal_policy(
        x,
        y,
        times,
        origin=phase["fit_origin"],
        label_horizon=metadata["label_horizon"],
        refit_frames=manifest["refit_frames"],
        policy=policy,
        task=kind,
        categorical=categorical,
        random_state=seed,
        row_limit=budget,
        model_params=manifest["model_params"],
    )
    if (
        audit["initial_cutoffs"] != phase["initial_cutoffs"]
        or audit["observed_refit_cutoffs"] != phase["observed_refit_cutoffs"]
    ):
        raise ValueError("Changing row budget changed the temporal support")
    prediction = np.asarray(
        model.predict_proba(vx)[:, 1] if kind == "classification" else model.predict(vx)
    ).reshape(-1)
    if budget == manifest["baseline_budget"]:
        reference = (
            Path(manifest["source_audit"]) / f"fold{fold}-{policy.name}-seed{seed}.npz"
        )
        verify_reference(prediction, truth, groups, reference)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gib = peak / (1024**3 if sys.platform == "darwin" else 1024**2)
    record = dict(
        fold=fold,
        origin=phase["origin"],
        seed=seed,
        row_budget=budget,
        score=_score(truth, prediction, kind),
        audit=audit,
        control_reproduced=budget == manifest["baseline_budget"],
        elapsed_seconds=time.perf_counter() - started,
        peak_rss_gib=peak_gib,
        experiment_sha256=manifest["experiment_sha256"],
    )
    temporary = prediction_path.with_name(prediction_path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, truth=truth, prediction=prediction, groups=groups)
    temporary.replace(prediction_path)
    write_json(record_path, record)
    return record


def run(args):
    import psutil

    started = time.perf_counter()
    audit_path = args.source_audit / "manifest.json"
    audit = json.loads(audit_path.read_text())
    metadata = audit["metadata"]
    if metadata["fold"] != "current" or not metadata["atomic_date_replay"]:
        raise ValueError("Corrected current inner-split feature cache required")
    policy = next(
        TemporalFitPolicy(**p) for p in audit["policies"] if p["name"] == args.policy
    )
    if (
        policy.mode != "pooled"
        or policy.readout != "direct"
        or policy.blend_weight != 1
    ):
        raise ValueError(
            "This test requires one fixed, unblended pooled direct readout"
        )
    budgets = tuple(map(int, args.budgets.split(",")))
    seeds = audit["seeds"]
    if (
        len(set(budgets)) != len(budgets)
        or budgets != tuple(sorted(budgets))
        or len(budgets) < 2
        or min(budgets) != audit["row_limit"]
    ):
        raise ValueError(
            "Distinct increasing budgets must start at the original row limit"
        )
    source_hashes = {str(audit_path): file_sha256(audit_path)}
    for name in ("materialization.json", "history.parquet", "validation.parquet"):
        path = Path(audit["cache"]) / name
        source_hashes[str(path)] = file_sha256(path)
    for fold in range(len(audit["folds"])):
        for name in ("manifest.json", "validation.parquet"):
            path = Path(audit["phase_frozen_cache"]) / f"fold{fold}" / name
            source_hashes[str(path)] = file_sha256(path)
        for seed in seeds:
            for suffix in ("json", "npz"):
                path = (
                    args.source_audit / f"fold{fold}-{policy.name}-seed{seed}.{suffix}"
                )
                source_hashes[str(path)] = file_sha256(path)
    params = dict(audit["model_params"], thread_count=args.threads)
    code = {
        name: file_sha256(name)
        for name in (
            __file__,
            "kurversc/temporal_policy.py",
            "kurversc/modeling.py",
            "kurversc/core.py",
            "kurversc/training_support.py",
        )
    }
    recipe = dict(
        source_audit=str(args.source_audit),
        source_sha256=source_hashes,
        metadata=metadata,
        folds=audit["folds"],
        fixed_policy=asdict(policy),
        seeds=seeds,
        budgets=budgets,
        baseline_budget=min(budgets),
        refit_frames=audit["observed_refit_frames"],
        model_params=params,
        code_sha256=code,
        versions={
            name: version(name) for name in ("catboost", "numpy", "pandas", "pyarrow")
        },
        scope="Retrospective inner validation only; original graph selection is not nested. Only pooled learner row budget changes. No official test access.",
    )
    identity = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["experiment_sha256"] != identity:
            raise ValueError("Cannot resume an experiment with changed inputs or code")
    else:
        headers = pd.concat(
            [
                pd.read_parquet(
                    Path(audit["cache"]) / name,
                    columns=[metadata["target"], "__event_cutoff", "__event_entity"],
                )
                for name in ("history.parquet", "validation.parquet")
            ],
            ignore_index=True,
        )
        phases = []
        for fold in range(len(audit["folds"])):
            phase = stage_fold(args.output, audit, headers, fold, policy)
            phases.append(phase)
            print(json.dumps(dict(event="fold_staged", fold=fold, **phase)), flush=True)
        manifest = dict(**recipe, experiment_sha256=identity, phases=phases)
        write_json(manifest_path, manifest)
        del headers
        gc.collect()
    jobs = canonical_jobs(len(audit["folds"]), seeds, budgets)
    workers = parallelism(
        jobs=len(jobs),
        requested=args.workers,
        cpus=os.cpu_count() or 1,
        available_gib=psutil.virtual_memory().available / 1024**3,
        threads=args.threads,
        reserve_cpus=args.reserve_cpus,
        reserve_gib=args.reserve_gib,
        worker_gib=args.estimated_worker_gib,
    )
    execution = dict(
        workers=workers,
        threads_per_fit=args.threads,
        maximum_fit_threads=workers * args.threads,
        estimated_worker_gib=args.estimated_worker_gib,
        reserved_node_gib=args.reserve_gib,
        jobs=len(jobs),
    )
    write_json(args.output / "execution.json", execution)
    print(json.dumps(dict(event="parallel_fits_start", **execution)), flush=True)
    (args.output / "fits").mkdir(exist_ok=True)
    completed = {}
    # Larger fits first; collection/selection below always uses canonical order.
    scheduled = sorted(jobs, key=lambda j: (-j[2], -j[0], j[1]))
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=configure_worker,
        max_tasks_per_child=1,
    ) as pool:
        futures = {
            pool.submit(fit_worker, str(args.output), manifest, job): job
            for job in scheduled
        }
        for future in as_completed(futures):
            job = futures[future]
            record = future.result()
            completed[job] = record
            print(
                json.dumps(
                    dict(
                        event="fit_complete",
                        completed=len(completed),
                        total=len(jobs),
                        **record,
                    )
                ),
                flush=True,
            )
            write_json(
                args.output / "progress.json",
                dict(
                    stage="fitting",
                    completed=len(completed),
                    total=len(jobs),
                    elapsed_seconds=time.perf_counter() - started,
                ),
            )
    records = [completed[job] for job in jobs]
    summary = dict(
        dataset=metadata["dataset"],
        task=metadata["task"],
        metric=metadata["metric"],
        records=records,
        **execution,
        elapsed_seconds=time.perf_counter() - started,
        scope=recipe["scope"],
    )
    write_json(args.output / "fit-summary.json", summary)
    evidence = []
    for fold, seed, budget in jobs:
        path = args.output / "fits" / f"fold{fold}-rows{budget}-seed{seed}.npz"
        with np.load(path) as data:
            evidence.append(
                dict(
                    origin=audit["folds"][fold]["origin"],
                    seed=seed,
                    policy=f"rows{budget}",
                    **{
                        key: data[key].copy()
                        for key in ("truth", "prediction", "groups")
                    },
                )
            )
    write_json(
        args.output / "progress.json",
        dict(
            stage="paired_selection",
            completed=len(jobs),
            total=len(jobs),
            elapsed_seconds=time.perf_counter() - started,
        ),
    )
    print(json.dumps(dict(event="all_fits_complete", completed=len(jobs))), flush=True)
    decision = select_temporal_policy(
        evidence,
        task="classification" if metadata["metric"] == "roc_auc" else "regression",
        incumbent=f"rows{min(budgets)}",
    )
    write_json(
        args.output / "results.json",
        dict(
            summary,
            decision=decision,
            elapsed_seconds=time.perf_counter() - started,
        ),
    )
    write_json(
        args.output / "progress.json",
        dict(stage="complete", completed=len(jobs), total=len(jobs)),
    )
    print(json.dumps(dict(event="complete", decision=decision)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="floor1000")
    parser.add_argument("--budgets", default="100000,200000,400000")
    parser.add_argument("--workers", type=int, default=27)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reserve-cpus", type=int, default=24)
    parser.add_argument("--reserve-gib", type=float, default=256)
    parser.add_argument("--estimated-worker-gib", type=float, default=48)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
