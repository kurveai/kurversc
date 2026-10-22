"""Production-matched, latest-per-root bounded peer-history challenger.

Append at most 50 values on the same two schema-derived interaction paths.
Every query first admits known outcomes, keeps the latest per historical root,
then applies the existing 50K bound. No native column or default is changed.
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

import kurversc
from kurversc.core import _normalize_tables
from kurversc.temporal_policy import TemporalFitPolicy, _score
from scripts.benchmark_large_gap_challengers import fit_prediction
from scripts.benchmark_recovered_readouts import matched_gate, phase_frames
from scripts.benchmark_sequence_recovery_20261020 import verify
from scripts.benchmark_signal_challengers import prepared
from scripts.engagement_local_comparison import freeze
from scripts.latest_completed_peers_20261020 import LatestCompletedPeers
from scripts.materialize_validation_origins import ValidationOnlySource
from scripts.run_gap_followup_queue import digest, dump


WEIGHTS = (.125, .5, 1.)
RECIPES = tuple(f"latest_completed_peers_{w:g}" for w in WEIGHTS)


def prepare(args, raw, manifest):
    meta = raw["metadata"]
    source = ValidationOnlySource(meta["dataset"], meta["task"])
    inner, task = source.inner_split(), source.task
    problem = kurversc.relbench_problem_from_objects(task, inner.db_state, inner.train_table, inner.eval_table,
        dataset_name=meta["dataset"], task_name=meta["task"], sample_rows=1000, search_training_frames=1,
        max_train_timestamps=3, schema_depth=4, max_base_table_occurrences=1, include_target_history=False, random_state=0)
    root, tables = _normalize_tables(problem.parent_node, problem.tables)
    availability = meta.get("label_available_at")
    columns = ["__event_entity", "__event_cutoff", meta["target"], *([availability] if availability else [])]
    history = pd.concat([pd.read_parquet(args.task_root / "cache" / n, columns=columns)
        for n in ("history.parquet", "validation.parquet")], ignore_index=True).rename(columns={meta["target"]: "__outcome"})
    if len(history) > 500_000:
        raise ValueError("Bounded latest-peer qualification requires <=500K complete label metadata rows")
    history["__available_at"] = (pd.to_datetime(history[availability]) if availability
        else pd.to_datetime(history.__event_cutoff)+pd.Timedelta(meta["label_horizon"]))
    policy = TemporalFitPolicy(**manifest["policy"])
    for fold in reversed(manifest["folds"]):
        train, held, phase = phase_frames(args.task_root, raw, policy, fold)
        print(json.dumps(dict(phase="latest_peer_memberships", fold=fold)), flush=True)
        provider = LatestCompletedPeers().fit(history, tables=tables, relationships=problem.relationships,
            root_name=root, phase_boundary=phase["fit_origin"])
        extra = [provider.transform(frame) for frame in (train, held)]
        if list(extra[0]) != list(extra[1]) or not 0 < len(extra[0].columns) <= 50:
            raise ValueError("Latest-peer schema exceeds its fixed budget")
        folder = args.output / f"fold{fold}"
        folder.mkdir(exist_ok=True)
        hashes = {}
        for name, values in zip(("train", "validation"), extra):
            path = folder / f"{name}-extra.parquet"
            values.to_parquet(path, index=False)
            hashes[str(path)] = digest(path)
        freeze(folder / "producer.json", dict(fold=fold, phase=phase, provider=provider.audit,
            snapshots=provider.snapshot_audit, features_sha256=hashes,
            native_columns_unchanged=True, official_test_accessed=False))
        del train, held, extra, provider
        gc.collect()


def case(task_root, production, output, fold, seed):
    task_root, production, output = map(Path, (task_root, production, output))
    m = json.loads((output / "manifest.json").read_text())
    raw = json.loads((task_root / "raw-audit/manifest.json").read_text())
    raw = {**raw, "model_params": m["model_params"]}
    meta, policy = raw["metadata"], TemporalFitPolicy(**m["policy"])
    train, held, phase = phase_frames(task_root, raw, policy, fold)
    x, vx = prepared(train, meta["feature_schema"]), prepared(held, meta["feature_schema"])
    with np.load(production / f"fold{fold}/seed{seed}.npz") as a:
        np.testing.assert_array_equal(a["truth"], held[meta["target"]])
        np.testing.assert_array_equal(a["groups"], held.__event_entity)
        control = a["control"].copy()
    kwargs = dict(raw=raw, meta=meta, policy=policy, phase=phase, seed=seed)
    native, _ = fit_prediction(x, train[meta["target"]], pd.to_datetime(train.__event_cutoff), vx, **kwargs)
    np.testing.assert_array_equal(native, control)
    producer = json.loads((output / f"fold{fold}/producer.json").read_text())
    verify(producer["features_sha256"])
    extras = [pd.read_parquet(output / f"fold{fold}/{n}-extra.parquet") for n in ("train", "validation")]
    if set(x) & set(extras[0]) or len(extras[0]) != len(x) or len(extras[1]) != len(vx):
        raise ValueError("Phase-aligned new peer features must not overwrite native columns")
    xx, vxx = [pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1) for a, b in ((x, extras[0]), (vx, extras[1]))]
    head, _ = fit_prediction(xx, train[meta["target"]], pd.to_datetime(train.__event_cutoff), vxx, **kwargs)
    if fold == max(m["folds"]) and seed == 0:
        again, _ = fit_prediction(xx, train[meta["target"]], pd.to_datetime(train.__event_cutoff), vxx, **kwargs)
        np.testing.assert_array_equal(head, again)
    arrays = dict(truth=held[meta["target"]].to_numpy(), groups=held.__event_entity.to_numpy(), control=control, head=head,
        **{n: (1-w)*control+w*head for n, w in zip(RECIPES, WEIGHTS)})
    path = output / f"fold{fold}/seed{seed}.npz"
    np.savez_compressed(path, **arrays)
    row = dict(fold=fold, seed=seed, origin=phase["origin"], native_exact=True,
        scores={n: _score(arrays["truth"], arrays[n], "classification") for n in ("control", *RECIPES)},
        vectors_sha256=digest(path), manifest_sha256=digest(output / "manifest.json"))
    dump(path.with_suffix(".json"), row)
    return row


def run(args):
    began = time.monotonic()
    raw = json.loads((args.task_root / "raw-audit/manifest.json").read_text())
    production = json.loads((args.production / "manifest.json").read_text())
    results = json.loads((args.production / "results.json").read_text())
    if (not results.get("complete_grid") or results.get("official_test_accessed") is not False
            or production["metadata"] != raw["metadata"] or raw["metadata"]["metric"] != "roc_auc"):
        raise ValueError("Complete authenticated production-native binary evidence required")
    verify(production["sources"])
    verify(production["inputs"])
    files = [args.task_root / "raw-audit/manifest.json", *[args.production / n for n in ("manifest.json", "results.json")],
        *sorted((args.task_root / "cache").glob("*.parquet")),
        *sorted((args.task_root / "snapshots").glob("fold*/*")), *sorted((args.source_root / "db").glob("*.parquet"))]
    for row in results["records"]:
        path = args.production / f"fold{row['fold']}/seed{row['seed']}.npz"
        if row["vectors_sha256"] != digest(path) or row["manifest_sha256"] != digest(args.production / "manifest.json"):
            raise ValueError("Production vectors changed")
        files.append(path)
    sources = {Path(__file__), *[Path(sys.modules[f.__module__].__file__) for f in
        (LatestCompletedPeers, fit_prediction, phase_frames, prepared, ValidationOnlySource)]}
    identity = dict(profile="latest_before_sample_completed_peers_v1", metadata=raw["metadata"], policy=production["policy"],
        model_params=production["model_params"], folds=production["folds"], seeds=[0, 1, 2], recipes=RECIPES,
        allowance=1024, maximum_new_columns=50, maximum_peer_rows=50_000, maximum_new_graphs=0,
        sources={str(p): digest(p) for p in sources}, inputs={str(p): digest(p) for p in files}, official_test_accessed=False)
    args.output.mkdir(parents=True, exist_ok=True)
    freeze(args.output / "manifest.json", identity)
    prepare(args, raw, identity)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as pool:
        jobs = [pool.submit(case, str(args.task_root), str(args.production), str(args.output), f, s)
            for f in reversed(identity["folds"]) for s in range(3)]
        for job in as_completed(jobs):
            rows.append(job.result())
            dump(args.output / "progress.json", dict(cases=len(rows), required_cases=len(jobs)))
    rows.sort(key=lambda r: (r["fold"], r["seed"]))
    paired = []
    for row in rows:
        with np.load(args.output / f"fold{row['fold']}/seed{row['seed']}.npz") as a:
            paired.append(dict(origin=row["origin"], seed=row["seed"], truth=a["truth"].copy(), groups=a["groups"].copy(),
                control=a["control"].copy(), vectors={n: a[n].copy() for n in RECIPES}))
    gate = matched_gate(paired, metric="roc_auc", recipes=RECIPES, allowance=1024)
    verify(identity["sources"])
    verify(identity["inputs"])
    dump(args.output / "results.json", dict(selected=gate["selected"], gate=gate, records=rows,
        complete_grid=True, official_test_accessed=False, default_changed=False, elapsed_seconds=time.monotonic()-began))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("task-root", "source-root", "production", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.workers <= 3:
        p.error("One to three independent case workers required")
    run(args)
