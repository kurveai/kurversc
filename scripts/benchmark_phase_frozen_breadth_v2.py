"""Matched phase-frozen object-breadth evaluation, including multi-cutoff holdouts."""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil

from kurversc.temporal_policy import TemporalFitPolicy, fit_temporal_policy, policy_support, _score
from scripts.activity_timeline_challenger import discover_channels, table_metadata
from scripts.benchmark_signal_challengers import dump, incumbent_evidence, prepared, values
from scripts.phase_frozen_object_breadth import PhaseFrozenObjectBreadth as ObjectBreadth
from scripts.select_activity_readout import cache_locations, digest


def row_digest(frame, columns):
    return hashlib.sha256(pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes()).hexdigest()


def task_root_table(metadata, source_meta):
    """Resolve RelBench's task-parquet placeholder using the declared task.

    Some task parquet metadata points to 'entity_table', not the physical DB
    table. Never guess a relation from task names or key values.
    """
    entity, root = next(iter(source_meta["fkey_col_to_pkey_table"].items()))
    declared = metadata.get("entity_table")
    if declared:
        if root not in {"entity_table",declared}:
            raise ValueError("Task context and source relation disagree about the predictive root")
        return declared
    if root == "entity_table":
        from relbench.tasks import get_task
        task = get_task(metadata["dataset"], metadata["task"], download=False)
        if task.entity_col != entity:
            raise ValueError("Task declaration and parquet root key differ")
        root = task.entity_table
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root", "source-root", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--folds", default="all")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--admission-gib", type=float, default=10)
    args = p.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.task_root/"selection/manifest.json").read_text())["metadata"]
    if not metadata["atomic_date_replay"] or metadata["fold"] != "current":
        raise ValueError("Corrected point-in-time current graph required")
    selected = TemporalFitPolicy(**json.loads((args.task_root/"selection/results.json").read_text())["selected_policy"])
    task = "classification" if metadata["metric"] == "roc_auc" else "regression"
    target, schema = metadata["target"], metadata["feature_schema"]
    source_meta = table_metadata(args.source_root/"tasks"/metadata["task"]/"train.parquet")
    foreign = source_meta["fkey_col_to_pkey_table"]
    if len(foreign) != 1:
        raise ValueError("One declared root entity required")
    root = task_root_table(metadata, source_meta)
    channels = discover_channels(args.source_root/"db", root)
    cache, snapshots = cache_locations(args.task_root)
    folds = sorted([int(p.parent.name[4:]) for p in snapshots.glob("fold*/manifest.json")], reverse=True)
    if args.folds != "all":
        folds = list(map(int, args.folds.split(",")))
    raw_params = json.loads((args.task_root / "raw-audit/manifest.json").read_text()).get("model_params", {})
    actual_params = {**raw_params}
    actual_params.setdefault("thread_count", args.threads)
    manifest = dict(metadata=metadata, policy=asdict(selected), model_params=actual_params,
        row_limit=100000, horizon=metadata["label_horizon"], history_days=730,
        channels=[dict(path=str(c.path), entity=c.entity, timestamp=c.timestamp) for c in channels],
        recipes={"object_breadth_full": 1., "object_breadth_blend": .5},
        sources={str(path): digest(path) for path in (Path(__file__),
            Path(__file__).with_name("object_breadth_challenger.py"),
            Path(__file__).with_name("phase_frozen_object_breadth.py"), Path("kurversc/temporal_policy.py"),
            Path("kurversc/modeling.py"), Path(__file__).with_name("benchmark_signal_challengers.py"))},
        scope="Inner validation only; same frozen graph/support/sampling/learner; no official or default changes")
    mp = args.output/"manifest.json"
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError("Different immutable object-breadth experiment already exists")
    dump(mp, manifest)
    while psutil.virtual_memory().available < args.admission_gib*2**30:
        if time.monotonic()-started > 3*3600:
            raise RuntimeError("Memory admission deadline reached")
        print(json.dumps(dict(phase="waiting_for_memory", available_gib=psutil.virtual_memory().available/2**30)), flush=True)
        time.sleep(10)
    times_frame = pd.read_parquet(cache/"history.parquet", columns=["__event_cutoff"])
    typed_timestamps = pd.api.types.is_datetime64_any_dtype(times_frame["__event_cutoff"])
    row_columns = [target, "__event_entity", "__event_cutoff"]
    columns = [*schema["columns"], *row_columns]
    seeds = list(map(int, args.seeds.split(",")))
    for fold in folds:
        folder = args.output/f"fold{fold}"
        folder.mkdir(exist_ok=True)
        if all((folder/f"seed{s}.json").exists() for s in seeds):
            continue
        snapshot = json.loads((snapshots/f"fold{fold}/manifest.json").read_text())
        origin = snapshot["origin"]
        fit_origin = snapshot["snapshot_cutoff"]
        if pd.Timestamp(fit_origin) > pd.Timestamp(origin) or pd.Timestamp(snapshot["latest_observed_label_complete_at"]) > pd.Timestamp(fit_origin):
            raise ValueError("Future source state or immature labels reached the phase")
        support = policy_support(times_frame, timestamp="__event_cutoff", origin=fit_origin,
            label_horizon=metadata["label_horizon"], refit_frames=metadata["validation_cutoffs"], policy=selected)
        filter_values = list(support["cutoffs"]) if typed_timestamps else list(map(str, support["cutoffs"]))
        train = pd.read_parquet(cache/"history.parquet", columns=columns,
            filters=[("__event_cutoff", "in", filter_values)])
        heldout = pd.read_parquet(snapshots/f"fold{fold}/validation.parquet", columns=columns)
        feature_identity = dict(train_hash=row_digest(train, row_columns), validation_hash=row_digest(heldout, row_columns),
            manifest_sha256=digest(mp), support={k:list(map(str,v)) for k,v in support.items()})
        ap = folder/"feature-audit.json"
        if ap.exists() and json.loads(ap.read_text()) != feature_identity:
            raise ValueError("Feature row identity or frozen implementation changed")
        for name, requests in (("training", train), ("validation", heldout)):
            destination = folder/f"breadth-{name}.parquet"
            if destination.exists():
                continue
            print(json.dumps(dict(phase="features", fold=fold, split=name, rows=len(requests))), flush=True)
            provider = ObjectBreadth(args.source_root/"db", root, horizon=metadata["label_horizon"],
                training_cutoff=pd.to_datetime(train["__event_cutoff"]).min(), temporary=folder/"spill", threads=1)
            dump(folder/"object-plan.json", provider.audit)
            try:
                block = provider.transform(requests, phase_boundary=fit_origin)
            finally:
                provider.close()
            block.to_parquet(destination, index=False)
            del block
        dump(ap, feature_identity)
        x, vx = prepared(train, schema), prepared(heldout, schema)
        y = pd.to_numeric(train[target]).copy()
        times = pd.to_datetime(train["__event_cutoff"]).copy()
        truth, groups = heldout[target].to_numpy(copy=True), heldout["__event_entity"].to_numpy(copy=True)
        del train, heldout
        gc.collect()
        extra = pd.read_parquet(folder/"breadth-training.parquet")
        valid_extra = pd.read_parquet(folder/"breadth-validation.parquet")
        if len(extra) != len(x) or len(valid_extra) != len(vx):
            raise ValueError("Aligned feature blocks required")
        # Unsupervised constancy pruning based on INITIAL training rows only.
        # Never let the held-out rows select columns.
        initial = times.isin(support["initial"])
        payload = [c for c in extra if extra.loc[initial,c].nunique(dropna=False)>1]
        dump(folder/"payload.json", dict(selected_columns=payload, removed_constant=[c for c in extra if c not in payload]))
        categorical = [i for i,c in enumerate(x) if c in schema["categorical_columns"]]
        for seed in seeds:
            rp = folder/f"seed{seed}.json"
            if rp.exists():
                continue
            control = incumbent_evidence(args.task_root/"raw-audit", fold, seed, selected)
            np.testing.assert_array_equal(control["truth"], truth)
            np.testing.assert_array_equal(control["groups"], groups)
            kwargs = dict(origin=fit_origin, label_horizon=metadata["label_horizon"],
                refit_frames=metadata["validation_cutoffs"], policy=selected, task=task,
                categorical=categorical, random_state=seed, row_limit=100000, model_params=manifest["model_params"])
            before = time.monotonic()
            if seed == 0:
                print(json.dumps(dict(phase="control_replay", fold=fold, seed=seed)), flush=True)
                base, base_audit = fit_temporal_policy(x, y, times, **kwargs)
                np.testing.assert_allclose(values(base, vx, task), control["prediction"], rtol=0, atol=1e-9,
                    err_msg="Selected production control must reproduce before judging augmentation")
                dump(folder/"control-replay.json", dict(reproduced=True, audit=base_audit))
                del base
                gc.collect()
            print(json.dumps(dict(phase="fit_challenger", fold=fold, seed=seed, added_columns=len(payload))), flush=True)
            xx, vxx = pd.concat([x,extra[payload]], axis=1), pd.concat([vx,valid_extra[payload]], axis=1)
            model, fit_audit = fit_temporal_policy(xx, y, times, **kwargs)
            prediction = values(model, vxx, task)
            blended = .5*(control["prediction"]+prediction)
            scores = {name: _score(truth, vector, task) for name,vector in
                (("control", control["prediction"]), ("candidate", prediction), ("candidate_blend", blended))}
            direction = 1 if task == "classification" else -1
            report = dict(fold=fold, seed=seed, origin=origin, task_type=task, scores=scores,
                gain=direction*(scores["candidate"]-scores["control"]),
                blended_gain=direction*(scores["candidate_blend"]-scores["control"]),
                manifest_sha256=digest(mp), elapsed_seconds=time.monotonic()-before,
                fit_audit=fit_audit, added_columns=payload)
            np.savez_compressed(folder/f"seed{seed}.npz", truth=truth, groups=groups,
                control=control["prediction"], candidate=prediction, candidate_blend=blended)
            dump(rp, report)
            print(json.dumps({k:v for k,v in report.items() if k not in ("fit_audit", "added_columns")}), flush=True)
            del model, xx, vxx
            gc.collect()
            if fold == folds[0] and seed == 0 and max(report["gain"], report["blended_gain"]) <= 0:
                dump(args.output/"screening.json", dict(selected="incumbent", reason="failed_recent_scout", report=report))
                return
        del x, vx, extra, valid_extra, y, times
        gc.collect()


if __name__ == "__main__":
    main()



