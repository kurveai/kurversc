"""Isolated, memory-bounded learner ablations against frozen temporal controls.

No graph/default changes and no official test access. Cutoff members are fitted
one at a time on their original sampled rows. The complete incumbent must
replay before a challenger is scored. Stronger support3 controls retain their
original legacy blend and all supplemental feature blocks.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from kurversc.modeling import fit_final_catboost, sample_training_rows
from kurversc.temporal_policy import TemporalFitPolicy, policy_support, _score
from scripts.benchmark_signal_challengers import dump, incumbent_evidence, prepared, values
from scripts.benchmark_support_transitions import checked_support_hash
from scripts.select_activity_readout import cache_locations


MENU_ALLOWANCE = 80
ARMS = ("full_rows", "ordered_boosting")
WEIGHTS = (0.125, 0.5, 1.0)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def positioned_frame(path, columns, positions, width=32):
    """Bound Parquet width/readahead, preserve dtypes and exact row ordering."""
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1 or len(set(positions)) != len(positions) or np.any(positions < 0):
        raise ValueError("Unique nonnegative row positions required")
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Unique nonempty ordered columns required")
    order = np.argsort(positions)
    wanted = positions[order]
    blocks = []
    for start in range(0, len(columns), width):
        names = columns[start:start+width]
        scanner = ds.dataset(path, format="parquet").scanner(columns=names,
            batch_size=2048, batch_readahead=0, fragment_readahead=1, use_threads=False,
            fragment_scan_options=ds.ParquetFragmentScanOptions(pre_buffer=False,
                use_buffered_stream=True, buffer_size=65536))
        pieces, offset = [], 0
        for batch in scanner.to_batches():
            lo, hi = np.searchsorted(wanted, [offset, offset+batch.num_rows])
            if hi > lo:
                pieces.append(batch.take(pa.array(wanted[lo:hi]-offset)))
            offset += batch.num_rows
            if len(wanted) and offset > wanted[-1]:
                break
        if len(wanted) and wanted[-1] >= offset:
            raise ValueError("Requested row outside source")
        schema = scanner.projected_schema
        table = pa.Table.from_batches(pieces, schema=schema)
        if table.num_rows != len(wanted):
            raise ValueError("Incomplete row projection")
        blocks.append(table.to_pandas().iloc[np.argsort(order)].reset_index(drop=True))
    return pd.concat(blocks, axis=1)[columns]


def member_positions(headers, *, cutoff, target, row_limit, task, seed, number):
    positions = np.flatnonzero(pd.to_datetime(headers.__event_cutoff).eq(cutoff).to_numpy())
    routed = pd.DataFrame({"position": positions})
    labels = pd.Series(headers.iloc[positions][target].to_numpy())
    selected, _ = sample_training_rows(routed, labels, limit=row_limit,
        task=task, random_state=seed+number)
    return selected.position.to_numpy()


def changed_parameters(arm, params, row_limit):
    if arm == "control":
        return dict(params), row_limit
    if arm == "full_rows":
        return dict(params), None
    if arm == "ordered_boosting":
        return dict(params, boosting_type="Ordered"), row_limit
    raise ValueError("Undeclared learner challenger")


def load_context(task_root, control_dir=None):
    metadata = json.loads((task_root/"selection/manifest.json").read_text())["metadata"]
    if metadata["fold"] != "current" or not metadata["atomic_date_replay"]:
        raise ValueError("Corrected current point-in-time cache required")
    raw = json.loads((task_root/"raw-audit/manifest.json").read_text())
    legacy = TemporalFitPolicy(**json.loads((task_root/"selection/results.json").read_text())["selected_policy"])
    if control_dir:
        manifest = json.loads((control_dir/"manifest.json").read_text())
        selection = json.loads((control_dir/"selection.json").read_text())
        if manifest["metadata"] != metadata or selection["decision"]["selected"] != "three_initial":
            raise ValueError("Only the exact admitted support3 control is supported")
        policy, params = TemporalFitPolicy("three_initial", training_frames=3), manifest["model_params"]
        row_limit, blend, blocks = 100000, .5, ("activity", "label_state", "recurrence")
    else:
        policy, params = legacy, raw["model_params"]
        row_limit, blend, blocks = raw["row_limit"], 1., ()
        if policy.blend_weight != 1 or policy.mode != "frame_ensemble" or policy.readout != "direct":
            raise ValueError("Use a matched adapter for blended/non-direct incumbents")
    return metadata, raw, legacy, policy, params, row_limit, blend, blocks


def run_case(args):
    args.output.mkdir(parents=True, exist_ok=True)
    before = time.monotonic()
    metadata, raw, legacy, policy, params, row_limit, blend, blocks = load_context(args.task_root, args.control_dir)
    cache, snapshots = cache_locations(args.task_root)
    target, schema = metadata["target"], metadata["feature_schema"]
    kind = "classification" if metadata["metric"] == "roc_auc" else "regression"
    snapshot = json.loads((snapshots/f"fold{args.fold}/manifest.json").read_text())
    origin, boundary = snapshot["origin"], snapshot["snapshot_cutoff"]
    if pd.Timestamp(boundary) > pd.Timestamp(origin) or pd.Timestamp(snapshot["latest_observed_label_complete_at"]) > pd.Timestamp(boundary):
        raise ValueError("Future observations reached the frozen validation phase")
    meta_columns = [target, "__event_entity", "__event_cutoff"]
    headers = pd.read_parquet(cache/"history.parquet", columns=meta_columns)
    held = pd.read_parquet(snapshots/f"fold{args.fold}/validation.parquet", columns=meta_columns)
    support = policy_support(headers, timestamp="__event_cutoff", origin=boundary,
        label_horizon=metadata["label_horizon"], refit_frames=metadata["validation_cutoffs"], policy=policy)
    eligible = np.flatnonzero(pd.to_datetime(headers.__event_cutoff).isin(support["cutoffs"]).to_numpy())
    if (pd.to_datetime(headers.iloc[eligible].__event_cutoff)+pd.Timedelta(metadata["label_horizon"]) > pd.Timestamp(boundary)).any():
        raise ValueError("Immature labels in learner support")
    base = incumbent_evidence(args.task_root/"raw-audit", args.fold, args.seed, legacy)
    if args.control_dir:
        checked_support_hash(headers.iloc[eligible], target,
            json.loads((args.control_dir/f"fold{args.fold}/support.json").read_text())["ordered_train_hash"])
        with np.load(args.control_dir/f"fold{args.fold}/seed{args.seed}.npz") as arrays:
            control = {k:arrays[k].copy() for k in ("truth", "groups")}
            control["prediction"] = arrays["candidate"].copy()
    else:
        control = base
    np.testing.assert_array_equal(control["truth"], held[target])
    np.testing.assert_array_equal(control["groups"], held.__event_entity)
    np.testing.assert_array_equal(base["truth"], held[target])
    np.testing.assert_array_equal(base["groups"], held.__event_entity)
    sources = [Path(__file__), Path("kurversc/modeling.py"), Path("kurversc/temporal_policy.py"),
        Path("scripts/benchmark_signal_challengers.py"), Path("scripts/benchmark_support_transitions.py")]
    identity = dict(metadata=metadata, task_root=str(args.task_root),
        control_dir=str(args.control_dir) if args.control_dir else None,
        policy=asdict(policy), model_params=params, row_limit=row_limit, legacy_blend=blend,
        supplemental_blocks=blocks, arms=ARMS, weights=WEIGHTS, menu_allowance=MENU_ALLOWANCE,
        validation_folds=raw["folds"], seeds=[0,1,2],
        source_sha256={str(p):sha256(p) for p in sources},
        scope="Frozen schema/support/objective/tree budget; only learner row cap or boosting scheme changes; validation only")
    # Normalize tuples as JSON before the immutable comparison.
    identity = json.loads(json.dumps(identity, default=str))
    manifest = args.output/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Immutable experiment changed")
    dump(manifest, identity)
    folder = args.output/f"fold{args.fold}"
    folder.mkdir(exist_ok=True)
    result_path = folder/f"{args.arm}-seed{args.seed}.json"
    if result_path.exists():
        print(json.dumps(dict(phase="reused", path=str(result_path))), flush=True)
        return
    v = positioned_frame(snapshots/f"fold{args.fold}/validation.parquet",
        schema["columns"], np.arange(len(held)))
    vx = prepared(v, schema)
    del v
    for block in blocks:
        vx = pd.concat([vx, pd.read_parquet(args.control_dir/f"fold{args.fold}/{block}-validation.parquet")], axis=1)
    categorical = [i for i,c in enumerate(vx) if c in schema["categorical_columns"]]
    if args.control_dir and list(vx) != json.loads((args.control_dir/f"fold{args.fold}/support.json").read_text())["columns"]:
        raise ValueError("Protected feature ordering changed")
    candidate_params, limit = changed_parameters(args.arm, params, row_limit)
    if args.arm != "control":
        checked = folder/f"control-seed{args.seed}.json"
        if not checked.exists() or not json.loads(checked.read_text())["control_reproduced"]:
            raise ValueError("Exact same-seed incumbent must replay first")
    counts = headers.iloc[eligible].groupby("__event_cutoff").size()
    if args.arm == "full_rows" and (row_limit is None or counts.max() <= row_limit):
        dump(result_path, dict(status="identical", reason="All complete cohorts already fit below cap",
            fold=args.fold, seed=args.seed, origin=origin, arm=args.arm, control_reproduced=False,
            manifest_sha256=sha256(manifest), elapsed_seconds=time.monotonic()-before))
        return
    predictions, rows, member_hashes = [], 0, []
    for number, cutoff in enumerate(support["cutoffs"], start=1):
        selected = member_positions(headers, cutoff=cutoff, target=target, row_limit=limit,
            task=kind, seed=args.seed, number=number)
        print(json.dumps(dict(phase="member_load", fold=args.fold, seed=args.seed, arm=args.arm,
            cutoff=str(cutoff), rows=len(selected))), flush=True)
        tx_raw = positioned_frame(cache/"history.parquet", schema["columns"], selected)
        tx = prepared(tx_raw, schema)
        del tx_raw
        for block in blocks:
            positions = np.searchsorted(eligible, selected)
            path = args.control_dir/f"fold{args.fold}/{block}-training.parquet"
            if pq.ParquetFile(path).metadata.num_rows != len(eligible):
                raise ValueError("Supplemental training cohort differs")
            tx = pd.concat([tx, positioned_frame(path, pq.ParquetFile(path).schema_arrow.names, positions)], axis=1)
        if list(tx) != list(vx):
            raise ValueError("Train/query feature ordering differs")
        ty = headers.iloc[selected][target].reset_index(drop=True)
        model, seconds, _ = fit_final_catboost(tx, ty, task=kind, categorical=categorical,
            random_state=args.seed, model_params=candidate_params)
        predictions.append(values(model, vx, kind))
        rows += len(tx)
        member_hashes.append(dict(cutoff=str(cutoff), rows=len(tx), elapsed_seconds=seconds,
            routing_sha256=hashlib.sha256(pd.util.hash_pandas_object(headers.iloc[selected][meta_columns], index=False).values.tobytes()).hexdigest()))
        print(json.dumps(dict(phase="member_complete", fold=args.fold, seed=args.seed,
            arm=args.arm, **member_hashes[-1])), flush=True)
        del model, tx, ty
        gc.collect()
    candidate = blend*np.mean(predictions, axis=0)+(1-blend)*base["prediction"]
    replayed = args.arm == "control"
    if replayed:
        np.testing.assert_allclose(candidate, control["prediction"], rtol=0, atol=1e-10)
    arrays = dict(truth=control["truth"], groups=control["groups"], control=control["prediction"], candidate=candidate)
    for weight in WEIGHTS:
        arrays[f"blend_{weight:g}"] = (1-weight)*arrays["control"]+weight*candidate
    scores = {name:_score(arrays["truth"], arrays[name], kind) for name in ("control", "candidate", *[f"blend_{w:g}" for w in WEIGHTS])}
    gains = {name:(value-scores["control"])*(1 if kind=="classification" else -1) for name,value in scores.items() if name!="control"}
    np.savez_compressed(result_path.with_suffix(".npz"), **arrays)
    report = dict(status="completed", fold=args.fold, seed=args.seed, origin=origin, arm=args.arm,
        task_type=kind, control_reproduced=replayed, scores=scores, gains=gains,
        fit_rows=rows, members=member_hashes, effective_parameters=candidate_params, row_limit=limit,
        covariate_count=len(vx.columns), manifest_sha256=sha256(manifest),
        elapsed_seconds=time.monotonic()-before)
    dump(result_path, report)
    print(json.dumps({k:v for k,v in report.items() if k!="members"}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-root", type=Path, required=True)
    p.add_argument("--control-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=("control", *ARMS), required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    run_case(p.parse_args())


if __name__ == "__main__":
    main()
