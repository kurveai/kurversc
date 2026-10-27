"""Audit final-fit-faithful temporal policy selection on frozen inner features.

No outer split, masked test rows, official scorer or test scores are accessed.
Cached features are reused; three chronological holdout blocks replay the
deployment refit lifecycle before selecting a bounded downstream challenger.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd

from kurversc.modeling import prepare_prediction_features
from kurversc.temporal_policy import (
    InsufficientPolicySupport,
    bounded_temporal_policies,
    fit_temporal_policy,
    select_temporal_policy,
    _score,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--origins", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--phase-frozen-cache", type=Path)
    parser.add_argument("--prune-vetoed", action="store_true",
        help="Omit only fits whose base and every declared blend already fail an existing selector gate")
    parser.add_argument("--pruning-blend-weights", default="0.125,0.5")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.cache / "materialization.json").read_text())
    if metadata["fold"] != "current" or not metadata["atomic_date_replay"]:
        raise ValueError("Corrected current inner-split materialization required")
    train, validation = [
        pd.read_parquet(args.cache / name)
        for name in ("history.parquet", "validation.parquet")
    ]
    frame = pd.concat([train, validation], ignore_index=True)
    del train, validation
    gc.collect()
    schema = metadata["feature_schema"]
    x = prepare_prediction_features(
        frame,
        columns=list(schema["columns"]),
        categorical_columns=set(schema["categorical_columns"]),
        datetime_columns=set(schema["datetime_columns"]),
    )
    # All model columns are now owned by x. Only routing/target columns are
    # used from the raw frame below; detach those before taking Series views
    # so a target view cannot keep a whole wide numeric block alive.
    frame = frame.loc[:, [metadata["target"], "__event_cutoff", "__event_entity"]].copy()
    gc.collect()
    categorical = [
        i
        for i, name in enumerate(x.columns)
        if name in set(schema["categorical_columns"])
    ]
    y = pd.to_numeric(frame[metadata["target"]])
    times = pd.to_datetime(frame["__event_cutoff"])
    cutoffs = sorted(times.unique())
    holdout_frames = metadata["validation_cutoffs"]
    if holdout_frames > 1 and args.phase_frozen_cache is None:
        raise ValueError("Multi-cutoff phases require --phase-frozen-cache")
    origin_count = min(
        args.origins, (len(cutoffs) - holdout_frames - 1) // holdout_frames
    )
    if args.origins < 2 or origin_count < 2:
        raise ValueError("Insufficient history for disjoint retrospective holdouts")
    folds = []
    for number in reversed(range(origin_count)):
        end = len(cutoffs) - number * holdout_frames
        cuts = cutoffs[end - holdout_frames : end]
        folds.append(
            dict(origin=str(pd.Timestamp(cuts[0])), cutoffs=list(map(str, cuts)))
        )
    task = "classification" if metadata["metric"] == "roc_auc" else "regression"
    policies = bounded_temporal_policies(task)
    manifest = dict(
        cache=str(args.cache),
        metadata=metadata,
        policies=[asdict(p) for p in policies],
        folds=folds,
        observed_refit_frames=holdout_frames,
        seeds=list(map(int, args.seeds.split(","))),
        row_limit=100_000,
        model_params=dict(thread_count=args.threads),
        phase_frozen_cache=str(args.phase_frozen_cache)
        if args.phase_frozen_cache
        else None,
        source_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
            + Path("kurversc/temporal_policy.py").read_bytes()
        ).hexdigest(),
        scope="Frozen inner features; graph selection is not nested, so this is a retrospective downstream-policy audit, not an independent graph benchmark",
    )
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Output contains a different immutable experiment")
    execution_path = args.output / "certified-execution.json"
    vetoes, skipped, unavailable = {}, [], []
    if args.prune_vetoed:
        from scripts.temporal_audit_veto import selector_vetoes, verify_selection_barrier
        from kurversc.temporal_policy import add_incumbent_blends
        weights = tuple(float(w) for w in args.pruning_blend_weights.split(",") if w)
        add_incumbent_blends([], weights=weights)  # Validate the complete readout menu.
        proof = dict(origins=[f["origin"] for f in folds], seeds=manifest["seeds"],
            policies=[p.name for p in policies], task=task, blend_weights=weights)
        selector_vetoes([], **proof)  # Single-seed/subset evidence cannot certify this experiment.
        execution = dict(strategy="existing_selector_veto_recent_first", experiment=proof,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            certificate_source_sha256=hashlib.sha256(
                Path(sys.modules[selector_vetoes.__module__].__file__).read_bytes()).hexdigest(),
            survivor_grid="complete original origins and seeds", learner_budget_changed=False)
        # JSON canonicalization also normalizes tuples for exact resume checks.
        execution = json.loads(json.dumps(execution))
        if execution_path.exists() and json.loads(execution_path.read_text()) != execution:
            raise ValueError("Certified audit execution identity changed")
        execution_path.write_text(json.dumps(execution, indent=2) + "\n")
    elif execution_path.exists():
        raise ValueError("A certified partial audit cannot resume as an exhaustive audit")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    records, evidence = [], []
    for seed in manifest["seeds"]:
        model_cache = {}
        fold_order = reversed(range(len(folds))) if args.prune_vetoed else range(len(folds))
        for fold_number in fold_order:
            # Preserve chronological fold IDs, snapshots and the declared grid.
            # Only execution order changes: the recent deployment seed can
            # deterministically reject a readout before older/other-seed fits.
            fold = folds[fold_number]
            mask = times.isin(pd.to_datetime(fold["cutoffs"]))
            positions = np.flatnonzero(mask.to_numpy())
            vx = x.iloc[positions]
            fit_origin = fold["origin"]
            if args.phase_frozen_cache:
                folder = args.phase_frozen_cache / f"fold{fold_number}"
                snapshot_manifest = json.loads((folder / "manifest.json").read_text())
                if (
                    snapshot_manifest["origin"] != fold["origin"]
                    or snapshot_manifest["checkpoint_sha256"]
                    != metadata["checkpoint_sha256"]
                ):
                    raise ValueError(
                        "Phase-frozen snapshot metadata differs from the fold"
                    )
                if pd.Timestamp(
                    snapshot_manifest["latest_observed_label_complete_at"]
                ) > pd.Timestamp(snapshot_manifest["snapshot_cutoff"]):
                    raise ValueError("Uncompleted labels reached a phase snapshot")
                fit_origin = snapshot_manifest["snapshot_cutoff"]
                if pd.Timestamp(fit_origin) > pd.Timestamp(fold["origin"]):
                    raise ValueError(
                        "Future database events reached the forecast phase"
                    )
                snapshot = pd.read_parquet(folder / "validation.parquet")
                positions = snapshot["__position"].to_numpy(dtype=int, copy=True)
                if not np.isin(positions, np.flatnonzero(mask.to_numpy())).all() or len(
                    np.unique(positions)
                ) != len(positions):
                    raise ValueError("Phase snapshot contains unknown/duplicate rows")
                np.testing.assert_array_equal(
                    snapshot[metadata["target"]], y.iloc[positions]
                )
                vx = prepare_prediction_features(
                    snapshot,
                    columns=list(schema["columns"]),
                    categorical_columns=set(schema["categorical_columns"]),
                    datetime_columns=set(schema["datetime_columns"]),
                )
                del snapshot
                gc.collect()
            truth = y.iloc[positions].to_numpy(dtype=float)
            groups = frame.iloc[positions]["__event_entity"].to_numpy()
            for policy in policies:
                basename = f"fold{fold_number}-{policy.name}-seed{seed}"
                prediction_path = args.output / f"{basename}.npz"
                audit_path = args.output / f"{basename}.json"
                try:
                    if prediction_path.exists() and audit_path.exists():
                        prediction = np.load(prediction_path)["prediction"]
                        record = json.loads(audit_path.read_text())
                        np.testing.assert_array_equal(
                            np.load(prediction_path)["truth"], truth
                        )
                    elif args.prune_vetoed and policy.name in vetoes:
                        record = dict(policy=policy.name, seed=seed, origin=fold["origin"],
                            certified_selector_veto=vetoes[policy.name])
                        skipped.append(record)
                        records.append(record)
                        print(json.dumps(dict(phase="certified_fit_skipped", **record)), flush=True)
                        continue
                    else:
                        model, audit = fit_temporal_policy(
                            x,
                            y,
                            times,
                            origin=fit_origin,
                            label_horizon=metadata["label_horizon"],
                            refit_frames=holdout_frames,
                            policy=policy,
                            task=task,
                            categorical=categorical,
                            random_state=seed,
                            model_params=manifest["model_params"],
                            frame_model_cache=model_cache,
                        )
                        prediction = np.asarray(
                            model.predict_proba(vx)[:, 1]
                            if task == "classification"
                            else model.predict(vx)
                        ).reshape(-1)
                        record = dict(
                            policy=policy.name,
                            seed=seed,
                            origin=fold["origin"],
                            score=_score(truth, prediction, task),
                            audit=audit,
                        )
                        np.savez_compressed(
                            prediction_path,
                            truth=truth,
                            prediction=prediction,
                            groups=groups,
                        )
                        audit_path.write_text(json.dumps(record, indent=2) + "\n")
                        del model
                    evidence.append(
                        dict(
                            origin=fold["origin"],
                            seed=seed,
                            policy=policy.name,
                            truth=truth,
                            prediction=prediction,
                            groups=groups,
                        )
                    )
                    if (args.prune_vetoed and policy.name != "incumbent"
                            and policy.name not in vetoes
                            and (seed == 0 or fold_number == len(folds)-1)):
                        # The original fitter has already prepared and authenticated
                        # these exact rows. Reuse its in-memory evidence instead of
                        # repeatedly decoding every saved prediction archive.
                        vetoes = selector_vetoes(evidence, **proof)
                except InsufficientPolicySupport as exc:
                    record = dict(
                        policy=policy.name,
                        seed=seed,
                        origin=fold["origin"],
                        unavailable=str(exc),
                    )
                    unavailable.append((fold["origin"], seed, policy.name))
                records.append(record)
                print(json.dumps(dict(phase="policy_fold", **record)), flush=True)
                gc.collect()
        del model_cache
        gc.collect()
    if args.prune_vetoed:
        vetoes = verify_selection_barrier(evidence, **proof, unavailable=unavailable)
        certificates = dict(vetoes=vetoes, skipped=skipped, complete_survivor_grid=True,
            original_menu_preserved=True, official_targets_used=False, execution=execution)
        (args.output / "certified-skips.json").write_text(json.dumps(certificates, indent=2) + "\n")
    decision = select_temporal_policy(evidence, task=task)
    selected = next(p for p in policies if p.name == decision["selected"])
    result = dict(
        dataset=metadata["dataset"],
        task=metadata["task"],
        metric=metadata["metric"],
        decision=decision,
        selected_policy=asdict(selected),
        records=records,
        elapsed_seconds=time.perf_counter() - started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        scope=manifest["scope"],
    )
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(phase="complete", decision=decision)), flush=True)


if __name__ == "__main__":
    main()
