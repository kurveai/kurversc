"""Fixed, validation-only transition ablation against the gated support3 head.

The optimized provider is signal-equivalent to the original channel-set
probe, not a new schema-specific feature. No official targets or defaults are
accessed. Complete row cohorts and identical model budgets are required.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil

from kurversc.temporal_policy import TemporalFitPolicy, fit_temporal_policy, _score
from scripts.activity_timeline_challenger import EventChannel
from scripts.audit_dormant_reactivation import positive_pair_errors
from scripts.benchmark_signal_challengers import dump, incumbent_evidence, prepared
from scripts.bounded_channel_transitions import BoundedChannelTransitions
from scripts.select_activity_readout import cache_locations, digest


def checked_support_hash(frame, target, expected):
    actual = hashlib.sha256(pd.util.hash_pandas_object(
        frame[["__event_entity", "__event_cutoff", target]], index=False).values.tobytes()).hexdigest()
    if actual != expected:
        raise ValueError("Full ordered mature training cohort changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", default="2,1,0")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    support = json.loads((args.support/"manifest.json").read_text())
    if json.loads((args.support/"selection.json").read_text())["decision"]["selected"] != "three_initial":
        raise ValueError("The complete support3 gate must have passed")
    metadata = support["metadata"]
    if json.loads((args.task_root/"selection/manifest.json").read_text())["metadata"] != metadata:
        raise ValueError("Frozen task/graph differs")
    sources = [Path(__file__), Path(__file__).with_name("bounded_channel_transitions.py"),
        Path(__file__).with_name("engagement_hypothesis_features.py"),
        Path(__file__).with_name("activity_timeline_challenger.py"),
        Path("kurversc/temporal_policy.py"), Path("kurversc/modeling.py")]
    manifest = dict(metadata=metadata, model_params=support["model_params"],
        sources={str(p): digest(p) for p in sources},
        support_manifest_sha256=digest(args.support/"manifest.json"),
        support_selection_sha256=digest(args.support/"selection.json"),
        head_blend=.5, initial_frames=3, channels=support["channels"],
        windows_in_horizons=[1,4], max_features=144,
        scope="Full-cohort validation-only transition increment over support3; no official/default choices")
    path = args.output/"manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Different immutable transition experiment exists")
    dump(path, manifest)
    threshold = (6 if args.features_only else 20)*2**30
    while psutil.virtual_memory().available < threshold:
        if time.monotonic()-started > 4*3600:
            raise RuntimeError("Admission deadline reached; completed artifacts retained")
        print(json.dumps(dict(phase="waiting_for_memory", available_gib=psutil.virtual_memory().available/2**30)), flush=True)
        time.sleep(30)
    schema, target = metadata["feature_schema"], metadata["target"]
    channels = [EventChannel(Path(c["path"]), c["entity"], c["timestamp"]) for c in support["channels"]]
    cache, snapshots = cache_locations(args.task_root)
    legacy_policy = TemporalFitPolicy(**json.loads((args.task_root/"selection/results.json").read_text())["selected_policy"])
    seeds = list(map(int, args.seeds.split(",")))
    for fold in map(int, args.folds.split(",")):
        root = args.output/f"fold{fold}"
        root.mkdir(exist_ok=True)
        if not args.features_only and all((root/f"seed{s}.json").exists() for s in seeds):
            continue
        snapshot = json.loads((snapshots/f"fold{fold}/manifest.json").read_text())
        origin = snapshot["origin"]
        if snapshot["snapshot_cutoff"] != origin:
            raise ValueError("Exact phase-frozen boundaries required")
        audit = json.loads((args.support/f"fold{fold}/seed0.json").read_text())["fit_audit"]
        cutoffs = audit["initial_cutoffs"]+audit["observed_refit_cutoffs"]
        columns = [target, "__event_entity", "__event_cutoff"]
        train = pd.read_parquet(cache/"history.parquet", columns=columns, filters=[("__event_cutoff", "in", cutoffs)])
        heldout = pd.read_parquet(snapshots/f"fold{fold}/validation.parquet", columns=columns)
        expected = json.loads((args.support/f"fold{fold}/support.json").read_text())["ordered_train_hash"]
        checked_support_hash(train, target, expected)
        feature_paths = [root/f"transitions-{s}.parquet" for s in ("training", "validation")]
        if not all(p.exists() for p in feature_paths):
            provider = BoundedChannelTransitions(channels, temporary=root/"spill", threads=1)
            before = time.monotonic()
            try:
                for frame,p in zip((train,heldout), feature_paths):
                    print(json.dumps(dict(phase="transition_features", fold=fold, rows=len(frame))), flush=True)
                    provider.transform(frame, phase_boundary=origin, horizon=metadata["label_horizon"]).to_parquet(p, index=False)
            finally:
                provider.close()
            dump(root/"feature-audit.json", dict(elapsed_seconds=time.monotonic()-before,
                features=4*len(channels)**2, ordered_train_hash=expected, manifest_sha256=digest(path)))
        if args.features_only:
            print(json.dumps(dict(phase="features_completed", fold=fold)), flush=True)
            continue
        train = pd.read_parquet(cache/"history.parquet", columns=[*schema["columns"],*columns],
            filters=[("__event_cutoff", "in", cutoffs)])
        heldout = pd.read_parquet(snapshots/f"fold{fold}/validation.parquet", columns=[*schema["columns"],*columns])
        x, vx = prepared(train, schema), prepared(heldout, schema)
        for block in ("activity", "label_state", "recurrence", "transitions"):
            paths = feature_paths if block=="transitions" else [args.support/f"fold{fold}/{block}-{s}.parquet" for s in ("training", "validation")]
            a, b = [pd.read_parquet(p) for p in paths]
            if [len(a),len(b)] != [len(train),len(heldout)]:
                raise ValueError("Ordered covariate cohort differs")
            x,vx = pd.concat([x,a],axis=1),pd.concat([vx,b],axis=1)
            del a,b
        categorical = [i for i,c in enumerate(x) if c in schema["categorical_columns"]]
        for seed in seeds:
            if (root/f"seed{seed}.json").exists():
                continue
            before = time.monotonic()
            print(json.dumps(dict(phase="fit", fold=fold, seed=seed, models=4)), flush=True)
            model,fit_audit = fit_temporal_policy(x, train[target], pd.to_datetime(train.__event_cutoff), origin=origin,
                label_horizon=metadata["label_horizon"],refit_frames=metadata["validation_cutoffs"],
                policy=TemporalFitPolicy("three_initial", training_frames=3), task="classification", categorical=categorical,
                random_state=seed,row_limit=100000,model_params=manifest["model_params"])
            for name in ("initial_cutoffs", "observed_refit_cutoffs", "training_rows"):
                if fit_audit[name] != audit[name]:
                    raise ValueError("Mature support changed")
            old = incumbent_evidence(args.task_root/"raw-audit",fold,seed,legacy_policy)
            with np.load(args.support/f"fold{fold}/seed{seed}.npz") as a:
                truth,groups,control = [a[k].copy() for k in ("truth", "groups", "candidate")]
            np.testing.assert_array_equal(truth, heldout[target])
            np.testing.assert_array_equal(groups, heldout.__event_entity)
            np.testing.assert_array_equal(old["truth"], truth)
            np.testing.assert_array_equal(old["groups"], groups)
            prediction = .5*(old["prediction"]+model.predict_proba(vx)[:,1])
            scores = dict(control=_score(truth,control,"classification"),candidate=_score(truth,prediction,"classification"))
            h = pd.Timedelta(metadata["label_horizon"]).total_seconds()/86400
            dormant = vx.rc_all_recency.ge(h).to_numpy() & (truth==1)
            delta = positive_pair_errors(truth,control)-positive_pair_errors(truth,prediction)
            denominator = np.sum(truth==1)*np.sum(truth==0)
            report = dict(fold=fold,seed=seed,origin=origin,scores=scores,gain=scores["candidate"]-scores["control"],
                fit_audit=fit_audit,elapsed_seconds=time.monotonic()-before,manifest_sha256=digest(path),
                dormant_positive_auc_contribution=float(delta[dormant].sum()/denominator),
                other_positive_auc_contribution=float(delta[~dormant].sum()/denominator))
            np.savez_compressed(root/f"seed{seed}.npz",truth=truth,groups=groups,control=control,candidate=prediction)
            dump(root/f"seed{seed}.json",report)
            print(json.dumps({k:v for k,v in report.items() if k!="fit_audit"}),flush=True)
            del model
            gc.collect()
            if fold==2 and seed==0 and report["gain"] <= 0:
                dump(args.output/"screening.json",dict(selected="incumbent",reason="failed_required_recent_scout",report=report))
                return
        del x,vx,train,heldout
        gc.collect()


if __name__ == "__main__":
    main()
