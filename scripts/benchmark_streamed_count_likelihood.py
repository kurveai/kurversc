"""Same-graph count likelihood with one full sampled cohort in memory at once.

This preserves the original frame ensemble, per-frame sampling, mature support
and feature order. It compares a count-likelihood median with BOTH freshly fit
and archived integer-projected controls. Failed scouts reject, never admit.
"""

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from kurversc.modeling import fit_final_catboost, sample_training_rows
from kurversc.temporal_policy import TemporalFitPolicy, _score
from scripts.benchmark_recovered_readouts import matched_gate, phase_frames
from scripts.benchmark_signal_challengers import incumbent_evidence, prepared
from scripts.count_likelihood_regressor import CountLikelihoodRegressor
from scripts.discrete_target_projection import TargetDomainProjection
from scripts.engagement_local_comparison import freeze
from scripts.recheck_zero_aware_strong_control import verify_hashes
from scripts.run_gap_followup_queue import digest, dump
from scripts.select_activity_readout import cache_locations


WEIGHTS = (.125, .5, 1.)


def run(args):
    started = time.monotonic()
    raw_path = args.task_root / "raw-audit/manifest.json"
    raw = json.loads(raw_path.read_text())
    meta = raw["metadata"]
    policy = TemporalFitPolicy(**json.loads((args.task_root / "selection/results.json").read_text())["selected_policy"])
    if (meta["metric"] != "mae" or not meta["atomic_date_replay"] or meta["fold"] != "current"
            or policy.mode != "frame_ensemble" or policy.readout != "direct" or policy.blend_weight != 1):
        raise ValueError("Corrected direct numeric frame-ensemble control required")
    cache, snapshots = cache_locations(args.task_root)
    folds = list(range(len(raw["folds"])))
    if len(folds) < 2 or raw["seeds"] != [0, 1, 2]:
        raise ValueError("All available origins and three seeds are required")
    inputs = [raw_path, args.task_root / "selection/results.json", cache / "history.parquet",
              *[snapshots / f"fold{f}" / n for f in folds for n in ("manifest.json", "validation.parquet")],
              *[args.task_root / f"raw-audit/fold{f}-{policy.name}-seed{s}.npz" for f in folds for s in range(3)]]
    sources = [__file__, "scripts/count_likelihood_regressor.py", "scripts/overdispersed_count_readout.py",
               "scripts/discrete_target_projection.py", "scripts/benchmark_recovered_readouts.py",
               "scripts/benchmark_signal_challengers.py", "kurversc/modeling.py", "kurversc/temporal_policy.py"]
    identity = dict(metadata=meta, policy=asdict(policy), folds=folds, seeds=[0, 1, 2],
        weights=WEIGHTS, allowance=1024, same_graph=True, same_complete_mature_support=True,
        same_frame_sampling=True, maximum_live_training_cohorts=1, same_scalar_tree_cap=True,
        default_changed=False, official_test_accessed=False,
        sources={n: digest(n) for n in sources}, inputs={str(p): digest(p) for p in inputs})
    freeze(args.output / "manifest.json", identity)
    recipes = [f"count_likelihood_{w:g}" for w in WEIGHTS]
    active, records, panels = set(recipes), [], {"fresh": [], "archive": []}
    schema, target = meta["feature_schema"], meta["target"]
    for fold in reversed(folds):
        if not active:
            break
        folder = args.output / f"fold{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        train, held, phase = phase_frames(args.task_root, raw, policy, fold, maximum_feature_cells=0)
        projector = TargetDomainProjection().fit(train[target], pd.to_datetime(train.__event_cutoff),
            origin=phase["fit_origin"], horizon=meta["label_horizon"])
        if not projector.integer_counts:
            dump(args.output / "results.json", dict(selected="incumbent", reason="not_integral_training_targets",
                 official_test_accessed=False, default_changed=False))
            return
        cutoffs = sorted(train.__event_cutoff.unique(), key=pd.Timestamp)
        payload = pd.read_parquet(snapshots / f"fold{fold}/validation.parquet", columns=schema["columns"])
        vx = prepared(payload, schema)
        del payload
        categorical = [i for i, n in enumerate(vx) if n in schema["categorical_columns"]]
        for seed in range(3):
            if not active:
                break
            archive = incumbent_evidence(args.task_root / "raw-audit", fold, seed, policy)
            np.testing.assert_array_equal(archive["truth"], held[target])
            np.testing.assert_array_equal(archive["groups"], held.__event_entity)
            native_members, count_members, fits = [], [], []
            for number, cutoff in enumerate(cutoffs, 1):
                print(json.dumps(dict(phase="fit_streamed_count_member", fold=fold, seed=seed,
                    member=number, members=len(cutoffs), cutoff=str(cutoff))), flush=True)
                frame = pd.read_parquet(cache / "history.parquet", columns=[*schema["columns"], target],
                                         filters=[("__event_cutoff", "==", cutoff)])
                x = prepared(frame, schema)
                x, y = sample_training_rows(x, frame[target], limit=raw["row_limit"], task="regression",
                                           random_state=seed+number)
                del frame
                gc.collect()
                native, _, _ = fit_final_catboost(x, y, task="regression", categorical=categorical,
                    random_state=seed, model_params=raw["model_params"])
                native_members.append(native.predict(vx))
                if fold == folds[-1] and seed == 0:
                    again, _, _ = fit_final_catboost(x, y, task="regression", categorical=categorical,
                        random_state=seed, model_params=raw["model_params"])
                    np.testing.assert_allclose(native_members[-1], again.predict(vx), atol=1e-9, rtol=0)
                    del again
                del native
                gc.collect()
                model = CountLikelihoodRegressor(seed=seed, model_params=raw["model_params"]).fit(x, y, categorical=categorical)
                prediction = model.predict(vx)
                np.testing.assert_array_equal(prediction, pickle.loads(pickle.dumps(model)).predict(vx))
                if fold == folds[-1] and seed == 0:
                    again = CountLikelihoodRegressor(seed=seed, model_params=raw["model_params"]).fit(x, y, categorical=categorical)
                    np.testing.assert_allclose(prediction, again.predict(vx), atol=1e-9, rtol=0)
                    del again
                count_members.append(prediction)
                fits.append(dict(cutoff=str(cutoff), rows=len(y), audit=model.audit))
                del model, x, y
                gc.collect()
            native, head = np.mean(native_members, axis=0), np.mean(count_members, axis=0)
            fresh = projector.predict_values(native, recipe="integer_counts")
            old = projector.predict_values(archive["prediction"], recipe="integer_counts")
            vectors = dict(truth=archive["truth"], groups=archive["groups"], native=native,
                           raw_count_median=head, fresh=fresh, archive=old)
            for name in active:
                weight = float(name.rsplit("_", 1)[1])
                vectors["fresh_"+name] = projector.predict_values((1-weight)*native+weight*head, recipe="integer_counts")
                vectors["archive_"+name] = projector.predict_values((1-weight)*archive["prediction"]+weight*head, recipe="integer_counts")
            scores = {n: _score(archive["truth"], p, "regression") for n, p in vectors.items() if n not in {"truth", "groups"}}
            report = dict(fold=fold, seed=seed, origin=phase["origin"], scores=scores, fits=fits,
                native_archive_max_error=float(np.max(np.abs(native-archive["prediction"]))),
                manifest_sha256=digest(args.output / "manifest.json"))
            np.savez_compressed(folder / f"seed{seed}.npz", **vectors)
            dump(folder / f"seed{seed}.json", report)
            records.append(report)
            for panel in panels:
                panels[panel].append(dict(origin=phase["origin"], seed=seed, truth=archive["truth"], groups=archive["groups"],
                    control=vectors[panel], vectors={n: vectors[panel+"_"+n] for n in active}))
            print(json.dumps(dict(phase="streamed_count_case_complete", fold=fold, seed=seed, scores=scores)), flush=True)
            active = {n for n in active if scores["fresh_"+n] < scores["fresh"] and scores["archive_"+n] < scores["archive"]}
            dump(args.output / "progress.json", dict(records=records, active_recipes=sorted(active)))
        del train, held, vx
        gc.collect()
    gates = {n: matched_gate(panel, metric="mae", recipes=recipes, allowance=1024) for n, panel in panels.items()}
    selected = gates["fresh"]["selected"]
    if selected not in active or any(g["candidates"].get(selected, {}).get("reason") != "eligible" for g in gates.values()):
        selected = "incumbent"
    verify_hashes(identity["sources"])
    verify_hashes(identity["inputs"])
    dump(args.output / "results.json", dict(selected=selected, gates=gates, records=records,
        complete_grid=len(records) == 3*len(folds), official_test_accessed=False, default_changed=False,
        elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(phase="count_likelihood_complete", selected=selected, cases=len(records))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
