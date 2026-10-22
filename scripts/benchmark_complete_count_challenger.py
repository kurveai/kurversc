"""Complete paired count evidence under the existing production temporal rule.

Unlike the early-rejection research scout, always finish the declared grid.
Preserve its graph, mature support, member sampling, budgets and head. Reuse
only hash-verified matching completed cases. Independent cases may run in
parallel, in disposable workers, without changing fitting or selection order.
No official split, score, historical graph winner or default is accessed.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gc
import json
import multiprocessing
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from kurversc.modeling import fit_final_catboost, sample_training_rows
from kurversc.temporal_policy import TemporalFitPolicy, _score
from scripts.benchmark_recovered_readouts import phase_frames
from scripts.benchmark_signal_challengers import incumbent_evidence, prepared
from scripts.discrete_target_projection import TargetDomainProjection
from scripts.engagement_local_comparison import freeze
from scripts.frozen_auc_bootstrap import frozen_auc_select_temporal_policy
from scripts.poisson_count_median import PoissonCountMedian
from scripts.run_gap_followup_queue import digest, dump
from scripts.select_activity_readout import cache_locations


WEIGHTS = (.125, .5, 1.)
RECIPES = tuple(f"count_likelihood_{w:g}" for w in WEIGHTS)
ALLOWANCE = 1024


def choose(panels, *, origins):
    gates = {name:frozen_auc_select_temporal_policy(records, task="regression",
        minimum_origins=origins, minimum_seeds=3, deployment_seed=0,
        bootstrap_repeats=1600, alpha=.05*len(RECIPES)/ALLOWANCE)
        for name, records in panels.items()}
    eligible = []
    for name in RECIPES:
        rows = [gate.get("candidates", {}).get(name, {}) for gate in gates.values()]
        if all(r.get("reason") == "eligible"
               and r["folds"][-1].get("recent_lower_bound", -1) > 0
               and r["folds"][-1].get("temporal_gain_lower_bound", -1) > 0
               and all(min(fold["seed_gains"]) >= -fold["practical_loss_cap"]
                       for fold in r["folds"][:-1]) for r in rows):
            eligible.append((rows[0]["folds"][-1]["gain"], float(name.rsplit("_",1)[1]), name))
    selected = min(eligible, key=lambda v:(-v[0],v[1]))[-1] if eligible else "incumbent"
    return selected, gates


def verify(files):
    if any(digest(path) != expected for path, expected in files.items()):
        raise ValueError("Frozen complete-count input or source changed")


def case(task_root, output, reuse, fold, seed):
    task_root, output = Path(task_root), Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    raw = json.loads((task_root / "raw-audit/manifest.json").read_text())
    meta, policy = raw["metadata"], TemporalFitPolicy(**manifest["policy"])
    cache, snapshots = cache_locations(task_root)
    train, held, phase = phase_frames(task_root, raw, policy, fold, maximum_feature_cells=0)
    projector = TargetDomainProjection().fit(train[meta["target"]], pd.to_datetime(train.__event_cutoff),
        origin=phase["fit_origin"], horizon=meta["label_horizon"])
    if not projector.integer_counts:
        raise ValueError("Variable mature integer targets required")
    archive = incumbent_evidence(task_root / "raw-audit", fold, seed, policy)
    np.testing.assert_array_equal(archive["truth"], held[meta["target"]])
    np.testing.assert_array_equal(archive["groups"], held.__event_entity)
    folder = output / f"fold{fold}"
    folder.mkdir(exist_ok=True)
    path, report = folder / f"seed{seed}.npz", folder / f"seed{seed}.json"
    if path.exists() and report.exists():
        record = json.loads(report.read_text())
        if record["vectors_sha256"] != digest(path):
            raise ValueError("Completed case changed")
        return record
    previous = Path(reuse) / f"fold{fold}/seed{seed}.npz" if reuse else None
    if previous is not None and str(previous) in manifest["reused_case_files"]:
        old_report = previous.with_suffix(".json")
        verify({str(p):manifest["reused_case_files"][str(p)] for p in (previous, old_report)})
        with np.load(previous) as saved:
            for key in ("truth", "groups"):
                np.testing.assert_array_equal(saved[key], archive[key])
            native, head = saved["native"].copy(), saved["raw_count_median"].copy()
        fits = json.loads(old_report.read_text())["fits"]
        reused = True
    else:
        payload = pd.read_parquet(snapshots / f"fold{fold}/validation.parquet", columns=meta["feature_schema"]["columns"])
        vx = prepared(payload, meta["feature_schema"])
        del payload
        categorical = [i for i,n in enumerate(vx) if n in meta["feature_schema"]["categorical_columns"]]
        native_members, head_members, fits = [], [], []
        cutoffs = sorted(train.__event_cutoff.unique(), key=pd.Timestamp)
        for index, cutoff in enumerate(cutoffs, 1):
            print(json.dumps(dict(phase="complete_count_member",fold=fold,seed=seed,member=index,members=len(cutoffs))),flush=True)
            frame = pd.read_parquet(cache / "history.parquet", columns=[*meta["feature_schema"]["columns"],meta["target"],
                *(["__event_entity"] if meta.get("label_available_at") else [])],
                filters=[("__event_cutoff","==",cutoff)])
            if meta.get("label_available_at"):
                known = train.loc[train.__event_cutoff.eq(cutoff),"__event_entity"]
                frame = frame.loc[frame.__event_entity.isin(known)].reset_index(drop=True)
            x = prepared(frame,meta["feature_schema"])
            x,y = sample_training_rows(x,frame[meta["target"]],limit=raw["row_limit"],task="regression",random_state=seed+index)
            del frame
            gc.collect()
            native_model,_,_ = fit_final_catboost(x,y,task="regression",categorical=categorical,
                random_state=seed,model_params=raw["model_params"])
            native_members.append(native_model.predict(vx))
            del native_model
            gc.collect()
            model = PoissonCountMedian(seed=seed,model_params=raw["model_params"]).fit(x,y,categorical=categorical)
            prediction = model.predict(vx)
            np.testing.assert_array_equal(prediction,pickle.loads(pickle.dumps(model)).predict(vx))
            if fold == max(manifest["folds"]) and seed == 0:
                repeated = PoissonCountMedian(seed=seed,model_params=raw["model_params"]).fit(x,y,categorical=categorical)
                np.testing.assert_array_equal(prediction,repeated.predict(vx))
                del repeated
            head_members.append(prediction)
            fits.append(dict(cutoff=str(cutoff),rows=len(y),audit=model.audit))
            del model,x,y
            gc.collect()
        native,head = np.mean(native_members,axis=0),np.mean(head_members,axis=0)
        reused = False
    np.testing.assert_allclose(native,archive["prediction"],atol=1e-9,rtol=0)
    arrays = dict(truth=archive["truth"],groups=archive["groups"],native=native,raw_count_median=head,
        fresh=projector.predict_values(native,recipe="integer_counts"),
        archive=projector.predict_values(archive["prediction"],recipe="integer_counts"))
    for weight,name in zip(WEIGHTS,RECIPES):
        arrays["fresh_"+name] = projector.predict_values((1-weight)*native+weight*head,recipe="integer_counts")
        arrays["archive_"+name] = projector.predict_values((1-weight)*archive["prediction"]+weight*head,recipe="integer_counts")
    np.savez_compressed(path,**arrays)
    record = dict(fold=fold,seed=seed,origin=phase["origin"],reused=reused,fits=fits,
        native_archive_max_error=float(np.max(np.abs(native-archive["prediction"]))),
        scores={n:_score(arrays["truth"],v,"regression") for n,v in arrays.items() if n not in {"truth","groups"}},
        vectors_sha256=digest(path),manifest_sha256=digest(output / "manifest.json"))
    dump(report,record)
    print(json.dumps(dict(phase="complete_count_case",fold=fold,seed=seed,scores=record["scores"],reused=reused)),flush=True)
    return record


def run(args):
    started = time.monotonic()
    raw_path = args.task_root / "raw-audit/manifest.json"
    raw = json.loads(raw_path.read_text())
    policy_path = args.task_root / "selection/results.json"
    policy = TemporalFitPolicy(**json.loads(policy_path.read_text())["selected_policy"])
    meta,folds = raw["metadata"],list(range(len(raw["folds"])))
    if (meta["metric"] != "mae" or not meta["atomic_date_replay"] or meta["fold"] != "current"
            or policy.mode != "frame_ensemble" or policy.readout != "direct" or policy.blend_weight != 1
            or len(folds) < 2 or raw["seeds"] != [0,1,2]):
        raise ValueError("Corrected complete direct frame-ensemble count audit required")
    cache,snapshots = cache_locations(args.task_root)
    files = [raw_path,policy_path,cache / "history.parquet",*[snapshots/f"fold{f}"/n for f in folds
        for n in ("manifest.json","validation.parquet")],*[args.task_root/f"raw-audit/fold{f}-{policy.name}-seed{s}.npz"
        for f in folds for s in range(3)]]
    reused = {}
    if args.reuse:
        previous = json.loads((args.reuse / "manifest.json").read_text())
        if previous["metadata"] != meta or previous["policy"] != asdict(policy) or previous["profile"] != "conditional_poisson_median":
            raise ValueError("Only matching head/support/graph evidence may be reused")
        verify(previous["sources"])
        verify(previous["inputs"])
        files.append(args.reuse / "manifest.json")
        for f in folds:
            for s in range(3):
                paths=[args.reuse/f"fold{f}/seed{s}.{ext}" for ext in ("npz","json")]
                if all(p.exists() for p in paths):
                    reused.update({str(p):digest(p) for p in paths})
    sources=[Path(__file__),Path("scripts/poisson_count_median.py"),Path("scripts/benchmark_streamed_count_likelihood.py"),
        Path("scripts/benchmark_recovered_readouts.py"),Path("scripts/frozen_auc_bootstrap.py"),
        Path("scripts/discrete_target_projection.py"),Path("kurversc/temporal_policy.py"),Path("kurversc/modeling.py")]
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=dict(metadata=meta,policy=asdict(policy),folds=folds,seeds=[0,1,2],weights=WEIGHTS,
        allowance=ALLOWANCE,profile="complete_conditional_count_production_rule",
        selection_rule="existing_production_temporal_veto",older_all_seed_relative_loss_cap=.01,
        sources={str(p):digest(p) for p in sources},inputs={str(p):digest(p) for p in files},reused_case_files=reused,
        workers=args.workers,default_changed=False,official_test_accessed=False)
    freeze(args.output / "manifest.json",manifest)
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn"),max_tasks_per_child=1) as pool:
        jobs={pool.submit(case,str(args.task_root),str(args.output),str(args.reuse) if args.reuse else None,f,s):(f,s)
              for f in reversed(folds) for s in range(3)}
        for future in as_completed(jobs):
            records.append(future.result())
            dump(args.output / "progress.json",dict(cases=len(records),required_cases=len(jobs),records=sorted(records,key=lambda r:(r["fold"],r["seed"]))))
    panels={n:[] for n in ("fresh","archive")}
    records.sort(key=lambda r:(r["fold"],r["seed"]))
    for r in records:
        with np.load(args.output/f"fold{r['fold']}/seed{r['seed']}.npz") as saved:
            for name in panels:
                panels[name].append(dict(origin=r["origin"],seed=r["seed"],truth=saved["truth"].copy(),
                    groups=saved["groups"].copy(),policy="incumbent",prediction=saved[name].copy()))
                for recipe in RECIPES:
                    panels[name].append(dict(origin=r["origin"],seed=r["seed"],truth=saved["truth"].copy(),
                        groups=saved["groups"].copy(),policy=recipe,prediction=saved[name+"_"+recipe].copy()))
    selected,gates=choose(panels,origins=len(folds))
    verify(manifest["sources"]);verify(manifest["inputs"]);verify(reused)
    dump(args.output / "results.json",dict(selected=selected,gates=gates,records=records,complete_grid=True,
        official_test_accessed=False,default_changed=False,elapsed_seconds=time.monotonic()-started,
        selection_rule=manifest["selection_rule"]))
    print(json.dumps(dict(phase="complete_count_selection",selected=selected,cases=len(records))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reuse",type=Path)
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    if not 1 <= args.workers <= 3:parser.error("One to three isolated case workers required")
    run(args)


if __name__ == "__main__":
    main()
