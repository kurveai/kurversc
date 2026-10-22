"""Verify one already-selected zero-aware head against the strongest readout.

The original weight is locked, not retuned. Both predictions receive the
already-admitted training-only integer-count readout. Recheck every saved
fold/seed and independently refit the newest control/head before admission.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from kurversc.temporal_policy import TemporalFitPolicy, _score
from scripts.benchmark_large_gap_challengers import fit_prediction
from scripts.benchmark_signal_challengers import prepared
from scripts.discrete_target_projection import TargetDomainProjection
from scripts.engagement_local_comparison import freeze
from scripts.frozen_auc_bootstrap import frozen_auc_select_temporal_policy
from scripts.run_gap_followup_queue import digest, dump


MENU_ALLOWANCE = 200


def complete_matched_cases(records, folds):
    expected = {(f,s) for f in folds for s in range(3)}
    keys = [(r["fold"],r["seed"]) for r in records]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError("Exactly the complete frozen fold/seed grid is required")


def verify_hashes(identity):
    if any(digest(path)!=sha for path,sha in identity.items()):
        raise ValueError("Locked implementation or evidence changed")


def projected_pair(head, control, candidate, recipe):
    return (head.predict_values(control,recipe=recipe),
            head.predict_values(candidate,recipe=recipe))


def run(args):
    started = time.monotonic()
    previous = json.loads((args.validation/"results.json").read_text())
    manifest = json.loads((args.validation/"manifest.json").read_text())
    choice = previous["decision"]["selected"]
    if (choice not in previous["complete_positive_recipes"] or not choice.startswith("zero_aware_")
            or previous["official_test_accessed"] or previous["default_changed"]
            or previous["decision"]["candidates"][choice]["reason"]!="eligible"
            or previous["decision"]["candidates"][choice]["folds"][-1].get("recent_lower_bound",0)<=0):
        raise ValueError("A complete, confident validation-selected zero-aware head is required")
    folds = manifest["folds"]
    complete_matched_cases(previous["records"],folds)
    raw = json.loads((args.task_root/"raw-audit/manifest.json").read_text())
    meta = raw["metadata"]
    if meta!=manifest["metadata"] or meta["metric"]!="mae":
        raise ValueError("Frozen graph or numeric task differs")
    policy = TemporalFitPolicy(**manifest["policy"])
    strong_manifest = json.loads((args.strong_control/"first/manifest.json").read_text())
    strong_selection = json.loads((args.strong_control/"first/selection.json").read_text())
    repeated = json.loads((args.strong_control/"repeat/selection.json").read_text())
    recipe = strong_selection["decision"]["selected"]
    if (recipe!="integer_counts" or strong_manifest["metadata"]!=meta
            or strong_manifest["policy"]!=vars(policy)
            or strong_selection["decision"]!=repeated["decision"]
            or strong_selection["decision"]["candidates"][recipe]["reason"]!="eligible"):
        raise ValueError("The independently admitted stronger integer control is required")
    sources = {**manifest["sources"],**manifest["input_files"]}
    for path in [__file__,"scripts/discrete_target_projection.py",
                 args.validation/"manifest.json",args.validation/"results.json",
                 args.strong_control/"first/manifest.json",args.strong_control/"first/selection.json",
                 args.strong_control/"repeat/selection.json"]:
        sources[str(path)] = digest(path)
    verify_hashes(sources)
    args.output.mkdir(parents=True,exist_ok=True)
    identity = dict(metadata=meta,policy=vars(policy),selected=choice,projection=recipe,
        menu_allowance=MENU_ALLOWANCE,folds=folds,seeds=[0,1,2],sources=sources,
        candidate_weight_retuned=False,default_changed=False,official_test_accessed=False)
    freeze(args.output/"manifest.json",identity)
    if (args.output/"results.json").exists():
        return
    evidence,reports = [],[]
    for fold in reversed(folds):
        staged = args.validation/f"fold{fold}/inputs/fold{fold}"
        phase = json.loads((staged/"phase.json").read_text())
        train = pd.read_parquet(staged/"train.parquet",columns=[meta["target"],"__event_cutoff"])
        head = TargetDomainProjection().fit(train[meta["target"]],train.__event_cutoff,
            origin=phase["fit_origin"],horizon=meta["label_horizon"])
        for seed in range(3):
            original_path = args.validation/f"fold{fold}/seed{seed}.npz"
            control_path = args.strong_control/f"first/fold{fold}/seed{seed}.npz"
            repeat_path = args.strong_control/f"repeat/fold{fold}/seed{seed}.npz"
            sources.update({str(p):digest(p) for p in (original_path,control_path,repeat_path)})
            with np.load(original_path) as a,np.load(control_path) as b,np.load(repeat_path) as c:
                for key in ("truth","groups"):
                    np.testing.assert_array_equal(a[key],b[key])
                    np.testing.assert_array_equal(b[key],c[key])
                control,candidate = projected_pair(head,a["control"],a[choice],recipe)
                np.testing.assert_array_equal(control,b[recipe])
                np.testing.assert_array_equal(b[recipe],c[recipe])
                arrays = dict(truth=a["truth"].copy(),groups=a["groups"].copy(),control=control,candidate=candidate)
            scores = {name:_score(arrays["truth"],arrays[name],"regression") for name in ("control","candidate")}
            report = dict(fold=fold,seed=seed,origin=phase["origin"],scores=scores,
                gain=scores["control"]-scores["candidate"],projection=head.audit)
            folder = args.output/f"fold{fold}"
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/f"seed{seed}.npz",**arrays)
            dump(folder/f"seed{seed}.json",report)
            reports.append(report)
            for name,key in (("incumbent","control"),(choice,"candidate")):
                evidence.append(dict(origin=phase["origin"],seed=seed,policy=name,
                    truth=arrays["truth"],groups=arrays["groups"],prediction=arrays[key]))
            print(json.dumps(report),flush=True)
    complete_matched_cases(reports,folds)
    decision = dict(selected="incumbent",reason="nonpositive_stronger_control_comparison")
    replay_error = None
    if all(r["gain"]>0 for r in reports):
        fold = max(folds)
        staged = args.validation/f"fold{fold}/inputs/fold{fold}"
        train,held = [pd.read_parquet(staged/n) for n in ("train.parquet","validation.parquet")]
        phase = json.loads((staged/"phase.json").read_text())
        x,vx = prepared(train,meta["feature_schema"]),prepared(held,meta["feature_schema"])
        y,times = train[meta["target"]],pd.to_datetime(train.__event_cutoff)
        print(json.dumps(dict(phase="independent_control_and_zero_head_refit",fold=fold,seed=0)),flush=True)
        replay,_ = fit_prediction(x,y,times,vx,raw=raw,meta=meta,policy=policy,phase=phase,seed=0)
        zero,_ = fit_prediction(x,y,times,vx,raw=raw,meta=meta,policy=policy,phase=phase,seed=0,zero=True)
        with np.load(args.validation/f"fold{fold}/seed0.npz") as a:
            np.testing.assert_allclose(replay,a["control"],atol=1e-9,rtol=0)
            np.testing.assert_allclose(zero,a["zero_aware_1"],atol=1e-9,rtol=0)
            replay_error = float(max(np.max(np.abs(replay-a["control"])),np.max(np.abs(zero-a["zero_aware_1"]))))
        decision = frozen_auc_select_temporal_policy(evidence,task="regression",minimum_origins=len(folds),
            minimum_seeds=3,deployment_seed=0,bootstrap_repeats=1600,alpha=.05/MENU_ALLOWANCE)
        if decision["selected"]!="incumbent" and decision["candidates"][choice]["folds"][-1].get("recent_lower_bound",0)<=0:
            decision = {**decision,"selected":"incumbent","reason":"no_confident_recent_gain"}
    verify_hashes(sources)
    dump(args.output/"locked-evidence.json",sources)
    dump(args.output/"results.json",dict(selected=decision["selected"],decision=decision,records=reports,
        independent_replay_max_error=replay_error,complete_grid=True,selected_weight_retuned=False,
        locked_evidence_sha256=digest(args.output/"locked-evidence.json"),
        elapsed_seconds=time.monotonic()-started,official_test_accessed=False,default_changed=False))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root","validation","strong-control","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())
