"""One locked official replay of an accepted pooled-row-budget challenger.

The full selected production blend must independently replay before the larger
budget is scored. Test labels cannot influence any recipe or blend choice.
"""

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from kurversc.signal_challengers import SignalChallenger
from kurversc.temporal_policy import fit_temporal_policy
from scripts.benchmark_selected_row_budget import primitive_policy, replacement_prediction
from scripts.benchmark_signal_challengers import materialize_outer, prepared, values
from scripts.engagement_local_comparison import freeze
from scripts.run_gap_followup_queue import digest, dump


def locked_choice(folder):
    selection = json.loads((folder/"selection.json").read_text())
    identity_path = folder/"selected-budget-identity.json"
    if selection["identity_sha256"] != digest(identity_path) or not selection["complete_grid"]:
        raise ValueError("Incomplete or changed validation evidence")
    name = selection["decision"]["selected"]
    match = re.fullmatch(r"rows(\d+)__weight_([.\d]+)", name)
    if not match or name not in selection["surviving"]:
        raise ValueError("Only an accepted validation-selected budget may be confirmed")
    budget, weight = int(match[1]), float(match[2])
    if len(selection["reports"]) != 9 or {(r["fold"],r["seed"]) for r in selection["reports"]} != {(f,s) for f in range(3) for s in range(3)}:
        raise ValueError("Complete three-origin/three-seed gate required")
    identity = json.loads(identity_path.read_text())
    if budget not in identity["budgets"][1:] or weight not in identity["weights"]:
        raise ValueError("Selected recipe differs from the frozen validation menu")
    return selection, identity, budget, weight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-root",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    started = time.monotonic()
    selected, identity, budget, weight = locked_choice(args.validation)
    raw = json.loads((args.task_root/"raw-audit/manifest.json").read_text())
    chosen = json.loads((args.task_root/"selection/results.json").read_text())["selected_policy"]
    policy, primitive = primitive_policy(chosen,raw["policies"])
    if asdict(policy) != identity["selected_policy"] or asdict(primitive) != identity["primitive_policy"]:
        raise ValueError("Protected production policy changed")
    metadata = raw["metadata"]
    baseline = json.loads((args.task_root/"official/results.json").read_text())
    if baseline["checkpoint_sha256"] != metadata["checkpoint_sha256"] or baseline["arm"] != policy.name:
        raise ValueError("Official control has a different plan/policy")
    freeze(args.output/"manifest.json",dict(validation_sha256=digest(args.validation/"selection.json"),
        identity_sha256=digest(args.validation/"selected-budget-identity.json"),
        baseline_sha256=digest(args.task_root/"official/predictions.npz"),
        selected=selected["decision"]["selected"],budget=budget,weight=weight,
        policy=asdict(policy),primitive=asdict(primitive),model_params=raw["model_params"],
        source_sha256=digest(__file__),scope="Locked validation decision; exact original replay before official scoring; no default changes"))
    if (args.output/"results.json").exists():
        return
    print(json.dumps(dict(phase="materialize_locked_official",budget=budget)),flush=True)
    source,outer,train,forecast = materialize_outer(args,metadata,SignalChallenger("row_budget_replay","support"),policy)
    schema = metadata["feature_schema"]
    x,vx = prepared(train,schema),prepared(forecast,schema)
    y,times = train[metadata["target"]].copy(),pd.to_datetime(train.__event_cutoff)
    origin = pd.to_datetime(outer.eval_table.df[source.task.time_col]).min()
    categories = [i for i,c in enumerate(x) if c in schema["categorical_columns"]]
    with np.load(args.task_root/"official/predictions.npz") as a:
        frame_control,strong = a["control"].copy(),a["candidate"].copy()
    del train,forecast
    gc.collect()
    vectors = {}
    for ceiling in (raw["row_limit"],budget):
        print(json.dumps(dict(phase="fit_locked_budget",rows=ceiling,threads=raw["model_params"].get("thread_count"))),flush=True)
        model,audit = fit_temporal_policy(x,y,times,origin=origin,
            label_horizon=metadata["label_horizon"],refit_frames=metadata["validation_cutoffs"],
            policy=primitive,task="classification",categorical=categories,random_state=0,
            row_limit=ceiling,model_params=raw["model_params"])
        vectors[ceiling] = values(model,vx,"classification")
        dump(args.output/f"fit-rows{ceiling}.json",audit)
        if ceiling == raw["row_limit"]:
            replay = (1-policy.blend_weight)*frame_control + policy.blend_weight*vectors[ceiling]
            error = float(np.max(np.abs(replay-strong)))
            dump(args.output/"control-replay.json",dict(maximum_absolute_error=error,tolerance=1e-9))
            np.testing.assert_allclose(replay,strong,atol=1e-9,rtol=0,
                err_msg="Official protected production blend did not independently replay")
        del model
        gc.collect()
    candidate = replacement_prediction(strong,vectors[raw["row_limit"]],vectors[budget],policy.blend_weight,weight)
    np.savez_compressed(args.output/"predictions.npz",control=strong,candidate=candidate)
    # Every choice and both ordered prediction vectors are fixed before scoring.
    scores = {name:float(source.task.evaluate(pred,None,metrics=[source.metric])[source.metric.__name__])
              for name,pred in (("control",strong),("candidate",candidate))}
    np.testing.assert_allclose(scores["control"],baseline["official_test_scores"]["candidate"],atol=1e-12,rtol=0)
    result = dict(selected=selected["decision"]["selected"],official_test_scores=scores,
        gain=scores["candidate"]-scores["control"],elapsed_seconds=time.monotonic()-started,
        default_changed=False,scope="Separate locked confirmation; no claim that accumulated earlier research completed within 24 hours")
    dump(args.output/"results.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
