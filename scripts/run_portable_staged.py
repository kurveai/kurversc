"""Run the complete staged recipe on an explicit, unregistered task context.

This development entry point reuses the score-producing stage implementations;
it does not replace them with a cheaper lookalike. No dataset download, known
winner, published score, or held-out answer table is required. Subprocesses
inherit one checksummed task context. Runtime/dependency packaging is separate
from the data-source protocol exercised here.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from kurversc.prediction_provenance import artifact_keys, align_prediction, masked_query_keys
from kurversc.task_context import _file_digest
from kurversc.task_source import CONTEXT_ENV, ContextTaskSource
from kurversc.staged_runtime import staged_source_identity
from kurversc.checkpointing import search_runtime_identity
from scripts.run_gap_followup_queue import dump
from scripts.run_reproducible_staged_cloud import PROFILE, install_recipe


def require_supported_availability(source):
    """Validate explicit label clocks before invoking any stage worker."""
    split = source.outer_split()
    task = source.task
    for table in (split.train_table,split.val_table):
        expected = pd.to_datetime(table.df[task.time_col])+task.timedelta
        actual = pd.to_datetime(table.df[task.label_available_at]) if task.label_available_at else expected
        if actual.isna().any() or (actual<expected).any():
            raise ValueError("Labels cannot be observable before their prediction horizon completes")


def prediction_artifact(root,slug):
    task = root/"workers"/slug/"temporal/tasks"/slug
    # Resolve the procedure actually admitted, never whichever prediction file
    # happens to have the most recent timestamp or the most favorable score.
    bounded = task/"bounded-views"
    if (bounded/"admission.json").exists():
        admission = json.loads((bounded/"admission.json").read_text())
        if admission["admitted"]:
            return bounded/"official/predictions.npz"
    sequence = task/"production-sequence"
    if (sequence/"admission.json").exists():
        admission = json.loads((sequence/"admission.json").read_text())
        if admission["admitted"]:
            return sequence/"official/predictions.npz"
    count = task/"conditional-count"
    admission = json.loads((count/"admission.json").read_text())
    if admission["admitted"]:
        return count/"official/predictions.npz"
    return task/"recovered-readouts/predictions.npz"


def remaining_context_budget(output, *, hours, preprocessing_seconds, now=None):
    """One original wall deadline, including preparation and interrupted waits."""
    now = time.time() if now is None else now
    path = Path(output)/"portable-state.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state["task_hours"]!=hours:
            raise ValueError("Cannot reset a portable task deadline by changing its budget")
    else:
        state = dict(task_hours=hours,first_started_epoch=now-preprocessing_seconds,
            deadline_epoch=now-preprocessing_seconds+hours*3600)
        dump(path,state)
    seconds = state["deadline_epoch"]-now
    if seconds<=0:
        raise TimeoutError("The original complete-procedure deadline is exhausted")
    return seconds


def run(args):
    source = ContextTaskSource(args.context)
    require_supported_availability(source)
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    dataset,task = source.manifest["dataset"],source.manifest["task"]
    for name in (dataset,task):
        if not name or Path(name).name!=name or name in {".",".."}:
            raise ValueError("Dataset/task namespace identifiers must be simple path components")
    os.environ[CONTEXT_ENV] = str(source.path)
    from scripts import run_fresh_cloud_campaign as fresh
    staged = install_recipe()
    identity = dict(profile=staged.PROFILE,contract=staged.CONTRACT,
        context_sha256=source.identity_digest,context=str(source.path),
        task_hours=args.hours,heldout_targets_present=False,
        benchmark_recipe_lookup=False,benchmark_registry_lookup=False,
        sources=staged_source_identity(Path(__file__).resolve().parents[1]),runtime=search_runtime_identity())
    manifest = output/"portable-manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())!=identity:
        raise ValueError("Refusing to resume a different portable procedure/context")
    dump(manifest,identity)
    remaining = remaining_context_budget(output,hours=args.hours,
        preprocessing_seconds=source.manifest.get("preprocessing_seconds",0.))
    key,slug = f"{dataset}/{task}",f"{dataset}_{task}"
    state_path = output/"campaign-status.json"
    if not state_path.exists():
        dump(state_path,dict(identity=dict(signal_challengers=True,complete_profile=staged.CONTRACT),
            tasks={key:dict(status="running",consumed_seconds=0)}))
    state = json.loads(state_path.read_text())
    state["tasks"][key]["consumed_seconds"] = args.hours*3600-remaining
    dump(state_path,state)
    started = time.monotonic()
    result = fresh.task_main(argparse.Namespace(output=output,dataset=dataset,task=task,
        search_workers=1,seed_workers=1,task_hours=args.hours,signal_challengers=True))
    if result:
        raise RuntimeError(f"Portable complete pipeline failed: exit={result}")
    if identity["sources"]!=staged_source_identity(Path(__file__).resolve().parents[1]):
        raise ValueError("Portable stage source changed during execution; refusing mixed-implementation output")
    report,stages = staged.verify_stages(output,dataset,task)
    artifact = prediction_artifact(output,slug)
    if not artifact.exists():
        raise ValueError("The admitted complete procedure did not save keyed predictions")
    with np.load(artifact) as a:
        prediction,keys = a["candidate"].copy(),artifact_keys(a)
    task_definition = source.task
    query = source.outer_split().eval_table.df
    expected = masked_query_keys(query,entity=task_definition.entity_col,
        timestamp=task_definition.time_col,target=task_definition.target_col)
    prediction = align_prediction(prediction,source_keys=keys,query_keys=expected)
    original = pd.read_parquet(source.root/"request-keys.parquet")
    if len(original)!=len(prediction):
        raise ValueError("Original request routing changed")
    original["prediction"] = prediction
    original.to_parquet(output/"predictions.parquet",index=False)
    dump(output/"portable-results.json",dict(status="completed",profile=staged.PROFILE,
        context_sha256=source.identity_digest,stages=stages,elapsed_seconds=time.monotonic()-started,
        predictions_sha256=_file_digest(output/"predictions.parquet"),
        intermediate_prediction_artifact=str(artifact),heldout_targets_used=False,test_score=None,
        scope="Full staged prediction on an explicit database; any holdout scoring is external"))
    print(json.dumps(dict(status="completed",predictions=str(output/"predictions.parquet"))),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--hours",type=float,default=23.75)
    args = p.parse_args()
    if not 0<args.hours<24:
        p.error("A positive complete-procedure budget below 24 hours is required")
    run(args)


if __name__=="__main__":
    main()
