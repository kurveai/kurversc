"""Explicit generic staged recipe without benchmark recovery extensions."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import os
import subprocess
import sys
import time

from scripts.benchmark_generic_readouts import CONTRACT as READOUT_CONTRACT
from scripts.run_gap_followup_queue import dump


PROFILE = "kurversc-generic-staged-v1"
CONTRACT = dict(
    graph_profile="staged-retention", base_cdf=True, peer_context="both",
    peer_max_history_rows=50_000, initial_production_training_frames=1,
    production_faithful_finalist_comparison=True,
    temporal_origins=3, temporal_seeds=[0,1,2],
    generic_readouts=READOUT_CONTRACT,
    source_event_cutoff="strict", label_availability_cutoff="inclusive",
    score_dependent_task_admission=False,
    task_hours=23.75,
)


def prediction_artifact(root, slug):
    return Path(root)/"workers"/slug/"temporal/tasks"/slug/"generic-readouts/predictions.npz"


def run_task(args):
    from scripts import run_fresh_cloud_campaign as fresh
    key = f"{args.dataset}/{args.task}"
    budget = fresh.remaining_task_seconds(args,key)
    started = time.monotonic()
    worker = args.output/"workers"/f"{args.dataset}_{args.task}"
    worker.mkdir(parents=True,exist_ok=True)
    graph_result = args.output/"graphs/results"/f"{args.dataset}_{args.task}.json"
    if not graph_result.exists():
        with (worker/"search.log").open("a",buffering=1) as log:
            subprocess.run([sys.executable,"-u","-m","scripts.run_generic_graph","--",
                *fresh.graph_arguments(args)],cwd=fresh.ROOT,env=dict(os.environ),
                stdout=log,stderr=subprocess.STDOUT,timeout=budget,check=True)
    # Invoke only the core temporal-policy stages. Optional candidates are a
    # separate explicit stage, not the old chain of recipe monkey patches.
    native = argparse.Namespace(**{**vars(args),"signal_challengers":False})
    code = fresh.task_main(native)
    if code:
        return code
    task_root = worker/"temporal/tasks"/f"{args.dataset}_{args.task}"
    output = task_root/"generic-readouts"
    if not (output/"results.json").exists():
        remaining = budget-(time.monotonic()-started)
        if remaining<=0:
            raise TimeoutError("Complete task budget exhausted before generic candidates")
        with (worker/"generic-readouts.log").open("a",buffering=1) as log:
            subprocess.run([sys.executable,"-u","-m","scripts.benchmark_generic_readouts",
                "--task-root",str(task_root),"--output",str(output),"--threads","4"],
                cwd=fresh.ROOT,env=dict(os.environ),stdout=log,
                stderr=subprocess.STDOUT,timeout=remaining,check=True)
    return 0


def verify_stages(root, dataset, task):
    slug = f"{dataset}_{task}"
    folder = Path(root)/"workers"/slug/"temporal/tasks"/slug
    graph = json.loads((Path(root)/"graphs/results"/f"{slug}.json").read_text())
    temporal = json.loads((folder/"selection/results.json").read_text())
    selected = json.loads((folder/"generic-readouts/selection.json").read_text())
    report = json.loads((folder/"generic-readouts/results.json").read_text())
    if graph["profile"]!="staged-retention" or graph["base_distribution_features"] is not True:
        raise ValueError("The generic graph search was omitted")
    if selected["selected"]!=report["selected"]:
        raise ValueError("Final prediction differs from validation selection")
    if selected["selected"]!="incumbent":
        if selected["decision"]["candidates"][selected["selected"]]["reason"]!="eligible":
            raise ValueError("Generic candidate lacks validation admission")
    return report,dict(graph_config=graph["config"],temporal_policy=temporal["selected_policy"],
        generic_selected=selected["selected"],generic_decision=selected["decision"],
        contract=CONTRACT)
