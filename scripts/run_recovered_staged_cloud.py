"""Fresh all-21 staged procedure with the omitted validated readout stages.

Run from an immutable source snapshot. Both attendance and engagement must
recover their client wins before the remaining nineteen tasks are admitted.
Acceptance scores never enter graph, feature, learner, or readout selection.
"""

import argparse
from argparse import Namespace
import json
import shutil
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
from kurversc.task_source import phase_source_root

from scripts import run_staged_complete_cloud as staged
from scripts import run_fresh_cloud_campaign as fresh
from scripts.run_gap_followup_queue import dump, validate_manifest


PROFILE = "staged-complete-recovered-v3"
CANARIES = (("rel-event", "user-attendance"), ("rel-stack", "user-engagement"))
READOUT_CONTRACT = dict(target_domain=True, zero_aware_quantiles=True,
    phase_frozen_object_history=True, recurrence_support3=True,
    joint_semantic_recurrence=True, prediction_keys_required=True,
    independent_semantic_repeat=True, task_specific_winner_lookup=False,
    deployment_seed=0, teacher_workers=4,
    optional_feature_cell_capacity=500_000_000,
    semantic_row_capacity=500_000, semantic_document_capacity=200_000,
    incumbent_preserved_when_challenger_exceeds_capacity=True)


def readout_root(root, key):
    return root / "workers" / key.replace("/", "_") / "temporal/tasks" / key.replace("/", "_") / "recovered-readouts"


def retain_without_temporal_evidence(folder):
    """An unsupported optional audit must not make an unseen short task fail."""
    baseline = folder.parent/"signal-challengers"
    report = json.loads((baseline/"results.json").read_text())
    folder.mkdir(parents=True,exist_ok=True)
    choice = dict(selected="incumbent",projection="incumbent",decisions={},complete_grid=False,
        fallback="insufficient_temporal_evidence",official_targets_used_for_selection=False)
    dump(folder/"selection.json",choice)
    shutil.copyfile(baseline/"predictions.npz",folder/"predictions.npz")
    dump(folder/"results.json",dict(**report,selected="incumbent",deployment_seed=0,
        recovered_readout_fallback=choice["fallback"]))


def recovery_manifest(manifest):
    manifest = json.loads(json.dumps(manifest))
    for job in manifest["jobs"]:
        job["command"] = ["scripts.run_recovered_staged_cloud" if x == "scripts.run_staged_complete_cloud" else x
                          for x in job["command"]]
        if job["name"] == "attendance-recovery-gate":
            job["name"] = "readout-recovery-gate"
            job["depends"] = ["leaderboard-snapshot", *[f"{d}_{t}" for d,t in CANARIES]]
        elif job["name"] in {f"{d}_{t}" for d,t in CANARIES}:
            job["depends"] = [p for p in job["depends"] if p != "attendance-recovery-gate"]
        elif "attendance-recovery-gate" in job.get("depends", []):
            job["depends"] = ["readout-recovery-gate" if p == "attendance-recovery-gate" else p for p in job["depends"]]
    manifest["run_profile"] = PROFILE
    manifest["scope"] = "Fresh complete graph/search/readout procedure. Two recovery canaries before nineteen tasks; no remembered graph or score fills."
    validate_manifest(manifest)
    return manifest


def canary_gate(args):
    staged.verify_profile(args.output)
    panel = pd.read_csv(args.output / "baseline/selected-panel.csv")
    clients = panel.loc[panel.model.str.startswith("tabpfn-rel-client-")]
    model = sorted(clients.model.unique())[-1]
    reports = []
    for dataset, task in CANARIES:
        report = json.loads((args.output / f"tasks/{dataset}_{task}/results.json").read_text())
        opponent = float(clients.loc[clients.model.eq(model) & clients.dataset.eq(dataset) &
                                    clients.task.eq(task), "test_score"].iloc[0])
        score = report["test_score"]
        passed = score > opponent if report["metric"] == "roc_auc" else score < opponent
        reports.append(dict(dataset=dataset, task=task, score=score, opponent=opponent, passed=passed))
    passed = all(r["passed"] for r in reports)
    dump(args.output / "recovery-gate/results.json", dict(status="passed" if passed else "failed",
        reports=reports, opponent_model=model, affects_model_selection=False,
        scope="Operational acceptance only; never supplied to model selection"))
    if not passed:
        raise RuntimeError("Recovery not verified; the other nineteen tasks remain blocked")


def install_recipe():
    """Execution-only routing; legacy source snapshots and scripts stay intact."""
    staged.PROFILE = PROFILE
    staged.CONTRACT = {**staged.CONTRACT, "recovered_readouts":READOUT_CONTRACT}
    original_builder = staged.build_manifest
    staged.build_manifest = lambda *a, **kw: recovery_manifest(original_builder(*a, **kw))
    staged.gate = canary_gate
    original_task = fresh.task_main
    original_official_path = fresh.official_path
    original_official_result = fresh.official_result
    original_verify = staged.verify_stages

    def task_main(args):
        started = time.monotonic()
        total = fresh.remaining_task_seconds(args, f"{args.dataset}/{args.task}")
        code = original_task(args)
        if code:
            return code
        folder = readout_root(args.output, f"{args.dataset}/{args.task}")
        if not (folder.parent/"raw-audit/manifest.json").exists():
            retain_without_temporal_evidence(folder)
            return 0
        if not (folder / "results.json").exists():
            from scripts.run_kurversc_local_campaign import _run_guarded
            remaining = total - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Complete task deadline exhausted before recovered readouts")
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "stage.log").open("a", buffering=1) as log:
                code, resources = _run_guarded([sys.executable, "-u", "-m", "scripts.benchmark_recovered_readouts",
                    "--task-root", str(folder.parent), "--source-root", str(phase_source_root(args.dataset)),
                    "--output", str(folder), "--workers", str(READOUT_CONTRACT["teacher_workers"])],
                    cwd=fresh.ROOT, environment=__import__("os").environ.copy(), log=log,
                    minimum_available_memory_gb=0, poll_seconds=5, task_time_limit_seconds=remaining)
            dump(folder / "resources.json", resources)
            if code or not (folder / "results.json").exists():
                raise RuntimeError(f"Recovered readout stage incomplete: exit={code}")
        return code

    def official_path(root, key):
        p = readout_root(root, key) / "results.json"
        return p if p.exists() else original_official_path(root, key)

    def official_result(root, key):
        p = readout_root(root, key) / "results.json"
        return json.loads(p.read_text()) if p.exists() else original_official_result(root, key)

    def verify_stages(root, dataset, task):
        official, stages = original_verify(root, dataset, task)
        folder = readout_root(root, f"{dataset}/{task}")
        selected = json.loads((folder / "selection.json").read_text())
        explicit_fallback = (selected.get("fallback")=="insufficient_temporal_evidence"
            and selected["selected"]==selected["projection"]=="incumbent"
            and not selected["decisions"])
        if (not selected["complete_grid"] and not explicit_fallback) or selected["official_targets_used_for_selection"]:
            raise ValueError("Recovered readout lacked complete validation-only evidence")
        if selected["selected"] != official["selected"] or official["deployment_seed"] != 0:
            raise ValueError("Official procedure differs from the validation-locked readout")
        stages.update(recovered_readout_selected=selected["selected"], projection=selected["projection"],
                      recovered_readout_decisions=selected["decisions"], query_keys_sha256=official["query_keys_sha256"])
        return official, stages

    fresh.task_main, fresh.official_path, fresh.official_result = task_main, official_path, official_result
    staged.verify_stages = verify_stages


def main():
    install_recipe()
    staged.main()


if __name__ == "__main__":
    main()
