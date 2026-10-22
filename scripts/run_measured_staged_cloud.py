"""Execution-only full staged reproduction with durable per-task RAM evidence.

All 21 tasks run regardless of canary scores; no test outcome selects a model
or suppresses other tasks. The current recovered staged algorithm is unchanged.
Resources alone affect concurrency. Fresh checkpoints and official scores only.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts import run_gap_followup_queue as queue


def measured_manifest(original, output, reserve_gib):
    result = json.loads(json.dumps(original))
    gate_names = {"attendance-recovery-gate", "readout-recovery-gate"}
    result["jobs"] = [j for j in result["jobs"] if j["name"] not in gate_names]
    for job in result["jobs"]:
        job["depends"] = [d for d in job.get("depends", []) if d not in gate_names]
        if "/" in job.get("task", ""):
            job["command"] = ["{python}", "-u", "-m", "scripts.cloud_task_metrics", "--output",
                str(output / "metrics" / job["name"]), "--", *job["command"]]
    result.update(reserve_gib=reserve_gib,
        scope="All 21 fresh complete staged tasks; no score-dependent admission or carried-forward results",
        resource_policy="Up to 8 task workers; 64 GiB node reserve; no per-task hard RAM caps; serial checkpoint retries under pressure")
    result["jobs"].append(dict(name="resource-report", command=["{python}", "-m",
        "scripts.run_measured_staged_cloud", "--mode", "resources", "--output", str(output)],
        ram_gib=1, threads=1, wait_for_terminal=[j["name"] for j in result["jobs"]],
        result=str(output / "resource-summary.json")))
    return result


def validate(manifest):
    if manifest["reserve_gib"] < 64:
        raise ValueError("At least 64 GiB node headroom is required")
    compatible = {**manifest, "reserve_gib": max(128, manifest["reserve_gib"])}
    original_validator(compatible)


original_validator = queue.validate_manifest


def resources(output):
    state = json.loads((output / "queue/state.json").read_text())
    rows = []
    for key, task in state["jobs"].items():
        command = task.get("command", [])
        if "scripts.cloud_task_metrics" not in command:
            continue
        reports = [json.loads(p.read_text()) for p in sorted((output / "metrics" / key).glob("attempt-*/resources.json"))]
        if not reports:
            continue
        dataset, name = key.split("_", 1)
        rows.append(dict(dataset=dataset, task=name, status=task["status"], attempts=len(reports),
            elapsed_seconds=sum(r["elapsed_seconds"] for r in reports),
            peak_process_tree_rss_gib=max(r["peak_process_tree_rss_gib"] for r in reports),
            kernel_task_group_peak_gib=max(r.get("kernel_memory_peak_gib", r.get("kernel_memory_peak_bytes", 0)/2**30) for r in reports),
            final_resource_reports_complete=all("finished_at" in r for r in reports),
            sample_interval_seconds=1, per_task_hard_memory_limit="none"))
    if rows:
        with (output / "task-resource-summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    queue.dump(output / "resource-summary.json", dict(tasks=rows,
        generated_at=datetime.now(timezone.utc).isoformat(),
        rss_definition="1-second sum of RSS across the full descendant tree; shared pages may be counted multiple times",
        kernel_peak_definition="Exact cgroup-v2 memory.peak, including memory/file cache charged to that task scope",
        source_identity=str(output / "source-identity.json"),
        missing_task_reports=[k for k,v in state["jobs"].items() if "scripts.cloud_task_metrics" in v.get("command", [])
                              and k not in {r["dataset"]+"_"+r["task"] for r in rows}]))


def run_queue(output):
    queue.validate_manifest = validate
    original_admission = queue.admission_available
    serial = [False]

    def admission(available, active, external):
        active = list(active)
        return float("-inf") if serial[0] and active else original_admission(available, active, external)

    queue.admission_available = admission
    state_path = output / "queue/state.json"
    for attempt in range(3):
        queue.run(argparse.Namespace(manifest=output / "manifest.json", output=output / "queue", detach=False))
        state = json.loads(state_path.read_text())
        deferred = [v for v in state["jobs"].values() if v["status"] == "resource_deferred"]
        if not deferred or attempt == 2:
            resources(output)
            return
        for row in state["jobs"].values():
            if row.get("started_epoch"):
                row.setdefault("first_started_epoch", row["started_epoch"])
            if row["status"] in {"resource_deferred", "blocked"}:
                row["status"] = "queued"
                if row.get("first_started_epoch"):
                    row["started_epoch"] = row["first_started_epoch"]
        # Pressure retries use the SAME task checkpoints/model recipe and the
        # original first-admission deadline, but only one job is admitted.
        serial[0] = True
        state["resource_retry_round"] = attempt+1
        queue.dump(state_path, state)


def launch(args):
    from scripts import run_default_relarena_cloud as base
    from scripts import run_recovered_staged_cloud as recovered
    from scripts import run_staged_complete_cloud as staged

    recovered.install_recipe()
    pairs = base.task_pairs()
    identity, sources, cpus = base.initialize(args.output, pairs)
    sources = sorted(set(sources) | set((base.ROOT / "scripts").glob("*.py")) |
        {base.ROOT / "docs/checkpoints/20261007/frozen-graph-campaign-21.csv",
         base.ROOT / "scripts/cloud_runtime_constraints_20261016.txt"})
    identity.update(run_profile=staged.PROFILE, effective_runtime_config=staged.CONTRACT,
        source_sha256={str(p): queue.digest(p) for p in sources},
        score_dependent_task_admission=False, resource_measurement="Per-task Linux cgroup-v2 and 1-second process-tree RSS",
        scope="Fresh full current staged procedure, no old winners/predictions/score substitutions")
    queue.dump(args.output / "source-identity.json", identity)
    manifest = measured_manifest(staged.build_manifest(args.output, pairs, sources, args.max_workers, cpus),
                                 args.output, args.reserve_gib)
    validate(manifest)
    queue.dump(args.output / "manifest.json", manifest)
    folder = args.output / "queue"
    folder.mkdir(exist_ok=True)
    command = [sys.executable, "-u", "-m", "scripts.run_measured_staged_cloud", "--mode", "queue", "--output", str(args.output)]
    with (folder / "coordinator.log").open("a") as log:
        child = subprocess.Popen(command, cwd=base.ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    queue.dump(folder / "pid.json", dict(pid=child.pid, command=command))
    print(json.dumps(dict(pid=child.pid, output=str(args.output), max_workers=manifest["max_workers"],
                         reserve_gib=manifest["reserve_gib"], profile=staged.PROFILE)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("launch", "queue", "resources"), default="launch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--reserve-gib", type=float, default=64)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.max_workers < 1 or args.reserve_gib < 64:
        parser.error("Positive concurrency and at least 64 GiB headroom required")
    if args.mode == "launch":
        launch(args)
    elif args.mode == "queue":
        run_queue(args.output)
    else:
        resources(args.output)


if __name__ == "__main__":
    main()
