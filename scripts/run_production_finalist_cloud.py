"""Fresh full staged task with a bounded production-faithful finalist check.

Runs separately from the preserved suite/source tree. Search still discovers
its own candidates; validation alone chooses the graph. All existing temporal,
signal, recovered-readout and optional count admission stages are retained.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_gap_followup_queue import digest, dump, validate_manifest
from scripts.benchmark_production_finalists import CONTRACT as FINALIST_CONTRACT


PROFILE = "staged-complete-production-finalists-count-v1"


def graph_command(command):
    command = list(command)
    if "-m" not in command:
        return command
    index = command.index("-m") + 1
    if command[index] == "scripts.run_kurversc_selected_relarena":
        command[index:index+1] = ["scripts.run_production_finalist_cloud", "--mode", "graph", "--"]
    return command


def install_recipe():
    from scripts import run_count_challenger_cloud as count
    from scripts import run_kurversc_local_campaign as guards
    original_guard = guards._run_guarded

    def guarded(command, **kwargs):
        # The independent suite does not know about this extra one-task run.
        # Preserve host headroom for it at every guarded subprocess boundary.
        kwargs["minimum_available_memory_gb"] = max(128, kwargs.get("minimum_available_memory_gb", 0))
        return original_guard(graph_command(command), **kwargs)

    guards._run_guarded = guarded
    staged = count.install_recipe()
    staged.PROFILE = PROFILE
    staged.CONTRACT = {**staged.CONTRACT, "production_finalist_check":FINALIST_CONTRACT}
    return staged


def finalized_graph_row(original, decision, *, validation_report):
    """Never attach the old graph's official score to a new selected graph."""
    row = dict(original)
    row.update(config=json.dumps(decision.get("config", json.loads(original["config"])), sort_keys=True),
        production_finalist_decision=decision["selected"],
        production_finalist_validation=str(validation_report),
        audit_source="Fresh staged search followed by bounded production-faithful recent-origin finalist comparison")
    if decision["selected"] != "incumbent":
        for key in ("test_score", "improvement", "beats_published_kurversc", "plan_fingerprint",
                    "inner_validation_score", "inner_full_validation_score"):
            row[key] = None
        row["graph_official_score_pending"] = True
    return row


def graph_stage(arguments):
    from scripts.run_fresh_cloud_campaign import ROOT
    from scripts.run_kurversc_local_campaign import _run_guarded
    import pandas as pd

    def value(flag):
        return arguments[arguments.index(flag)+1]

    output = Path(value("--output")).resolve()
    dataset, task = value("--dataset"), value("--task")
    graph_root = output.parent.parent
    check = graph_root / "production-finalist-check"
    original_output = check / "search-stage" / output.name
    original_result = original_output.with_suffix(".json")
    started = time.monotonic()
    check.mkdir(parents=True, exist_ok=True)
    if not original_result.exists():
        search = list(arguments)
        search[search.index("--output")+1] = str(original_output)
        # The enclosing full-task process guard owns the cumulative deadline.
        code = subprocess.call([sys.executable, "-u", "-m",
            "scripts.run_kurversc_selected_relarena", *search], cwd=ROOT, env=dict(os.environ))
        if code or not original_result.exists():
            raise RuntimeError(f"Fresh graph search incomplete: {code}")
    validation = check / "validation"
    completion = validation / "results.json"
    if not completion.exists():
        with (check / "validation.log").open("a", buffering=1) as log:
            code, usage = _run_guarded([sys.executable, "-u", "-m", "scripts.benchmark_production_finalists",
                "--graph-root", str(graph_root), "--search-result", str(original_result),
                "--dataset", dataset, "--task", task, "--output", str(validation)],
                cwd=ROOT, environment=dict(os.environ), log=log, minimum_available_memory_gb=128,
                poll_seconds=5, task_time_limit_seconds=4*3600)
        dump(check / "validation-resources.json", usage)
        if code or not completion.exists():
            raise RuntimeError(f"Production-faithful finalist validation incomplete: {code}")
    decision = json.loads(completion.read_text())
    if "checkpoint" in decision and digest(decision["checkpoint"]) != decision["checkpoint_sha256"]:
        raise ValueError("Validation-selected fresh checkpoint changed")
    row = finalized_graph_row(json.loads(original_result.read_text()), decision, validation_report=completion)
    row["production_finalist_graph_stage_seconds"] = time.monotonic()-started
    # This is the first publication of the selected graph to downstream stages.
    # A crash before this point leaves resumable search/check caches, not an
    # apparently completed graph result that silently skipped the comparison.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    pd.DataFrame([row]).to_csv(temporary, index=False)
    temporary.replace(output)
    dump(output.with_suffix(".json"), row)
    print(json.dumps(dict(event="production_finalist_graph_selected", dataset=dataset,
        task=task, selected=decision["selected"], validation=str(completion))), flush=True)


def launch(args):
    from scripts import run_default_relarena_cloud as base
    staged = install_recipe()
    pair = (args.dataset, args.task)
    if pair not in base.task_pairs():
        raise ValueError("Official task required")
    identity, sources, cpus = base.initialize(args.output, [pair])
    sources = sorted(set(sources) | set((base.ROOT / "scripts").glob("*.py")) |
        set(Path(__file__).parent.glob("*.py")) |
        {base.ROOT / "docs/checkpoints/20261007/frozen-graph-campaign-21.csv"})
    identity.update(run_profile=PROFILE, effective_runtime_config=staged.CONTRACT,
        source_sha256={str(p):digest(p) for p in sources},
        scope="Fresh full staged task; bounded production-faithful finalist check; no reused graph, prediction or previous score")
    dump(args.output / "source-identity.json", identity)
    slug = f"{args.dataset}_{args.task}"
    def command(mode):
        return ["{python}", "-u", "-m", "scripts.run_production_finalist_cloud", "--mode", mode,
            "--dataset", args.dataset, "--task", args.task, "--output", str(args.output)]
    jobs = [dict(name="prepare", command=command("prepare"), ram_gib=2, threads=1,
        result=str(args.output / "datasets" / args.dataset / "results.json")),
        dict(name=slug, task=f"{args.dataset}/{args.task}", command=["{python}", "-u", "-m",
            "scripts.cloud_task_metrics", "--output", str(args.output / "metrics" / slug),
            "--", *command("task")], ram_gib=staged.ram_estimate(*pair), threads=8,
            depends=["prepare"], result=str(args.output / "tasks" / slug / "results.json"))]
    manifest = dict(sources=list(identity["source_sha256"]), reserve_gib=128,
        task_hours=23.75, max_workers=1, max_threads=cpus, jobs=jobs, run_profile=PROFILE)
    validate_manifest(manifest)
    dump(args.output / "manifest.json", manifest)
    queue = args.output / "queue"
    queue.mkdir()
    with (queue / "coordinator.log").open("a") as log:
        child = subprocess.Popen([sys.executable, "-u", "-m", "scripts.run_production_finalist_cloud",
            "--mode", "queue", "--dataset", args.dataset, "--task", args.task,
            "--output", str(args.output)], cwd=base.ROOT, env=dict(os.environ),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(queue / "pid.json", dict(pid=child.pid))
    print(json.dumps(dict(pid=child.pid, output=str(args.output), profile=PROFILE,
        launched_at=datetime.now(timezone.utc).isoformat())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("launch", "queue", "prepare", "task", "graph"), default="launch")
    parser.add_argument("--dataset")
    parser.add_argument("--task")
    parser.add_argument("--output", type=Path)
    args, remaining = parser.parse_known_args()
    if args.mode == "graph":
        return graph_stage(remaining[1:] if remaining and remaining[0] == "--" else remaining)
    if remaining or args.output is None or not args.dataset or not args.task:
        parser.error("An explicit task/output and no unrecognized arguments required")
    args.output = args.output.resolve()
    if args.mode == "launch":
        return launch(args)
    if args.mode == "queue":
        from scripts.run_measured_staged_cloud import run_queue
        return run_queue(args.output)
    staged = install_recipe()
    return {"prepare":staged.prepare, "task":staged.task}[args.mode](args)


if __name__ == "__main__":
    main()
