"""Fresh staged task plus a narrow validation-gated conditional-count readout.

Keep the preserved full staged recipe unchanged. Only its already-admitted
integer-count/frame-ensemble result can receive this extra challenger. Other
domains, fitting procedures and stronger optional readouts are exact fallbacks.
Fresh graph search remains schema-derived; no historical winner is imported.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.count_challenger_protocol import count_stage_eligible, validated_count_recipe
from scripts.run_gap_followup_queue import digest, dump, validate_manifest


PROFILE = "staged-complete-recovered-count-v1"
COUNT_CONTRACT = dict(optional=True, target_domain="mature nonnegative integer targets",
    comparator="already admitted integer-count projection", weights=[.125, .5, 1.],
    complete_origin_seed_grid=True, independent_native_replay=True,
    paired_recent_and_temporal_confidence=True, allowance=1024,
    original_sampler_and_member_identity=True, native_official_replay_required=True,
    graph_search_expanded=False, historical_winner_lookup=False,
    selection_rule="existing_production_temporal_veto", older_all_seed_relative_loss_cap=.01)


def count_root(root, key):
    slug = key.replace("/", "_")
    return root / "workers" / slug / "temporal/tasks" / slug / "conditional-count"


def install_recipe():
    from scripts import run_recovered_staged_cloud as recovered
    from scripts import run_staged_complete_cloud as staged
    from scripts import run_fresh_cloud_campaign as fresh
    from scripts.run_kurversc_local_campaign import _run_guarded

    recovered.install_recipe()
    staged.PROFILE = PROFILE
    staged.CONTRACT = {**staged.CONTRACT, "conditional_count_challenger":COUNT_CONTRACT}
    original_task = fresh.task_main
    old_path, old_result, old_verify = fresh.official_path, fresh.official_result, staged.verify_stages

    def task_main(args):
        started = time.monotonic()
        key = f"{args.dataset}/{args.task}"
        budget = fresh.remaining_task_seconds(args, key)
        code = original_task(args)
        if code:
            return code
        root = count_root(args.output, key)
        root.mkdir(parents=True, exist_ok=True)
        if (root / "results.json").exists():
            return 0
        task_root = root.parent
        def load(path):
            return json.loads(path.read_text()) if path.exists() else {}

        raw = load(task_root / "raw-audit/manifest.json")
        policy = load(task_root / "selection/results.json").get("selected_policy", {})
        selected = load(task_root / "recovered-readouts/selection.json")
        applicable = count_stage_eligible(raw, policy, selected)

        def stage(name, command, completion):
            if completion.exists():
                return
            remaining = budget - (time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError("Original complete-task deadline exhausted before count readout")
            with (root / f"{name}.log").open("a", buffering=1) as log:
                status, resources = _run_guarded(command, cwd=fresh.ROOT,
                    environment=dict(os.environ), log=log, minimum_available_memory_gb=0,
                    poll_seconds=5, task_time_limit_seconds=remaining)
            dump(root / f"{name}-resources.json", resources)
            if status or not completion.exists():
                raise RuntimeError(f"Count challenger stage {name} failed: {status}")

        choice = None
        if applicable:
            validation = root / "validation"
            stage("validation", [sys.executable, "-u", "-m", "scripts.benchmark_complete_count_challenger",
                "--task-root", str(task_root), "--output", str(validation), "--workers", "3"], validation / "results.json")
            choice = validated_count_recipe(json.loads((validation / "results.json").read_text()),
                json.loads((validation / "manifest.json").read_text()), raw["metadata"])
            if choice is not None:
                official = root / "official"
                stage("official", [sys.executable, "-u", "-m", "scripts.benchmark_validated_count_challenger",
                    "--task-root", str(task_root), "--validation", str(validation),
                    "--output", str(official)], official / "results.json")
                report = json.loads((official / "results.json").read_text())
        dump(root / "admission.json", dict(applicable=applicable, admitted=choice is not None,
            choice=choice, official_targets_used_for_selection=False))
        if choice is None:
            report = {**old_result(args.output, key), "count_challenger_retained_incumbent":True}
        dump(root / "results.json", report)
        return 0

    def official_path(root, key):
        path = count_root(root, key) / "results.json"
        return path if path.exists() else old_path(root, key)

    def official_result(root, key):
        path = count_root(root, key) / "results.json"
        return json.loads(path.read_text()) if path.exists() else old_result(root, key)

    def verify_stages(root, dataset, task):
        # The preserved verifier must audit ITS recovered stage, not compare
        # its selected name with the subsequent conditional-count head.
        fresh.official_result = old_result
        try:
            _, stages = old_verify(root, dataset, task)
        finally:
            fresh.official_result = official_result
        folder = count_root(root, f"{dataset}/{task}")
        admission = json.loads((folder / "admission.json").read_text())
        report = json.loads((folder / "results.json").read_text())
        if admission["admitted"]:
            if report["selected"] != admission["choice"]["selected"]:
                raise ValueError("Count official replay differs from validation-locked selection")
            audit = json.loads((folder / "official/control-replay-audit.json").read_text())
            if not audit["strongest_integer_control_exact"] or audit["native_max_error"] > 1e-9:
                raise ValueError("Count readout must reproduce both prior controls")
        stages["conditional_count"] = admission
        return report, stages

    fresh.task_main, fresh.official_path, fresh.official_result = task_main, official_path, official_result
    staged.verify_stages = verify_stages
    return staged


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
        scope="Fresh full staged task and validation-gated count challenger; no reused graph, prediction or prior score")
    dump(args.output / "source-identity.json", identity)
    slug = f"{args.dataset}_{args.task}"
    command = lambda mode: ["{python}", "-u", "-m", "scripts.run_count_challenger_cloud", "--mode", mode,
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
        child = subprocess.Popen([sys.executable, "-u", "-m", "scripts.run_count_challenger_cloud",
            "--mode", "queue", "--dataset", args.dataset, "--task", args.task,
            "--output", str(args.output)],
            cwd=base.ROOT, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    dump(queue / "pid.json", dict(pid=child.pid))
    print(json.dumps(dict(pid=child.pid, output=str(args.output), profile=PROFILE,
        launched_at=datetime.now(timezone.utc).isoformat())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("launch", "queue", "prepare", "task"), default="launch")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.mode == "launch":
        launch(args)
    elif args.mode == "queue":
        from scripts.run_measured_staged_cloud import run_queue
        run_queue(args.output)
    else:
        staged = install_recipe()
        {"prepare":staged.prepare, "task":staged.task}[args.mode](args)


if __name__ == "__main__":
    main()
