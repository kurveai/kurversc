"""Cloud orchestration for fresh stable graph search plus temporal selection.

No old graph winners, predictions or hypothesis providers are imported. Each
task searches first, then runs the existing temporal-policy recipe against its
newly selected checkpoint. Only execution scheduling differs from the preserved
recipe. The host guard stops just the largest task tree under critical pressure.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
RSC_COMMIT = "5819ee84496255baae60b2e4565657041022ade9"
GR_COMMIT = "0c9c2dcc81b0a2f81a55c9667c3fbc5214b1a2fc"


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def official_path(root, key):
    slug = key.replace("/", "_")
    return root / "workers" / slug / "temporal/tasks" / slug / "official/results.json"


def official_result(root, key):
    """Only a validated official artifact is evidence of task completion."""
    path = official_path(root, key)
    campaign_path = root / "campaign-status.json"
    if campaign_path.exists() and json.loads(campaign_path.read_text()).get("identity", {}).get("signal_challengers", False):
        path = path.parent.parent / "signal-challengers/results.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
        dataset, task = key.split("/")
        if (result["dataset"], result["task"]) != (dataset, task):
            return None
        if result["metric"] not in {"mae", "roc_auc"} or not result["arm"]:
            return None
        if not all(math.isfinite(float(result["official_test_scores"][arm]))
                   for arm in ("control", "candidate")):
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def completion_status(root, key, returncode, stop_reason=None):
    # A cancelled Python worker can exit zero after handling SIGTERM. Never
    # mistake that for a successful task, even if a partial artifact exists.
    if stop_reason == "global_resource_pressure":
        return "resource_deferred"
    if stop_reason == "total_task_time_budget":
        return "time_budget_exceeded"
    if stop_reason or returncode != 0 or official_result(root, key) is None:
        return "failed"
    return "completed"


def remaining_task_seconds(args, key):
    record = json.loads((args.output / "campaign-status.json").read_text())["tasks"][key]
    return max(0, args.task_hours * 3600 - record.get("consumed_seconds", 0))


def duckdb_limit_for_task(total_bytes, isolated):
    # This is an engine spill threshold, NOT a task RSS limit. A single engine
    # otherwise defaults to 80% of host RAM, leaving too little room for pandas
    # and returned frames even when the high-memory task runs by itself.
    return f"{max(1, int(total_bytes / 1024**3 * 0.5))}GiB" if isolated else "none"


def reconcile_resume(root, state, pairs):
    """Keep verified results and account for interrupted active execution."""
    if set(state["tasks"]) != {f"{d}/{t}" for d, t in pairs}:
        raise ValueError("Resume task set changed")
    observed = state.get("resources", {}).get("observed_at", state["started_at"])
    stopped_at = datetime.fromisoformat(observed)
    for key, record in state["tasks"].items():
        if official_result(root, key) is not None:
            record["status"] = "completed"
            continue
        record.setdefault("consumed_seconds", record.get("elapsed_seconds", 0))
        if record["status"] == "running" and record.get("started_at"):
            record["consumed_seconds"] += max(
                0, (stopped_at - datetime.fromisoformat(record["started_at"])).total_seconds()
            )
        if record["status"] == "running":
            record.setdefault("attempts", []).append(dict(
                status="interrupted", pid=record.get("pid"),
                started_at=record.get("started_at"), observed_until=observed,
            ))
        record.update(status="queued", current_tree_rss_gib=0)
        for field in ("pid", "slot", "started_at", "stop_reason", "returncode", "finished_at"):
            record.pop(field, None)
    return state


def task_pairs():
    with (ROOT / "docs/checkpoints/20261007/frozen-graph-campaign-21.csv").open() as f:
        pairs = [(row["dataset"], row["task"]) for row in csv.DictReader(f)]
    if len(pairs) != 21 or len(set(pairs)) != 21:
        raise ValueError("Expected exactly 21 distinct tasks")
    # Interleave schemas to avoid starting ten large tasks on the same database.
    datasets = ["rel-f1", "rel-trial", "rel-event", "rel-stack", "rel-hm", "rel-amazon", "rel-avito"]
    buckets = {dataset: sorted(p for p in pairs if p[0] == dataset) for dataset in datasets}
    return [buckets[d][i] for i in range(4) for d in datasets if i < len(buckets[d])]


def graph_arguments(args):
    graph = args.output / "graphs"
    return [
        "--dataset", args.dataset, "--task", args.task,
        "--output", str(graph / "results" / f"{args.dataset}_{args.task}.csv"),
        "--checkpoint-root", str(graph / "checkpoints"),
        "--profile", "staged-retention", "--base-cdf", "on",
        "--peer-context-mode", "both", "--peer-max-history-rows", "50000",
        "--search-parallelism", str(args.search_workers),
        "--root-reachable-materialization", "--no-rerank-nested-depth-safety",
        "--full-training-frames", "1", "--seed", "0",
        "--duckdb-memory-limit", "none", "--duckdb-temp-limit", "none",
    ]


def dispatch(command, graph, seed_workers=2):
    command = list(command)
    index = command.index("-m") + 1
    module = command[index]
    if module == "scripts.benchmark_temporal_policy":
        command.append("--prune-vetoed")
    elif module == "scripts.benchmark_supported_official":
        module = "scripts.run_official_cloud"
        command.extend(["--threads", "8", "--duckdb-threads", "8"])
    command[index:index + 1] = [
        "scripts.run_fresh_cloud_campaign", "--mode", "stage",
        "--graph-root", str(graph), "--stage-module", module, "--",
    ]
    return command


def stage_main(args, remaining):
    allowed = {
        "scripts.benchmark_sparse_training", "scripts.materialize_policy_folds",
        "scripts.benchmark_temporal_policy", "scripts.run_temporal_audit_parallel", "scripts.reselect_temporal_policy",
        "scripts.run_official_cloud",
        "scripts.benchmark_signal_challengers",
    }
    if args.stage_module not in allowed:
        raise ValueError("Unexpected stage module")
    from scripts import benchmark_frozen_event_summaries as frozen
    import kurversc.core as core

    frozen.CAMPAIGN = args.graph_root.resolve()
    original_scope = core._connection_scope

    @contextmanager
    def unbounded_scope(connection, *, memory_limit=None, max_temp_directory_size=None):
        # Original standalone stage parsers take strings; normalize their
        # "none" sentinel before handing resource arguments to DuckDB.
        def limit(value):
            return None if isinstance(value, str) and value.lower() in {"none", "null", "unlimited"} else value

        engine_limit = (limit(os.environ["KURVERSC_DUCKDB_MEMORY_LIMIT"])
            if "KURVERSC_DUCKDB_MEMORY_LIMIT" in os.environ else limit(memory_limit))
        temp_limit = (limit(os.environ["KURVERSC_DUCKDB_MAX_TEMP_DIRECTORY_SIZE"])
            if "KURVERSC_DUCKDB_MAX_TEMP_DIRECTORY_SIZE" in os.environ else limit(max_temp_directory_size))
        with original_scope(connection, memory_limit=engine_limit,
                            max_temp_directory_size=temp_limit) as con:
            yield con

    core._connection_scope = unbounded_scope
    sys.argv = [args.stage_module, *remaining]
    return importlib.import_module(args.stage_module).main()


def task_main(args):
    from scripts import run_temporal_policy_campaign as campaign
    from scripts.run_kurversc_local_campaign import _run_guarded

    started = time.monotonic()
    slug = f"{args.dataset}_{args.task}"
    key = f"{args.dataset}/{args.task}"
    budget_seconds = remaining_task_seconds(args, key)
    if budget_seconds <= 0:
        raise TimeoutError("Total task budget already exhausted")
    folder = args.output / "workers" / slug
    folder.mkdir(parents=True, exist_ok=True)
    graph_result = args.output / "graphs/results" / f"{slug}.json"
    if not graph_result.exists():
        with (folder / "search.log").open("a", buffering=1) as log:
            code, resources = _run_guarded(
                [sys.executable, "-u", "-m", "scripts.run_kurversc_selected_relarena", *graph_arguments(args)],
                cwd=ROOT, environment=dict(os.environ), log=log,
                minimum_available_memory_gb=0, poll_seconds=5,
                task_time_limit_seconds=budget_seconds,
            )
        atomic_json(resources, folder / "search-resources.json")
        if code or not graph_result.exists():
            raise RuntimeError(f"Fresh search failed: exit={code}; resources={resources}")
    budget = (budget_seconds - (time.monotonic() - started)) / 3600
    if budget <= 0:
        raise TimeoutError("Fresh search used the total task time budget")
    campaign.SOURCE_CAMPAIGN = args.output / "graphs"
    campaign.campaign_tasks = lambda _: [(args.dataset, args.task)]
    original_guard = campaign._run_guarded

    def guarded(command, **kwargs):
        # Central coordinator owns memory-pressure interventions; not ten
        # independent guards simultaneously killing otherwise healthy tasks.
        kwargs["minimum_available_memory_gb"] = 0
        kwargs["task_time_limit_seconds"] = min(
            kwargs["task_time_limit_seconds"],
            max(0, budget_seconds - (time.monotonic() - started)),
        )
        return original_guard(dispatch(command, campaign.SOURCE_CAMPAIGN, args.seed_workers), **kwargs)

    campaign._run_guarded = guarded
    temporal_manifest = folder / "temporal/campaign-manifest.json"
    if temporal_manifest.exists():
        previous = json.loads(temporal_manifest.read_text())
        # The existing immutable campaign identity includes its original time
        # allowance. Retain it while the outer guard enforces the remaining
        # cumulative budget across the interrupted attempts.
        budget = previous["identity"]["task_hours"]
        task_state = previous["tasks"][key]
        if task_state["status"] != "completed":
            task_state["status"] = "queued"
        atomic_json(previous, temporal_manifest)
    config = argparse.Namespace(
        output=folder / "temporal", adopt_active_task=None,
        task_hours=budget, origins=3, threads=4,
        minimum_available_gib=8, minimum_disk_gib=20,
        duckdb_memory_limit="none", duckdb_temp_limit="none",
    )
    code = campaign.run(config)
    manifest = json.loads((config.output / "campaign-manifest.json").read_text())
    if any(state["status"] != "completed" for state in manifest["tasks"].values()):
        raise RuntimeError(f"Temporal stage incomplete: {manifest['tasks']}")
    if getattr(args, "signal_challengers", False):
        task_root = config.output / "tasks" / slug
        complete = task_root / "signal-challengers/results.json"
        if not complete.exists():
            remaining = max(0, budget_seconds - (time.monotonic() - started))
            if remaining <= 0:
                raise TimeoutError("No budget remains for optional signal selection")
            with (folder / "signal-challengers.log").open("a", buffering=1) as log:
                command = dispatch([sys.executable, "-u", "-m", "scripts.benchmark_signal_challengers",
                    "--task-root", str(task_root), "--output", str(complete.parent), "--threads", "4"],
                    campaign.SOURCE_CAMPAIGN, args.seed_workers)
                code, resources = _run_guarded(command, cwd=ROOT, environment=dict(os.environ), log=log,
                    minimum_available_memory_gb=0, poll_seconds=5, task_time_limit_seconds=remaining)
            atomic_json(resources, folder / "signal-challenger-resources.json")
            if code or not complete.exists():
                raise RuntimeError(f"Optional signal selection incomplete: exit={code}; resources={resources}")
    return code


def tree_memory(pid):
    try:
        root = psutil.Process(pid)
        nodes = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0, []
    rss = 0
    live = []
    for process in nodes:
        try:
            rss += process.memory_info().rss
            live.append(process)
        except psutil.Error:
            continue
    return rss / 1024**3, live


def protected_campaign_busy(path, *, isolated=False):
    """Do not overlap high-memory isolated waves from independent campaigns."""
    if path is None:
        return False
    try:
        state = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return True  # Fail closed, not a guess about another campaign's RAM.
    active = [r for r in state.get("tasks", {}).values() if r["status"] == "running"]
    return bool(active) if isolated else any(r.get("isolated", False) for r in active)


def stop_task(pid):
    _, processes = tree_memory(pid)
    # Stage workers deliberately own their own process groups. Stop every
    # verified descendant, rather than leaving those groups orphaned.
    for process in reversed(processes):
        try:
            process.send_signal(signal.SIGTERM)
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass


def prepare_dataset(dataset, pairs):
    from relbench.datasets import download_dataset
    from relbench.tasks import download_task

    print(json.dumps(dict(event="download_started", dataset=dataset)), flush=True)
    download_dataset(dataset)
    for _, task in pairs:
        download_task(dataset, task)
    print(json.dumps(dict(event="download_complete", dataset=dataset)), flush=True)


def summarize(root, state):
    rows = []
    for key, task in state["tasks"].items():
        if task["status"] != "completed":
            continue
        dataset, name = key.split("/")
        result = official_result(root, key)
        if result is None:
            print(json.dumps(dict(event="missing_completion_artifact", task=key)), flush=True)
            continue
        rows.append(dict(
            dataset=dataset, task=name, metric=result["metric"],
            test_score=result["official_test_scores"]["candidate"],
            control_score=result["official_test_scores"]["control"],
            selected_policy=result["arm"], elapsed_seconds=task.get("elapsed_seconds", 0),
            peak_tree_rss_gib=task["peak_tree_rss_gib"],
        ))
    temporary = root / "campaign-results.csv.tmp"
    with temporary.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "task", "metric", "test_score", "control_score", "selected_policy", "elapsed_seconds", "peak_tree_rss_gib"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "campaign-results.csv")


def campaign_main(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "scheduler.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pairs = task_pairs()
    path = args.output / "campaign-status.json"
    state = dict(
        status="preparing", started_at=now(), scheduler_pid=os.getpid(),
        identity=dict(
            kurversc_commit=RSC_COMMIT, graphreduce_commit=GR_COMMIT,
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            task_parallelism=args.task_workers, search_parallelism=args.search_workers,
            seed_parallelism=args.seed_workers, task_hours=args.task_hours,
            profile="staged-retention", base_cdf=True, peer_context="both",
            full_training_frames=1, temporal_origins=3, temporal_seeds=[0, 1, 2],
            duckdb_memory_limit=None, duckdb_temp_limit=None,
            signal_challengers=getattr(args, "signal_challengers", False),
            scope="Fresh graph search then unchanged temporal selection, with explicit optional signal challengers"
                  if getattr(args, "signal_challengers", False) else
                  "Fresh graph search then unchanged generic temporal-policy selection; no local hypotheses",
        ),
        tasks={f"{d}/{t}": dict(status="queued", peak_tree_rss_gib=0) for d, t in pairs},
    )
    if getattr(args, "signal_challengers", False):
        from scripts.benchmark_signal_challengers import implementation_digest
        state["identity"]["signal_implementation_sha256"] = implementation_digest()
    if path.exists():
        if not args.resume:
            raise ValueError("Existing campaign requires explicit --resume")
        previous = json.loads(path.read_text())
        before, after = previous["identity"], state["identity"]
        for field in ("kurversc_commit", "graphreduce_commit", "search_parallelism", "seed_parallelism",
                      "task_hours", "profile", "base_cdf", "peer_context", "full_training_frames",
                      "temporal_origins", "temporal_seeds"):
            if before[field] != after[field]:
                raise ValueError(f"Resume refused: recipe changed: {field}")
        if before.get("signal_challengers", False) != after.get("signal_challengers", False):
            raise ValueError("Resume refused: optional challenger setting changed")
        if before.get("signal_implementation_sha256") != after.get("signal_implementation_sha256"):
            raise ValueError("Resume refused: optional signal implementation changed")
        # A scheduler lock prevents duplicate coordinators; verify that none
        # of the old task interpreters survived outside that lock as well.
        for record in previous["tasks"].values():
            if record.get("pid"):
                try:
                    command = psutil.Process(record["pid"]).cmdline()
                    if "scripts.run_fresh_cloud_campaign" in command and str(args.output) in command:
                        raise RuntimeError("A saved task worker is still alive; resume refused")
                except psutil.Error:
                    pass
        atomic_json(previous, args.output / f"campaign-status-before-resume-{time.time_ns()}.json")
        state = reconcile_resume(args.output, previous, pairs)
        state.setdefault("execution_revisions", []).append(dict(
            resumed_at=now(), scheduler_pid=os.getpid(), wrapper_sha256=after["wrapper_sha256"],
            isolated_tasks=args.isolated_task, isolated_search_workers=args.isolated_search_workers,
            admission_ram_gib=args.admission_ram_gib, critical_ram_gib=args.critical_ram_gib,
        ))
        state["scheduler_pid"] = os.getpid()
    else:
        if args.resume:
            raise ValueError("Cannot resume a missing campaign")
        atomic_json(state, path)
        datasets = sorted({dataset for dataset, _ in pairs})
        with ThreadPoolExecutor(max_workers=7) as pool:
            list(pool.map(lambda d: prepare_dataset(d, [p for p in pairs if p[0] == d]), datasets))
    state["status"] = "running"
    isolated_keys = set(args.isolated_task)
    if not isolated_keys <= set(state["tasks"]):
        raise ValueError("Unknown isolated task")
    unfinished = [pair for pair in pairs if state["tasks"]["/".join(pair)]["status"] != "completed"]
    queue = [pair for pair in unfinished if "/".join(pair) not in isolated_keys]
    isolated_queue = [pair for pair in unfinished if "/".join(pair) in isolated_keys]
    for dataset, task in isolated_queue:
        state["tasks"][f"{dataset}/{task}"]["status"] = "queued_isolated"
    active = {}
    summarize(args.output, state)
    atomic_json(state, path)
    cpus = psutil.Process().cpu_affinity() if hasattr(psutil.Process(), "cpu_affinity") else list(range(os.cpu_count() or 1))
    cores_per_task = max(1, len(cpus) // args.task_workers)
    low_memory_polls = 0
    while queue or isolated_queue or active:
        available = psutil.virtual_memory().available / 1024**3
        free_disk = shutil.disk_usage(args.output).free / 1024**3
        for key, entry in list(active.items()):
            process, log, slot, started = entry
            rss, _ = tree_memory(process.pid)
            record = state["tasks"][key]
            record["peak_tree_rss_gib"] = max(record["peak_tree_rss_gib"], rss)
            record["current_tree_rss_gib"] = rss
            consumed = record.get("consumed_seconds", 0)
            if process.poll() is None and consumed + time.monotonic() - started > args.task_hours * 3600:
                record["stop_reason"] = "total_task_time_budget"
                atomic_json(state, path)
                stop_task(process.pid)
            if process.poll() is not None:
                log.close()
                status = completion_status(args.output, key, process.returncode, record.get("stop_reason"))
                elapsed = time.monotonic() - started
                record.setdefault("attempts", []).append(dict(
                    returncode=process.returncode, status=status, elapsed_seconds=elapsed,
                    stop_reason=record.get("stop_reason"), started_at=record["started_at"],
                    finished_at=now(),
                ))
                record.update(status=status, returncode=process.returncode,
                              finished_at=now(), elapsed_seconds=consumed + elapsed,
                              consumed_seconds=consumed + elapsed)
                del active[key]
                print(json.dumps(dict(event="task_finished", task=key, **record)), flush=True)
                # One automatic resource retry, isolated from all other tasks.
                if status == "resource_deferred" and record.get("resource_retries", 0) < 1:
                    record["resource_retries"] = record.get("resource_retries", 0) + 1
                    record["status"] = "queued_isolated"
                    isolated_keys.add(key)
                    isolated_queue.append(tuple(key.split("/")))
                atomic_json(state, path)
                summarize(args.output, state)
        critical = (getattr(args, "isolated_critical_ram_gib", args.critical_ram_gib)
                    if len(active) == 1 and all(k in isolated_keys for k in active)
                    else args.critical_ram_gib)
        low_memory_polls = low_memory_polls + 1 if available < critical else 0
        if active and (low_memory_polls >= 2 or available < 128 or free_disk < 30):
            offender = max(active, key=lambda k: state["tasks"][k]["current_tree_rss_gib"])
            state["tasks"][offender]["stop_reason"] = "global_resource_pressure"
            atomic_json(state, path)
            print(json.dumps(dict(event="global_guard", task=offender, available_gib=available, free_disk_gib=free_disk)), flush=True)
            stop_task(active[offender][0].pid)
            low_memory_polls = 0
            # Do not admit new tasks on the pre-cancellation memory reading.
            available = psutil.virtual_memory().available / 1024**3
        used_slots = {entry[2] for entry in active.values()}
        if not queue and not active and isolated_queue:
            queue.append(isolated_queue.pop(0))
        isolated_active = any(key in isolated_keys for key in active)
        while queue and len(active) < args.task_workers and available >= args.admission_ram_gib and free_disk >= 160:
            dataset, task = queue.pop(0)
            key, slug = f"{dataset}/{task}", f"{dataset}_{task}"
            if protected_campaign_busy(getattr(args, "protected_campaign", None), isolated=key in isolated_keys):
                queue.insert(0, (dataset, task))
                break
            if isolated_active or (key in isolated_keys and active):
                queue.insert(0, (dataset, task))
                break
            record = state["tasks"][key]
            if record.get("consumed_seconds", 0) >= args.task_hours * 3600:
                record["status"] = "time_budget_exceeded"
                continue
            slot = next(i for i in range(args.task_workers) if i not in used_slots)
            used_slots.add(slot)
            folder = args.output / "workers" / slug
            folder.mkdir(parents=True, exist_ok=True)
            log = (folder / "worker.log").open("a", buffering=1)
            command = [sys.executable, "-u", "-m", "scripts.run_fresh_cloud_campaign", "--mode", "task",
                       "--output", str(args.output), "--dataset", dataset, "--task", task,
                       "--search-workers", str(args.isolated_search_workers if key in isolated_keys else args.search_workers),
                       "--seed-workers", str(args.seed_workers),
                       "--task-hours", str(args.task_hours)]
            if getattr(args, "signal_challengers", False):
                command.append("--signal-challengers")
            engine_limit = duckdb_limit_for_task(psutil.virtual_memory().total, key in isolated_keys)
            process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                       KURVERSC_DUCKDB_MEMORY_LIMIT=engine_limit, KURVERSC_DUCKDB_MAX_TEMP_DIRECTORY_SIZE="none"),
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                psutil.Process(process.pid).cpu_affinity(cpus[slot * cores_per_task:(slot + 1) * cores_per_task])
            except (psutil.Error, AttributeError):
                pass
            active[key] = (process, log, slot, time.monotonic())
            record.pop("stop_reason", None)
            record.update(status="running", pid=process.pid, slot=slot, started_at=now(),
                          isolated=key in isolated_keys,
                          duckdb_spill_threshold=engine_limit,
                          search_workers=args.isolated_search_workers if key in isolated_keys else args.search_workers)
            if key in isolated_keys:
                isolated_active = True
            print(json.dumps(dict(event="task_started", task=key, pid=process.pid, slot=slot)), flush=True)
        state["resources"] = dict(available_ram_gib=available, free_disk_gib=free_disk,
                                  active_tasks=len(active), queued_tasks=len(queue) + len(isolated_queue),
                                  isolated_queue=["/".join(pair) for pair in isolated_queue], observed_at=now())
        atomic_json(state, path)
        time.sleep(args.poll_seconds)
    state.update(status="finished", finished_at=now())
    atomic_json(state, path)
    summarize(args.output, state)
    return int(any(task["status"] != "completed" for task in state["tasks"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("campaign", "task", "stage"), default="campaign")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--task")
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--stage-module")
    parser.add_argument("--task-workers", type=int, default=10)
    parser.add_argument("--search-workers", type=int, default=2)
    parser.add_argument("--seed-workers", type=int, default=2)
    parser.add_argument("--task-hours", type=float, default=23)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--signal-challengers", action="store_true",
                        help="Opt-in bounded text/exposure/support challengers after unchanged temporal selection")
    parser.add_argument("--isolated-task", action="append", default=[])
    parser.add_argument("--isolated-search-workers", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--admission-ram-gib", type=float, default=384)
    parser.add_argument("--critical-ram-gib", type=float, default=320)
    parser.add_argument("--isolated-critical-ram-gib", type=float,
                        help="Optional lower host reserve only when one known heavy task runs alone")
    parser.add_argument("--protected-campaign", type=Path,
                        help="Avoid overlapping isolated high-memory workers in another campaign")
    args, remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    if args.mode == "stage":
        if not args.graph_root or not args.stage_module:
            parser.error("Stage requires a fresh graph root and module")
        return stage_main(args, remaining)
    if remaining or not args.output or not 0 < args.task_hours < 24:
        parser.error("Require output, no unknown arguments and a positive <24h task budget")
    if min(args.task_workers, args.search_workers, args.seed_workers, args.isolated_search_workers) < 1 or args.poll_seconds <= 0:
        parser.error("Worker counts must be positive")
    if args.mode == "task":
        if (args.dataset, args.task) not in task_pairs():
            parser.error("Unknown task")
        return task_main(args)
    if args.critical_ram_gib < 8 or args.admission_ram_gib < args.critical_ram_gib:
        parser.error("RAM admission reserve must exceed the critical guard")
    if args.isolated_critical_ram_gib is None:
        args.isolated_critical_ram_gib = args.critical_ram_gib
    if not 128 <= args.isolated_critical_ram_gib <= args.critical_ram_gib:
        parser.error("Isolated reserve must be at least 128 GiB and no greater than the normal reserve")
    return campaign_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
