"""Detached, dependency-aware research queue; never modifies model defaults.

Admission accounts for expected *additional* RSS, already admitted jobs, and
an external-job reserve. Only children owned by this coordinator may be stopped.
Each experimental task shares a wall-time budget across all of its stages.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
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


TERMINAL = {"completed", "failed", "blocked", "resource_deferred", "time_budget_exceeded"}
RESOURCE_ENVIRONMENT = {"KURVERSC_DUCKDB_MEMORY_LIMIT", "KURVERSC_DUCKDB_MAX_TEMP_DIRECTORY_SIZE"}


def digest(path):
    # Historical wide-frame Parquets can be tens of GiB. Fingerprinting them
    # must not allocate a second full-file copy in an otherwise healthy task.
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def validate_manifest(manifest):
    jobs = manifest["jobs"]
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names) or not names:
        raise ValueError("Distinct explicit jobs required")
    if manifest["reserve_gib"] < 128 or not 0 < manifest["task_hours"] < 24:
        raise ValueError("At least 128 GiB reserve and a <24-hour task budget required")
    for job in jobs:
        environment = job.get("environment", {})
        if (not isinstance(environment, dict) or set(environment) - RESOURCE_ENVIRONMENT
                or any(not isinstance(v, str) or not v.strip() for v in environment.values())):
            raise ValueError("Job environment may contain only explicit DuckDB resource settings")
        if not job.get("external") and (not job.get("command") or job["ram_gib"] <= 0 or job["threads"] <= 0):
            raise ValueError("Runnable command and positive resource estimate required")
        if any(d not in names or d == job["name"] for d in [*job.get("depends", []), *job.get("wait_for_terminal", [])]):
            raise ValueError("Unknown/self dependency")
    remaining = set(names)
    while remaining:
        ready = {j["name"] for j in jobs if j["name"] in remaining
                 and not set([*j.get("depends", []), *j.get("wait_for_terminal", [])]) & remaining}
        if not ready:
            raise ValueError("Dependency cycle")
        remaining -= ready


def tree_rss(pid):
    try:
        root = psutil.Process(pid)
        return sum(p.memory_info().rss for p in [root, *root.children(recursive=True)]
                   if p.is_running()) / 2**30
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.


def admission_available(available, active, external_reserve):
    # Current free RAM already accounts for actual RSS. Reserve only the
    # unconsumed part of each admitted estimate, not its entire estimate again.
    return available - external_reserve - sum(max(0., item["estimate"] - item["rss"]) for item in active)


def external_growth_reserve(job, report):
    estimate = job.get("pending_reserve_gib", 0.)
    rss = report.get("worker_tree_rss_gib")
    def valid(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    if not valid(rss):
        rss = None
    if rss is None and isinstance(report.get("jobs"), dict):
        active = [entry for entry in report["jobs"].values() if entry.get("status") == "running"]
        if all(valid(entry.get("rss_gib")) for entry in active):
            rss = sum(entry["rss_gib"] for entry in active)
    if report.get("status") == "running" and rss is not None:
        # Actual external RSS already reduces host available RAM. Reserve
        # growth up to the estimate, rather than charging actual RSS twice.
        return max(0., estimate - rss)
    return estimate


def stop_owned(child, created):
    if child.poll() is not None:
        return
    try:
        root = psutil.Process(child.pid)
        if root.create_time() != created or root.ppid() != os.getpid():
            raise RuntimeError("Refusing to signal an unowned/reused PID")
        descendants = root.children(recursive=True)
        os.killpg(child.pid, signal.SIGTERM)
        for process in descendants:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs([root, *descendants], timeout=10)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        child.wait(timeout=10)
    except psutil.NoSuchProcess:
        child.wait()


def run(args):
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, "-u", "-m", "scripts.run_gap_followup_queue",
                   "--manifest", str(args.manifest.resolve()), "--output", str(args.output.resolve())]
        with (args.output / "coordinator.log").open("a") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        dump(args.output / "pid.json", {"pid": child.pid, "command": command})
        print(json.dumps({"pid": child.pid, "status_file": str(args.output / "state.json")}), flush=True)
        return 0
    lock = (args.output / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {"manifest_sha256": digest(args.manifest), "scheduler_sha256": digest(__file__),
                "sources": {p: digest(p) for p in manifest.get("sources", [])}}
    state_path = args.output / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["identity"] != identity:
            raise ValueError("Frozen queue changed; use a new output directory")
        for entry in state["jobs"].values():
            if entry["status"] == "running":
                # Do not silently duplicate an orphan after coordinator failure.
                raise RuntimeError("Unresolved running job: inspect its PID before resuming")
    else:
        state = {"identity": identity, "jobs": {j["name"]: {"status": "queued"} for j in manifest["jobs"]}}
    active = {}
    state.update(status="running", pid=os.getpid())

    def save():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["available_gib"] = psutil.virtual_memory().available / 2**30
        state["disk_free_gib"] = shutil.disk_usage(args.output).free / 2**30
        dump(state_path, state)

    def interrupt(number, _):
        raise KeyboardInterrupt(f"Stopping own queue only: signal {number}")

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    save()
    try:
        while True:
            pending_external_reserve = 0.
            for job in manifest["jobs"]:
                entry = state["jobs"][job["name"]]
                if not job.get("external") or entry["status"] in TERMINAL:
                    continue
                path = Path(job["external"]["status_file"])
                report = json.loads(path.read_text()) if path.exists() else {"status": "waiting_for_status_file"}
                phase = report.get("status")
                entry.update(status="external_waiting", external_status=phase, report=report)
                if phase in {"completed", "finished"}:
                    entry["status"] = "completed"
                elif phase in TERMINAL:
                    entry["status"] = "failed"
                else:
                    pending_external_reserve += external_growth_reserve(job, report)

            for name, item in list(active.items()):
                entry, child = state["jobs"][name], item["child"]
                item["rss"] = tree_rss(child.pid)
                entry["rss_gib"] = item["rss"]
                entry["peak_rss_gib"] = max(entry.get("peak_rss_gib", 0.), item["rss"])
                entry["elapsed_seconds"] = time.time() - entry["started_epoch"]
                if child.poll() is None and time.time() > item["deadline"]:
                    stop_owned(child, item["created"])
                    entry.update(status="time_budget_exceeded", reason="Shared task budget exhausted")
                if child.poll() is not None:
                    item["log"].close()
                    code = child.returncode
                    if entry["status"] == "running":
                        result = Path(item["job"]["result"])
                        entry.update(status="completed" if code == 0 and result.exists() else "failed",
                                     returncode=code, result=str(result))
                        if result.exists():
                            entry["summary"] = json.loads(result.read_text())
                    del active[name]

            available = psutil.virtual_memory().available / 2**30
            disk = shutil.disk_usage(args.output).free / 2**30
            if active and (available < manifest["reserve_gib"] or disk < 80):
                # A host-buffer violation is not a per-task RSS limit. Preserve
                # active work not owned by this coordinator, including engagement.
                name, item = max(active.items(), key=lambda pair: pair[1]["rss"])
                stop_owned(item["child"], item["created"])
                state["jobs"][name].update(status="resource_deferred", reason="Host RAM/disk buffer guard; checkpoints preserved")
                save()
                continue

            for job in manifest["jobs"]:
                entry = state["jobs"][job["name"]]
                if entry["status"] != "queued" or job.get("external"):
                    continue
                parents = [state["jobs"][d]["status"] for d in job.get("depends", [])]
                if any(s in TERMINAL - {"completed"} for s in parents):
                    entry.update(status="blocked", reason="Required preparation/control did not complete")
                    continue
                if any(s != "completed" for s in parents):
                    entry["wait_reason"] = "dependency"
                    continue
                if any(state["jobs"][d]["status"] not in TERMINAL for d in job.get("wait_for_terminal", [])):
                    entry["wait_reason"] = "prior_job_must_release_resources"
                    continue
                if len(active) >= manifest["max_workers"] or sum(i["job"]["threads"] for i in active.values()) + job["threads"] > manifest["max_threads"]:
                    entry["wait_reason"] = "concurrency_budget"
                    continue
                if job.get("exclusive") and active or any(i["job"].get("exclusive") for i in active.values()):
                    entry["wait_reason"] = "exclusive_resource_allocation"
                    continue
                capacity = psutil.virtual_memory().total / 2**30 - manifest["reserve_gib"]
                if job["ram_gib"] > capacity:
                    entry.update(status="resource_deferred", reason="Declared task allocation exceeds this host's safe capacity",
                        required_gib=job["ram_gib"]+manifest["reserve_gib"], host_gib=psutil.virtual_memory().total/2**30)
                    continue
                free = admission_available(psutil.virtual_memory().available / 2**30, active.values(), pending_external_reserve)
                if free < manifest["reserve_gib"] + job["ram_gib"] or disk < 100:
                    entry["wait_reason"] = "resource_admission"
                    continue
                if any(digest(p) != h for p, h in identity["sources"].items()):
                    raise ValueError("Frozen experiment implementation changed")
                task = job.get("task", job["name"])
                previous_starts = [e["started_epoch"] for j in manifest["jobs"]
                                   if j.get("task", j["name"]) == task
                                   and (e := state["jobs"][j["name"]]).get("started_epoch")]
                began = min(previous_starts, default=time.time())
                deadline = began + manifest["task_hours"] * 3600
                if time.time() >= deadline:
                    entry.update(status="time_budget_exceeded", reason="Shared task budget exhausted before admission")
                    continue
                folder = args.output / job["name"]
                folder.mkdir(exist_ok=True)
                temporary = folder / "physical-tmp"
                temporary.mkdir(exist_ok=True)
                environment = {k: v for k, v in os.environ.items() if not k.startswith("KURVERSC_")}
                environment.update(TMPDIR=str(temporary.resolve()), OMP_NUM_THREADS="1",
                                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                environment.update(job.get("environment", {}))
                command = [sys.executable if arg == "{python}" else arg for arg in job["command"]]
                log = (folder / "worker.log").open("a", buffering=1)
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=environment, start_new_session=True)
                active[job["name"]] = dict(child=child, created=psutil.Process(child.pid).create_time(),
                    deadline=deadline, log=log, job=job, estimate=job["ram_gib"], rss=0.)
                entry.update(status="running", pid=child.pid, started_epoch=time.time(),
                             command=command, estimated_ram_gib=job["ram_gib"], peak_rss_gib=0.)
                entry.pop("wait_reason", None)
                print(json.dumps({"started": job["name"], "pid": child.pid}), flush=True)
            save()
            if all(e["status"] in TERMINAL for e in state["jobs"].values()):
                failures = {name: entry["status"] for name, entry in state["jobs"].items()
                    if entry["status"] != "completed"}
                state.update(status="failed" if failures else "finished",
                    all_jobs_completed=not failures, incomplete_jobs=failures)
                save()
                return 1 if failures else 0
            time.sleep(5)
    except BaseException as error:
        for name, item in active.items():
            stop_owned(item["child"], item["created"])
            item["log"].close()
            state["jobs"][name].update(status="resource_deferred", reason="Own coordinator interrupted")
        state.update(status="interrupted", error=str(error))
        save()
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--detach", action="store_true")
    raise SystemExit(run(p.parse_args()))
