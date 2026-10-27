"""Measure a complete task tree in its own unlimited Linux cgroup-v2 scope.

RSS is sampled every second and sums descendants (shared pages may be counted
more than once). Kernel memory.peak is separate and includes charged file cache
and kernel memory. Neither number is a single-process ru_maxrss surrogate.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from scripts.run_gap_followup_queue import dump, stop_owned, tree_rss


def dedicated_cgroup(text, base=Path("/sys/fs/cgroup")):
    paths = [line.split("::", 1)[1] for line in text.splitlines() if line.startswith("0::")]
    if len(paths) != 1 or not paths[0].endswith(".scope") or "kurversc-task-" not in paths[0]:
        raise ValueError("A dedicated KurveRSC task scope is required; refusing host-level peaks")
    path = (base / paths[0].lstrip("/")).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Invalid task cgroup path")
    return path


def number(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def inside(args, command):
    group = dedicated_cgroup(Path("/proc/self/cgroup").read_text())
    if (group / "memory.max").read_text().strip() != "max":
        raise ValueError("This benchmark must not have a per-task hard RAM cap")
    folder = args.output / f"attempt-{time.time_ns()}-{os.getpid()}"
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "resources.json"
    start = time.monotonic()
    state = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
        command=command, monitor_pid=os.getpid(), cgroup=str(group), memory_max="max",
        sample_interval_seconds=1, peak_process_tree_rss_gib=0., kernel_memory_peak_bytes=0,
        peak_definition="Sampled descendant RSS sum; separate cgroup peak includes charged file cache/kernel memory")
    child = None

    def sample():
        rss = tree_rss(child.pid) if child else 0.
        state.update(elapsed_seconds=time.monotonic()-start,
            process_tree_rss_gib=rss,
            peak_process_tree_rss_gib=max(state["peak_process_tree_rss_gib"], rss),
            host_available_gib=psutil.virtual_memory().available/2**30)
        peak = number(group / "memory.peak")
        if peak is not None:
            state["kernel_memory_peak_bytes"] = max(state["kernel_memory_peak_bytes"], peak)
        state["kernel_memory_peak_gib"] = state["kernel_memory_peak_bytes"]/2**30
        return dict(elapsed_seconds=state["elapsed_seconds"], rss_gib=rss,
            cgroup_current_bytes=number(group / "memory.current"),
            cgroup_peak_bytes=peak, host_available_gib=state["host_available_gib"])

    def interrupted(signum, _):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise KeyboardInterrupt(f"Task monitor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    code = 1
    try:
        child = subprocess.Popen(command, env=dict(os.environ), start_new_session=True)
        created = psutil.Process(child.pid).create_time()
        state["task_pid"] = child.pid
        with (folder / "samples.jsonl").open("a", buffering=1) as stream:
            while child.poll() is None:
                stream.write(json.dumps(sample())+"\n")
                dump(path, state)
                time.sleep(1)
        code = child.wait()
        state.update(status="completed" if code == 0 else "failed", returncode=code)
    except BaseException as error:
        state.update(status="interrupted", error=str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            stop_owned(child, created)
        sample()
        state["kernel_memory_peak_gib"] = state["kernel_memory_peak_bytes"]/2**30
        events = group / "memory.events"
        state["memory_events"] = events.read_text() if events.exists() else None
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        dump(path, state)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Explicit benchmark command required")
    if args.inside:
        return inside(args, command)
    unit = f"kurversc-task-{time.time_ns()}-{os.getpid()}"
    return subprocess.call(["systemd-run", "--user", "--scope", "--quiet", "--collect",
        "--unit", unit, "--property=MemoryAccounting=yes", "--property=TasksMax=infinity",
        "--property=MemoryMax=infinity", sys.executable, "-u", "-m", "scripts.cloud_task_metrics",
        "--inside", "--output", str(args.output), "--", *command])


if __name__ == "__main__":
    raise SystemExit(main())
