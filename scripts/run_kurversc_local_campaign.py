#!/usr/bin/env python3
"""Run the reproducible KurveRSC RelArena campaign within a 48 GB envelope.

Tasks are deliberately sequential.  Parallelism is spent inside each graph
search and is reduced as the cached database grows; a sustained-memory guard
retries an interrupted task at half the prior worker count.  Every task has an
independent durable checkpoint and is skipped after an atomic result appears.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SCRIPT_ROOT = Path(__file__).resolve().parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from kurversc.checkpointing import search_runtime_identity  # noqa: E402

from run_kurversc_selected_relarena import PUBLISHED_REFERENCES  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover - optional operational guard
    psutil = None


def _task_token(value: str) -> tuple[str, str]:
    try:
        dataset, task = value.split("/", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tasks must use DATASET/TASK") from exc
    key = (dataset, task)
    if key not in PUBLISHED_REFERENCES:
        raise argparse.ArgumentTypeError(f"unknown published task: {value}")
    return key


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _aggregate(output_root: Path, tasks: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for dataset, task in tasks:
        result = output_root / "results" / f"{dataset}_{task}.json"
        if result.exists():
            rows.append(json.loads(result.read_text()))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["dataset", "task"]).reset_index(drop=True)
    temporary = output_root / "campaign-results.csv.tmp"
    frame.to_csv(temporary, index=False)
    temporary.replace(output_root / "campaign-results.csv")
    return frame


def _cache_root(cache_dir: str | None) -> Path:
    return (
        Path(cache_dir).expanduser()
        if cache_dir
        else Path.home() / "Library" / "Caches" / "relbench"
    )


def _dataset_bytes(dataset: str, *, cache_dir: str | None) -> int:
    root = _cache_root(cache_dir) / dataset
    if not root.exists():
        return 2**63 - 1
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _task_parallelism(
    dataset_bytes: int,
    *,
    maximum: int,
) -> int:
    """Choose a conservative within-task worker count from local data scale."""

    if maximum < 1:
        raise ValueError("search_parallelism must be positive")
    if dataset_bytes <= 64 * 1024**2:
        return min(maximum, 4)
    if dataset_bytes <= 1024**3:
        return min(maximum, 2)
    return 1


def _run_guarded(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log: Any,
    minimum_available_memory_gb: float,
    poll_seconds: float,
    task_time_limit_seconds: float,
) -> tuple[int, dict[str, Any]]:
    """Run one task and interrupt only that task after sustained memory pressure."""

    portable = bool(environment.get("KURVERSC_TASK_CONTEXT"))
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        # A public portable supervisor owns the entire task group. Nested
        # stage sessions would escape its RAM/deadline cleanup. Preserve the
        # original independent groups for the benchmark-only path.
        start_new_session=not portable,
    )

    def stop_process_group() -> None:
        for interrupt, grace_seconds in (
            (signal.SIGINT, 60),
            (signal.SIGTERM, 30),
            (signal.SIGKILL, 10),
        ):
            if process.poll() is not None:
                return
            if portable:
                process.send_signal(interrupt)
            else:
                os.killpg(process.pid, interrupt)
            try:
                process.wait(timeout=grace_seconds)
                return
            except subprocess.TimeoutExpired:
                continue

    peak_tree_rss = 0
    minimum_available = None
    low_memory_polls = 0
    guard_triggered = False
    time_limit_triggered = False
    started = time.monotonic()
    try:
        while process.poll() is None:
            if time.monotonic() - started >= task_time_limit_seconds:
                time_limit_triggered = True
                stop_process_group()
                break
            if psutil is not None:
                # System headroom is the safety-critical signal.  Keep it
                # independent from process-tree accounting because hardened
                # macOS/Linux environments may allow virtual_memory() while
                # denying inspection of another process.  An RSS permission
                # error must never silently disable the memory guard.
                try:
                    available = psutil.virtual_memory().available
                    minimum_available = (
                        available
                        if minimum_available is None
                        else min(minimum_available, available)
                    )
                    threshold = minimum_available_memory_gb * 1024**3
                    low_memory_polls = (
                        low_memory_polls + 1 if available < threshold else 0
                    )
                    if low_memory_polls >= 2:
                        guard_triggered = True
                        stop_process_group()
                        break
                except (psutil.Error, ProcessLookupError, PermissionError):
                    pass
                try:
                    root = psutil.Process(process.pid)
                    tree_rss = root.memory_info().rss
                    for child in root.children(recursive=True):
                        try:
                            if child.is_running():
                                tree_rss += child.memory_info().rss
                        except (psutil.Error, ProcessLookupError, PermissionError):
                            continue
                    peak_tree_rss = max(peak_tree_rss, tree_rss)
                except (psutil.Error, ProcessLookupError, PermissionError):
                    pass
            time.sleep(poll_seconds)
    except BaseException:
        stop_process_group()
        raise
    returncode = process.wait()
    return returncode, {
        "peak_tree_rss_gb": peak_tree_rss / 1024**3,
        "minimum_system_available_gb": (
            None if minimum_available is None else minimum_available / 1024**3
        ),
        "memory_guard_triggered": guard_triggered,
        "time_limit_triggered": time_limit_triggered,
        "elapsed_seconds": time.monotonic() - started,
    }


def run(args: argparse.Namespace) -> int:
    if args.peer_max_history_rows < 1:
        raise ValueError("peer_max_history_rows must be positive")
    if args.minimum_available_memory_gb < 0:
        raise ValueError("minimum_available_memory_gb must be non-negative")
    if args.memory_poll_seconds <= 0:
        raise ValueError("memory_poll_seconds must be positive")
    if not 0 < args.task_time_limit_hours < 24:
        raise ValueError("task_time_limit_hours must be between zero and 24")
    output_root = Path(args.output_root).expanduser().resolve()
    results_root = output_root / "results"
    logs_root = output_root / "logs"
    checkpoint_root = output_root / "checkpoints"
    for path in (output_root, results_root, logs_root, checkpoint_root):
        path.mkdir(parents=True, exist_ok=True)

    dataset_sizes = {
        dataset: _dataset_bytes(dataset, cache_dir=args.cache_dir)
        for dataset, _task in PUBLISHED_REFERENCES
    }
    tasks = args.tasks or sorted(
        PUBLISHED_REFERENCES,
        key=lambda key: (dataset_sizes[key[0]], key[0], key[1]),
    )
    manifest_path = output_root / "campaign-manifest.json"
    manifest = {
        "format_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "profile": args.profile,
        "base_cdf": args.base_cdf,
        "peer_context_mode": args.peer_context_mode,
        "peer_max_history_rows": args.peer_max_history_rows,
        "root_reachable_materialization": args.root_reachable_materialization,
        "rerank_nested_depth_safety": args.rerank_nested_depth_safety,
        "seed": args.seed,
        "search_parallelism": args.search_parallelism,
        "task_parallelism": {
            f"{dataset}/{task}": _task_parallelism(
                dataset_sizes[dataset], maximum=args.search_parallelism
            )
            for dataset, task in tasks
        },
        "dataset_size_gb": {
            dataset: size / 1024**3 for dataset, size in dataset_sizes.items()
        },
        "duckdb_memory_limit": args.duckdb_memory_limit,
        "duckdb_temp_limit": args.duckdb_temp_limit,
        "minimum_available_memory_gb": args.minimum_available_memory_gb,
        "task_time_limit_hours": args.task_time_limit_hours,
        "tasks": [f"{dataset}/{task}" for dataset, task in tasks],
        "runtime": search_runtime_identity(),
        "status": "running",
        "completed": [],
        "failed": [],
        "resources": {},
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        resume_keys = (
            "profile",
            "base_cdf",
            "peer_context_mode",
            "peer_max_history_rows",
            "root_reachable_materialization",
            "rerank_nested_depth_safety",
            "seed",
            "tasks",
            "runtime",
        )
        mismatched = [
            key for key in resume_keys if previous.get(key) != manifest.get(key)
        ]
        if mismatched:
            raise ValueError(
                "cannot resume campaign with changed configuration: "
                + ", ".join(mismatched)
            )
        manifest["started_at"] = previous.get("started_at", manifest["started_at"])
        manifest["resumed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["completed"] = [
            f"{dataset}/{task}"
            for dataset, task in tasks
            if (results_root / f"{dataset}_{task}.json").exists()
        ]
        manifest["resources"] = {
            task: resources
            for task, resources in previous.get("resources", {}).items()
            if task in manifest["completed"]
        }
        _aggregate(output_root, tasks)
    _atomic_json(manifest, manifest_path)

    runner = Path(__file__).with_name("run_kurversc_selected_relarena.py")
    failures = 0
    for index, (dataset, task) in enumerate(tasks, start=1):
        output = results_root / f"{dataset}_{task}.csv"
        result_json = output.with_suffix(".json")
        if result_json.exists() and not args.force:
            print(
                json.dumps(
                    {
                        "event": "campaign_task_skipped",
                        "task": f"{dataset}/{task}",
                        "reason": "atomic result already exists",
                    }
                ),
                flush=True,
            )
            if f"{dataset}/{task}" not in manifest["completed"]:
                manifest["completed"].append(f"{dataset}/{task}")
            _atomic_json(manifest, manifest_path)
            continue

        task_parallelism = _task_parallelism(
            dataset_sizes[dataset], maximum=args.search_parallelism
        )
        log_path = logs_root / f"{dataset}_{task}.log"
        print(
            json.dumps(
                {
                    "event": "campaign_task_started",
                    "task": f"{dataset}/{task}",
                    "position": index,
                    "total": len(tasks),
                    "log": str(log_path),
                    "dataset_size_gb": dataset_sizes[dataset] / 1024**3,
                    "search_parallelism": task_parallelism,
                }
            ),
            flush=True,
        )
        environment = os.environ.copy()
        attempts: list[dict[str, Any]] = []
        task_started = time.monotonic()
        while True:
            remaining_task_seconds = (
                args.task_time_limit_hours * 3600
                - (time.monotonic() - task_started)
            )
            if remaining_task_seconds <= 0:
                resources["time_limit_triggered"] = True
                break
            command = [
                sys.executable,
                str(runner),
                "--dataset",
                dataset,
                "--task",
                task,
                "--profile",
                args.profile,
                "--base-cdf",
                args.base_cdf,
                "--peer-context-mode",
                args.peer_context_mode,
                "--peer-max-history-rows",
                str(args.peer_max_history_rows),
                "--search-parallelism",
                str(task_parallelism),
                "--duckdb-memory-limit",
                args.duckdb_memory_limit,
                "--duckdb-temp-limit",
                args.duckdb_temp_limit,
                "--seed",
                str(args.seed),
                "--output",
                str(output),
                "--checkpoint-root",
                str(checkpoint_root),
            ]
            if args.cache_dir:
                command.extend(("--cache-dir", args.cache_dir))
            if args.root_reachable_materialization:
                command.append("--root-reachable-materialization")
            if args.rerank_nested_depth_safety is not None:
                command.append(
                    "--rerank-nested-depth-safety"
                    if args.rerank_nested_depth_safety
                    else "--no-rerank-nested-depth-safety"
                )
            with log_path.open("a", buffering=1) as log:
                returncode, resources = _run_guarded(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    environment=environment,
                    log=log,
                    minimum_available_memory_gb=args.minimum_available_memory_gb,
                    poll_seconds=args.memory_poll_seconds,
                    task_time_limit_seconds=remaining_task_seconds,
                )
            attempts.append(
                {
                    "search_parallelism": task_parallelism,
                    "returncode": returncode,
                    **resources,
                }
            )
            if (
                resources["time_limit_triggered"]
                or not resources["memory_guard_triggered"]
                or task_parallelism == 1
            ):
                break
            task_parallelism = max(1, task_parallelism // 2)
            print(
                json.dumps(
                    {
                        "event": "campaign_task_memory_retry",
                        "task": f"{dataset}/{task}",
                        "search_parallelism": task_parallelism,
                    }
                ),
                flush=True,
            )
        resources = {
            "attempts": attempts,
            "peak_tree_rss_gb": max(
                attempt["peak_tree_rss_gb"] for attempt in attempts
            ),
            "minimum_system_available_gb": min(
                (
                    attempt["minimum_system_available_gb"]
                    for attempt in attempts
                    if attempt["minimum_system_available_gb"] is not None
                ),
                default=None,
            ),
            "memory_guard_triggered": any(
                attempt["memory_guard_triggered"] for attempt in attempts
            ),
            "time_limit_triggered": any(
                attempt["time_limit_triggered"] for attempt in attempts
            ),
            "elapsed_seconds": sum(attempt["elapsed_seconds"] for attempt in attempts),
        }
        manifest["resources"][f"{dataset}/{task}"] = resources
        if returncode == 0 and result_json.exists():
            manifest["completed"].append(f"{dataset}/{task}")
            print(
                json.dumps(
                    {
                        "event": "campaign_task_completed",
                        "task": f"{dataset}/{task}",
                        "result": json.loads(result_json.read_text()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            failures += 1
            manifest["failed"].append(
                {
                    "task": f"{dataset}/{task}",
                    "returncode": returncode,
                    "log": str(log_path),
                    **resources,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "campaign_task_failed",
                        "task": f"{dataset}/{task}",
                        "returncode": returncode,
                        "log": str(log_path),
                        **resources,
                    }
                ),
                flush=True,
            )
            if args.stop_on_error:
                break
        _aggregate(output_root, tasks)
        _atomic_json(manifest, manifest_path)

    frame = _aggregate(output_root, tasks)
    manifest["status"] = "completed" if failures == 0 else "completed_with_failures"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["result_rows"] = len(frame)
    _atomic_json(manifest, manifest_path)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--profile",
        choices=(
            "published-compat-search",
            "staged-production",
            "staged-bounded",
            "staged-retention",
            "staged-calibrated",
            "staged-diverse",
            "staged-adaptive",
        ),
        default="staged-production",
    )
    parser.add_argument(
        "--base-cdf",
        choices=("default", "on", "off"),
        default="default",
        help="Force the base-family CDF representation for every searched graph.",
    )
    parser.add_argument(
        "--peer-context-mode",
        choices=("off", "unweighted", "both"),
        default="off",
    )
    parser.add_argument("--peer-max-history-rows", type=int, default=50_000)
    parser.add_argument("--root-reachable-materialization", action="store_true")
    parser.add_argument("--tasks", nargs="+", type=_task_token)
    parser.add_argument(
        "--rerank-nested-depth-safety",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--search-parallelism",
        type=int,
        default=4,
        help="maximum within-task workers; the scheduler reduces this by data scale",
    )
    parser.add_argument("--duckdb-memory-limit", default="8GB")
    parser.add_argument("--duckdb-temp-limit", default="100GB")
    parser.add_argument("--minimum-available-memory-gb", type=float, default=6.0)
    parser.add_argument("--memory-poll-seconds", type=float, default=10.0)
    parser.add_argument(
        "--task-time-limit-hours",
        type=float,
        default=23.5,
        help="hard wall-clock limit per task; must remain below RelArena's 24h cap",
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
