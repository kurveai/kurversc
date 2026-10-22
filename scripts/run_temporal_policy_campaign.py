"""Sequential, resumable validation-selected temporal-policy campaign.

Reuse frozen graph-search winners, not test-selected graph configurations.
Every new task uses the same bounded policy menu and validation gate. Only
completed stages are reused; resource failures are reported, never replaced
with an old score. Large stages run in separate guarded processes.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import pandas as pd
import psutil

from kurversc.checkpointing import search_runtime_identity
from kurversc.temporal_policy import TemporalFitPolicy
from scripts.run_kurversc_local_campaign import (
    _atomic_json,
    _dataset_bytes,
    _run_guarded,
)
from scripts.temporal_campaign_support import retrospective_folds


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CAMPAIGN = ROOT / "results/kurversc-root-reachable-context-fixed-all21-20261003"


def now():
    return datetime.now(timezone.utc).isoformat()


def stage_environment():
    environment = dict(os.environ,OMP_NUM_THREADS="1")
    if environment.get("KURVERSC_TASK_CONTEXT"):
        # Keep the resolved/pinned installed GraphReduce and portable runtime.
        # A sibling-repository PYTHONPATH is only a legacy benchmark layout.
        return environment
    environment["PYTHONPATH"] = os.pathsep.join([
        str(ROOT.parent.parent/"graphreduce"),str(ROOT),str(ROOT/".relarena-v2/src")])
    return environment


def campaign_tasks(path):
    frame = pd.read_csv(path, usecols=["dataset", "task"])
    tasks = list(frame.itertuples(index=False, name=None))
    if len(tasks) != 21 or len(set(tasks)) != 21:
        raise ValueError("Campaign must contain exactly 21 unique dataset/task pairs")
    sizes = {dataset: _dataset_bytes(dataset, cache_dir=None) for dataset, _ in tasks}
    return sorted(tasks, key=lambda pair: (sizes[pair[0]], *pair))


def remaining_seconds(resources, limit_hours):
    used = sum(float(stage.get("elapsed_seconds", 0)) for stage in resources.values())
    return max(0, limit_hours * 3600 - used)


def current_run_cache(root, dataset, task):
    """Resume only this run's frames, never a task-named historical winner.

    A legacy cache's own checkpoint checksum proves its internal integrity,
    not that its graph/data match the current campaign's selection. Keeping
    the cache inside the run also prevents local scratch data from changing
    the procedure observed by a clean independent reproduction.
    """
    if not all(name and Path(name).name == name for name in (dataset, task)):
        raise ValueError("Simple dataset and task identifiers required")
    return Path(root) / "tasks" / f"{dataset}_{task}" / "cache"


def cache_folds(cache, origins):
    metadata = json.loads((cache / "materialization.json").read_text())
    availability = metadata.get("label_available_at")
    headers = pd.concat(
        [
            pd.read_parquet(cache / name, columns=["__event_cutoff",*([availability] if availability else [])])
            for name in ("history.parquet", "validation.parquet")
        ],
        ignore_index=True,
    )
    folds = retrospective_folds(
        pd.to_datetime(headers["__event_cutoff"]).unique(),
        size=metadata["validation_cutoffs"],
        origins=origins,
    )
    if availability:
        from kurversc.temporal_policy import InsufficientPolicySupport,policy_support
        for held in folds:
            try:
                policy_support(headers,timestamp="__event_cutoff",
                    origin=pd.Timestamp(held[0])-pd.Timedelta(metadata.get("phase_boundary_lag","0d")),
                    label_horizon=metadata["label_horizon"],refit_frames=metadata["validation_cutoffs"],
                    policy=TemporalFitPolicy("incumbent"))
            except InsufficientPolicySupport:
                return []
    return folds


def retain_incumbent(cache, output):
    """Explicit target-blind fallback when two historical phases do not exist."""
    metadata = json.loads((cache / "materialization.json").read_text())
    frames = json.loads((cache / "frames.json").read_text())
    score = metadata.get("control_score", frames["control_score"])
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        dict(
            metadata=metadata,
            cache=str(cache),
            phase_frozen_cache=None,
            fallback="insufficient_history_for_two_retrospective_phases",
        ),
        output / "manifest.json",
    )
    _atomic_json(
        dict(
            dataset=metadata["dataset"],
            task=metadata["task"],
            selected_policy=asdict(TemporalFitPolicy("incumbent")),
            decision=dict(
                selected="incumbent",
                candidates={},
                reason="insufficient_history_for_two_retrospective_phases",
            ),
            records=[
                dict(policy="incumbent", seed=0, origin="1900-01-01", score=score)
            ],
        ),
        output / "results.json",
    )


def source_identity():
    paths = [
        "scripts/run_temporal_policy_campaign.py",
        "scripts/temporal_campaign_support.py",
        "scripts/benchmark_sparse_training.py",
        "scripts/streaming_feature_cache.py",
        "scripts/materialize_policy_folds.py",
        "scripts/benchmark_temporal_policy.py",
        "scripts/reselect_temporal_policy.py",
        "scripts/benchmark_supported_official.py",
        "scripts/run_kurversc_local_campaign.py",
        "scripts/run_kurversc_selected_relarena.py",
        "scripts/benchmark_frozen_event_summaries.py",
        "kurversc/temporal_policy.py",
        "kurversc/training_support.py",
    ]
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths
    }


def summarize(root, manifest):
    rows = []
    for key, state in manifest["tasks"].items():
        if state["status"] != "completed":
            continue
        dataset, task = key.split("/")
        result = json.loads(
            (root / "tasks" / f"{dataset}_{task}" / "official/results.json").read_text()
        )
        stages = state["resources"]
        rows.append(
            dict(
                dataset=dataset,
                task=task,
                metric=result["metric"],
                control_score=result["official_test_scores"]["control"],
                test_score=result["official_test_scores"]["candidate"],
                selected_policy=result["arm"],
                elapsed_seconds=sum(
                    stage.get("elapsed_seconds", 0) for stage in stages.values()
                ),
                peak_rss_gib=result.get("peak_rss_gib"),
                stage_resources=json.dumps(stages),
            )
        )
    temporary = root / "campaign-results.csv.tmp"
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(root / "campaign-results.csv")


def run(args):
    if not 0 < args.task_hours < 24 or args.origins < 2 or args.threads < 1:
        raise ValueError(
            "Require a positive <24h total task budget, >=2 origins and threads>=1"
        )
    if args.minimum_available_gib < 8 or args.minimum_disk_gib < 20:
        raise ValueError("Preserve at least 8 GiB RAM headroom and 20 GiB free disk")
    root = args.output.resolve()
    for name in ("tasks", "logs", "resources"):
        (root / name).mkdir(parents=True, exist_ok=True)
    # Keep the descriptor alive for the entire run. A crash releases the lock;
    # a second scheduler cannot accidentally duplicate every active stage.
    campaign_lock = (root / "scheduler.lock").open("a")
    try:
        fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("A scheduler already owns this campaign") from exc
    tasks = campaign_tasks(SOURCE_CAMPAIGN / "campaign-results.csv")
    if args.adopt_active_task:
        adopted = tuple(args.adopt_active_task.split("/"))
        if adopted not in tasks:
            raise ValueError("Adopted task is not in the 21-task campaign")
        tasks.remove(adopted)
        tasks.insert(0, adopted)
    identity = dict(
        source=source_identity(),
        runtime=search_runtime_identity(),
        queue=["/".join(pair) for pair in tasks],
        task_parallelism=1,
        threads=args.threads,
        origins=args.origins,
        seeds=[0, 1, 2],
        blend_weights=[0.125, 0.5],
        task_hours=args.task_hours,
        minimum_available_gib=args.minimum_available_gib,
        minimum_disk_gib=args.minimum_disk_gib,
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_temp_limit=args.duckdb_temp_limit,
        scope="Temporal fitting-policy selection on frozen prior graph-search winners; no new joint graph search",
    )
    path = root / "campaign-manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["identity"] != identity:
            raise ValueError(
                "Resume refused: campaign source/runtime/configuration changed"
            )
        manifest["resumed_at"] = now()
    else:
        manifest = dict(
            identity=identity,
            started_at=now(),
            status="running",
            active=None,
            tasks={
                key: dict(status="queued", resources={}) for key in identity["queue"]
            },
        )
    manifest["scheduler_pid"] = os.getpid()
    _atomic_json(manifest, path)
    environment = stage_environment()

    def stage(key, name, module, arguments, complete):
        state = manifest["tasks"][key]
        if complete.exists():
            return
        if source_identity() != identity["source"]:
            raise RuntimeError(
                "Source changed during campaign; refusing mixed-implementation results"
            )
        seconds = remaining_seconds(state["resources"], args.task_hours)
        if seconds <= 0:
            raise TimeoutError("Total task budget exhausted")
        available = psutil.virtual_memory().available / 1024**3
        disk = shutil.disk_usage(root).free / 1024**3
        if available < args.minimum_available_gib or disk < args.minimum_disk_gib:
            raise MemoryError(
                f"Admission deferred: RAM available={available:.1f} GiB; disk free={disk:.1f} GiB"
            )
        state["status"] = "running"
        state.setdefault("started_at", now())
        manifest["active"] = dict(task=key, stage=name, started_at=now())
        _atomic_json(manifest, path)
        slug = key.replace("/", "_")
        with (root / "logs" / f"{slug}-{name}.log").open("a", buffering=1) as log:
            code, resources = _run_guarded(
                [sys.executable, "-u", "-m", module, *map(str, arguments)],
                cwd=ROOT,
                environment=environment,
                log=log,
                minimum_available_memory_gb=args.minimum_available_gib,
                poll_seconds=10,
                task_time_limit_seconds=seconds,
            )
        resources["returncode"] = code
        state["resources"][name] = resources
        _atomic_json(resources, root / "resources" / f"{slug}-{name}.json")
        _atomic_json(manifest, path)
        if code or not complete.exists():
            if resources["memory_guard_triggered"]:
                raise MemoryError(f"{name}: memory guard stopped worker")
            if resources["time_limit_triggered"]:
                raise TimeoutError(f"{name}: total task budget exhausted")
            raise RuntimeError(
                f"{name}: exit {code}, completion artifact={complete.exists()}"
            )
        print(json.dumps(dict(task=key, completed_stage=name, **resources)), flush=True)

    for dataset, task in tasks:
        key, slug = f"{dataset}/{task}", f"{dataset}_{task}"
        state = manifest["tasks"][key]
        if state["status"] in {
            "completed",
            "failed",
            "resource_deferred",
            "time_budget_exceeded",
        }:
            continue
        folder = root / "tasks" / slug
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if key == args.adopt_active_task:
                manifest["active"] = dict(
                    task=key, stage="adopted_policy_pipeline", started_at=now()
                )
                state["status"] = "running"
                state["adopted"] = True
                _atomic_json(manifest, path)
                finish = root / "resources" / f"{slug}-official.json"
                started = time.monotonic()
                while not finish.exists():
                    if time.monotonic() - started > args.task_hours * 3600:
                        raise TimeoutError(
                            "Adopted pipeline did not finish within task budget"
                        )
                    time.sleep(10)
                for name in ("audit", "selector", "official"):
                    resource_file = root / "resources" / f"{slug}-{name}.json"
                    if resource_file.exists():
                        state["resources"][name] = json.loads(resource_file.read_text())
                if not (folder / "official/results.json").exists():
                    if any(
                        r.get("memory_guard_triggered")
                        for r in state["resources"].values()
                    ):
                        raise MemoryError("Adopted pipeline hit memory guard")
                    raise RuntimeError(
                        "Adopted pipeline exited without official result; inspect stage logs"
                    )
            else:
                cache = current_run_cache(root, dataset, task)
                state["cache"] = str(cache)
                limits = [
                    "--duckdb-memory-limit",
                    args.duckdb_memory_limit,
                    "--duckdb-temp-limit",
                    args.duckdb_temp_limit,
                ]
                stage(
                    key,
                    "cache",
                    "scripts.benchmark_sparse_training",
                    [
                        "--dataset",
                        dataset,
                        "--task",
                        task,
                        "--fold",
                        "current",
                        "--output",
                        cache,
                        "--atomic-date-replay",
                        "--materialize-only",
                        "--policy-origins",
                        args.origins,
                        "--frame-counts",
                        "1,12",
                        "--row-floors",
                        "1000",
                        "--max-frames",
                        "64",
                        *limits,
                    ],
                    cache / "materialization.json",
                )
                selection = folder / "selection"
                if not cache_folds(cache, args.origins):
                    retain_incumbent(cache, selection)
                    state["selection_note"] = (
                        "Insufficient history: incumbent retained without challenger promotion"
                    )
                else:
                    snapshots = folder / "snapshots"
                    stage(
                        key,
                        "snapshots",
                        "scripts.materialize_policy_folds",
                        [
                            "--cache",
                            cache,
                            "--output",
                            snapshots,
                            "--origins",
                            args.origins,
                            *limits,
                        ],
                        snapshots / "manifest.json",
                    )
                    audit = folder / "raw-audit"
                    stage(
                        key,
                        "audit",
                        "scripts.benchmark_temporal_policy",
                        [
                            "--cache",
                            cache,
                            "--output",
                            audit,
                            "--phase-frozen-cache",
                            snapshots,
                            "--origins",
                            args.origins,
                            "--seeds",
                            "0,1,2",
                            "--threads",
                            args.threads,
                        ],
                        audit / "results.json",
                    )
                    stage(
                        key,
                        "selector",
                        "scripts.reselect_temporal_policy",
                        [
                            "--input",
                            audit,
                            "--output",
                            selection,
                            "--blend-weights",
                            "0.125,0.5",
                        ],
                        selection / "results.json",
                    )
                stage(
                    key,
                    "official",
                    "scripts.benchmark_supported_official",
                    [
                        "--dataset",
                        dataset,
                        "--task",
                        task,
                        "--policy-audit",
                        selection / "results.json",
                        "--output",
                        folder / "official",
                        *limits,
                    ],
                    folder / "official/results.json",
                )
            state.update(status="completed", finished_at=now())
            print(json.dumps(dict(task=key, status="completed")), flush=True)
        except Exception as exc:
            status = (
                "resource_deferred"
                if isinstance(exc, MemoryError)
                else "time_budget_exceeded"
                if isinstance(exc, TimeoutError)
                else "failed"
            )
            state.update(
                status=status, finished_at=now(), error=f"{type(exc).__name__}: {exc}"
            )
            print(
                json.dumps(dict(task=key, status=status, error=state["error"])),
                flush=True,
            )
        manifest["active"] = None
        _atomic_json(manifest, path)
        summarize(root, manifest)
    manifest.update(status="finished", finished_at=now(), active=None)
    _atomic_json(manifest, path)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adopt-active-task")
    parser.add_argument("--task-hours", type=float, default=23)
    parser.add_argument("--origins", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--minimum-available-gib", type=float, default=8)
    parser.add_argument("--minimum-disk-gib", type=float, default=30)
    parser.add_argument("--duckdb-memory-limit", default="12GB")
    parser.add_argument("--duckdb-temp-limit", default="120GB")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
