"""Isolated validation audit of a small, recent-competitive graph finalist.

Use only THIS campaign's completed full-fidelity trials. The extra graph is
chosen by schema-generic complexity/recent-validation criteria, never by a
historical graph or official score. Replay the ordinary point-in-time caches
and native frame estimator before testing the existing count-median menu.
This is an experiment, not an automatic graph replacement or default change.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

from scripts.engagement_local_comparison import freeze
from scripts.run_default_relarena_cloud import verify_sources
from scripts.run_gap_followup_queue import digest, dump


def small_finalist(trials, incumbent, *, relative_recent_loss_cap=.01):
    if not 0 <= relative_recent_loss_cap <= .01:
        raise ValueError("A bounded recent MAE loss cap is required")
    completed = [(path, t) for path, t in trials if t.status == "completed"
        and t.stage == "rerank" and t.metric == "mae" and t.execution_plan
        and len(t.cutoff_scores) >= 2
        and all(v is not None and math.isfinite(v) for v in t.cutoff_scores)]
    same = [t for _, t in completed if t.config == incumbent]
    if not same:
        raise ValueError("The deployed graph must have complete full-fidelity evidence")
    width = same[0].feature_count
    newest = min(t.cutoff_scores[-1] for _, t in completed)
    candidates = [(p, t) for p, t in completed if t.config != incumbent
        and t.feature_count < width
        and t.cutoff_scores[-1] <= newest * (1 + relative_recent_loss_cap) + 1e-12]
    return min(candidates, key=lambda pair: (pair[1].feature_count,
        pair[1].config.complexity, json.dumps(asdict(pair[1].config), sort_keys=True),
        str(pair[0])), default=None)


def plan(args):
    identity = verify_sources(args.campaign)
    slug = f"{args.dataset}_{args.task}"
    graph = args.campaign / "tasks" / slug / "procedure/graphs"
    selected_path = graph / "results" / f"{slug}.json"
    incumbent = json.loads(json.loads(selected_path.read_text())["config"])
    trials = []
    for path in sorted((graph / "checkpoints").glob(f"*/{slug}/*/trials/*.pkl")):
        with path.open("rb") as handle:
            trial = pickle.load(handle)["trial"]  # trusted, current-run artifacts
        if trial.stage == "rerank":
            trials.append((path, trial))
    matches = [t.config for _, t in trials
        if json.loads(json.dumps(asdict(t.config))) == incumbent]
    if not matches:
        raise ValueError("Current selected graph is not in its own rerank checkpoints")
    selected = small_finalist(trials, matches[0])
    if selected is None:
        raise ValueError("No smaller recent-competitive completed finalist exists")
    path, trial = selected
    result = dict(dataset=args.dataset, task=args.task, campaign=str(args.campaign),
        current_campaign_identity_sha256=digest(args.campaign / "source-identity.json"),
        checkpoint=str(path), checkpoint_sha256=digest(path), config=asdict(trial.config),
        feature_count=trial.feature_count, cutoff_scores=trial.cutoff_scores,
        sources=identity["source_sha256"], probe_source=str(Path(__file__).resolve()),
        probe_source_sha256=digest(__file__), selection="smallest completed recent-competitive finalist",
        relative_recent_loss_cap=.01, model_defaults_changed=False, official_test_accessed=False)
    freeze(args.output / "plan.json", result)
    return result


@contextmanager
def normalized_engine_limits():
    # Same normalization used by the full staged executor; process-local only.
    import kurversc.core as core
    original = core._connection_scope

    @contextmanager
    def scope(connection, *, memory_limit=None, max_temp_directory_size=None):
        def normalize(v):
            return None if isinstance(v, str) and v.lower() in {"none", "null", "unlimited"} else v
        with original(connection, memory_limit=normalize(memory_limit),
                      max_temp_directory_size=normalize(max_temp_directory_size)) as con:
            yield con

    core._connection_scope = scope
    try:
        yield
    finally:
        core._connection_scope = original


def materialize(args):
    from scripts import benchmark_sparse_training as sparse
    chosen = json.loads((args.output / "plan.json").read_text())
    path = Path(chosen["checkpoint"])
    if digest(path) != chosen["checkpoint_sha256"]:
        raise ValueError("Selected finalist bytes changed")
    with path.open("rb") as handle:
        trial = pickle.load(handle)["trial"]
    original = sparse._checkpoint

    def checkpoint(dataset, task):
        if (dataset, task) != (args.dataset, args.task):
            raise ValueError("Unexpected dataset/task")
        return path, trial

    sparse._checkpoint = checkpoint
    sys.argv = [sparse.__file__, "--dataset", args.dataset, "--task", args.task,
        "--fold", "current", "--output", str(args.output / "task/cache"),
        "--atomic-date-replay", "--materialize-only", "--policy-origins", "3",
        "--frame-counts", "1,12", "--row-floors", "1000", "--max-frames", "64",
        "--duckdb-memory-limit", "none", "--duckdb-temp-limit", "none"]
    try:
        with normalized_engine_limits():
            sparse.main()
    finally:
        sparse._checkpoint = original


def native_audit(args):
    from kurversc.temporal_policy import TemporalFitPolicy
    from scripts import benchmark_temporal_policy as audit
    original = audit.bounded_temporal_policies
    # Audit only the already-declared native procedure: do not spend another
    # full temporal-policy search or silently select a different comparator.
    audit.bounded_temporal_policies = lambda task: (TemporalFitPolicy("incumbent"),)
    task_root = args.output / "task"
    sys.argv = [audit.__file__, "--cache", str(task_root / "cache"),
        "--output", str(task_root / "raw-audit"), "--seeds", "0,1,2",
        "--origins", "3", "--threads", "4", "--phase-frozen-cache", str(task_root / "snapshots")]
    try:
        audit.main()
    finally:
        audit.bounded_temporal_policies = original
    raw = json.loads((task_root / "raw-audit/manifest.json").read_text())
    freeze(task_root / "selection/manifest.json", raw)
    freeze(task_root / "selection/results.json", dict(selected_policy=asdict(TemporalFitPolicy("incumbent")),
        reason="Unchanged native frame-ensemble comparator, not a new temporal selection",
        official_test_accessed=False, model_defaults_changed=False))


def run(args):
    from scripts.run_kurversc_local_campaign import _run_guarded
    started = time.monotonic()
    chosen = plan(args)
    dump(args.output / "status.json", dict(status="running", pid=os.getpid(), active="cache"))
    task = args.output / "task"
    own = [sys.executable, "-u", str(Path(__file__).resolve()), "--campaign", str(args.campaign),
        "--dataset", args.dataset, "--task", args.task, "--output", str(args.output)]
    stages = [
        ("cache", [*own, "--mode", "materialize"], task / "cache/materialization.json"),
        ("snapshots", [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", "snapshots",
            "--campaign", str(args.campaign), "--dataset", args.dataset, "--task", args.task,
            "--output", str(args.output)], task / "snapshots/manifest.json"),
        ("native", [*own, "--mode", "native"], task / "raw-audit/results.json"),
        ("count", [sys.executable, "-u", "-m", "scripts.benchmark_poisson_count_median",
            "--task-root", str(task), "--output", str(args.output / "count")], args.output / "count/results.json"),
    ]
    resources = {}
    for name, command, completion in stages:
        verify_sources(args.campaign)
        if digest(__file__) != chosen["probe_source_sha256"]:
            raise ValueError("Probe changed during execution")
        if completion.exists():
            continue
        dump(args.output / "status.json", dict(status="running", pid=os.getpid(), active=name, resources=resources))
        with (args.output / f"{name}.log").open("a", buffering=1) as log:
            code, usage = _run_guarded(command, cwd=Path(__file__).resolve().parents[1]
                if (Path(__file__).resolve().parents[1] / "kurversc").exists()
                else Path(__import__("kurversc").__file__).resolve().parents[1],
                environment=dict(os.environ), log=log, minimum_available_memory_gb=128,
                poll_seconds=5, task_time_limit_seconds=max(1, 4*3600-(time.monotonic()-started)))
        resources[name] = usage
        dump(args.output / f"{name}-resources.json", usage)
        if code or not completion.exists():
            dump(args.output / "status.json", dict(status="failed", active=name, code=code, resources=resources))
            raise RuntimeError(f"Small-finalist stage {name} incomplete: exit {code}")
    result = json.loads((args.output / "count/results.json").read_text())
    dump(args.output / "status.json", dict(status="completed", selected=result["selected"],
        resources=resources, elapsed_seconds=time.monotonic()-started,
        model_defaults_changed=False, official_test_accessed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "plan", "materialize", "snapshots", "native"), default="run")
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.campaign, args.output = args.campaign.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "snapshots":
        from scripts import materialize_policy_folds as snapshots
        sys.argv = [snapshots.__file__, "--cache", str(args.output / "task/cache"),
            "--output", str(args.output / "task/snapshots"), "--origins", "3",
            "--duckdb-memory-limit", "none", "--duckdb-temp-limit", "none"]
        with normalized_engine_limits():
            snapshots.main()
    else:
        {"run":run, "plan":plan, "materialize":materialize, "native":native_audit}[args.mode](args)


if __name__ == "__main__":
    main()
