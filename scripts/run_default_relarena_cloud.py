"""Fresh fixed-profile RelArena reproduction, with RAM-aware admission.

This wrapper never chooses graphs, temporal policies, or experimental readouts.
It calls the registered system through the official runner and changes only the
location of its durable search checkpoints. Elo requires all 21 fresh scores.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from scripts.run_gap_followup_queue import digest, dump, validate_manifest
from scripts.recovered_reference_profile import execution_profile


ROOT = Path(__file__).resolve().parents[1]
BASELINE_URL = "https://raw.githubusercontent.com/PriorLabs/relarena/main/baseline_results/results.csv"
DATASETS = ("rel-amazon", "rel-avito", "rel-event", "rel-f1", "rel-hm", "rel-stack", "rel-trial")
PREPARATION = "prepare-"


def now():
    return datetime.now(timezone.utc).isoformat()


def ram_estimate(dataset, task):
    """Scheduling reservations, not RSS limits or model hyperparameters."""
    estimates = {"rel-f1": 8, "rel-trial": 48, "rel-event": 40,
                 "rel-stack": 140, "rel-avito": 160, "rel-amazon": 256,
                 "rel-hm": 320}
    special = {("rel-stack", "user-badge"): 256,
               ("rel-avito", "user-clicks"): 220}
    return special.get((dataset, task), estimates[dataset])


def task_pairs():
    from relarena.tasks import list_entity_tasks

    pairs = [(spec.dataset, spec.task) for spec in list_entity_tasks(DATASETS)]
    if len(pairs) != 21 or len(set(pairs)) != 21:
        raise ValueError("Exactly 21 distinct RelBench v1 tasks required")
    return sorted(pairs, key=lambda pair: (ram_estimate(*pair), *pair))


def clean_environment():
    # Refuse accidental opt-in hypotheses, graph profiles, known winners,
    # downstream budgets, or cutoff overrides inherited from a research shell.
    unexpected = [key for key in os.environ if key.startswith("KURVERSC_")]
    if unexpected:
        raise ValueError(f"Registered defaults require no inherited overrides: {unexpected}")


def source_files():
    import graphreduce
    import relarena
    import kurversc

    files = {Path(__file__).resolve(), ROOT / "scripts/run_gap_followup_queue.py",
        ROOT / "scripts/recovered_reference_profile.py"}
    for module in (kurversc, graphreduce, relarena):
        package = Path(module.__file__).resolve().parent
        for pattern in ("*.py", "*.json", "*.yaml", "*.yml", "py.typed"):
            files.update(package.rglob(pattern))
    return sorted(files)


def verify_sources(output):
    identity = json.loads((output / "source-identity.json").read_text())
    for path, expected in identity["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen reproduction source changed: {path}")
    return identity


def build_manifest(output, pairs, files, max_workers, cpus, profile="registered"):
    execution_profile(profile)
    def command(mode, *args):
        return ["{python}", "-u", "-m", "scripts.run_default_relarena_cloud",
                "--mode", mode, "--output", str(output),
                *(["--profile", profile] if profile != "registered" else []), *args]

    jobs = []
    for dataset in DATASETS:
        jobs.append(dict(name=PREPARATION + dataset, task=PREPARATION + dataset,
            command=command("prepare", "--dataset", dataset), ram_gib=2, threads=2,
            result=str(output / "datasets" / dataset / "results.json")))
    jobs.append(dict(name="leaderboard-snapshot", command=command("snapshot"),
        ram_gib=2, threads=1, result=str(output / "baseline/results.json")))
    for dataset, task in pairs:
        slug = f"{dataset}_{task}"
        jobs.append(dict(name=slug, task=f"{dataset}/{task}",
            command=command("task", "--dataset", dataset, "--task", task),
            ram_gib=ram_estimate(dataset, task), threads=8,
            depends=[PREPARATION + dataset],
            result=str(output / "tasks" / slug / "results.json")))
    jobs.append(dict(name="final-official-elo", command=command("collect"),
        ram_gib=4, threads=1,
        depends=["leaderboard-snapshot", *[f"{d}_{t}" for d, t in pairs]],
        result=str(output / "elo/results.json")))
    manifest = dict(sources=[str(path) for path in files], reserve_gib=128,
        task_hours=23.75, max_workers=min(max_workers, max(1, cpus // 8)),
        max_threads=cpus, jobs=jobs,
        run_profile=profile,
        scope="Fresh fixed KurveRSC execution profile; no previous scores, winners, or unverified challengers")
    validate_manifest(manifest)
    return manifest


def initialize(output, pairs, profile="registered"):
    import psutil
    from relarena.models.kurversc.model import KurveRSCSystem, _runtime_config
    from relarena.identity import relbench_run_identity

    clean_environment()
    # This cheap identity lookup catches omitted runtime package assets before
    # launching any detached task, rather than reporting a false running suite.
    for dataset, task in pairs:
        relbench_run_identity(dataset, task)
    _, effective_config = execution_profile(profile)
    output = output.resolve()
    # Never reuse an old campaign or previously chosen graph implicitly.
    if output.exists() and any(output.iterdir()):
        raise ValueError("Fresh run requires a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    files = source_files()
    cpus = len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), "cpu_affinity") else os.cpu_count()
    packages = {}
    for name in ("kurversc", "graphreduce", "relarena", "relbench", "catboost", "duckdb", "pandas", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "source checkout"
    identity = dict(created_at=now(), seed=0, registered_runtime_config=_runtime_config(),
        run_profile=profile, effective_runtime_config=effective_config,
        registered_system_attributes=dict(schema_depth=KurveRSCSystem.schema_depth,
            rerank_practical_recent_ties=KurveRSCSystem.rerank_practical_recent_ties),
        packages=packages, python=sys.version, source_sha256={str(p): digest(p) for p in files},
        task_pairs=pairs, initial_available_gib=psutil.virtual_memory().available / 2**30,
        total_ram_gib=psutil.virtual_memory().total / 2**30, cpus=cpus,
        model_overrides={}, checkpoint_policy="Fresh per-task directory; interrupted trials may resume only within this run",
        scope="Registered default system through RelArena's nested temporal splits and official test scorer")
    dump(output / "source-identity.json", identity)
    return identity, files, cpus


def launch(args):
    output = args.output.resolve()
    pairs = task_pairs()
    identity, files, cpus = initialize(output, pairs, args.profile)
    manifest = build_manifest(output, pairs, files, args.max_workers, cpus, args.profile)
    dump(output / "manifest.json", manifest)
    command = [sys.executable, "-u", "-m", "scripts.run_gap_followup_queue",
        "--manifest", str(output / "manifest.json"), "--output", str(output / "queue"), "--detach"]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    print(completed.stdout, end="", flush=True)
    print(json.dumps(dict(output=str(output), max_task_workers=manifest["max_workers"],
        reserve_gib=128, run_profile=args.profile,
        effective_runtime_config=identity["effective_runtime_config"])), flush=True)


def single(args):
    """One local native-default task, not a staged research runner."""
    if (args.dataset, args.task) not in task_pairs():
        raise ValueError("Unknown official v1 entity task")
    initialize(args.output, [(args.dataset, args.task)], args.profile)
    return task(args)


def prepare(args):
    from relbench.datasets import download_dataset
    from relbench.tasks import download_task
    from relbench.utils import get_relbench_cache_dir

    identity = verify_sources(args.output)
    if args.profile != identity["run_profile"]:
        raise ValueError("Execution profile changed between admission and execution")
    system, effective_config = execution_profile(args.profile)
    if effective_config != identity["effective_runtime_config"]:
        raise ValueError("Effective execution recipe changed")
    download_dataset(args.dataset)
    for dataset, task in identity["task_pairs"]:
        if dataset == args.dataset:
            download_task(dataset, task)
    cache = Path(get_relbench_cache_dir()) / args.dataset
    # Record the exact official source bytes; no materialized feature frames
    # or research result artifacts are accepted as task inputs.
    files = sorted(cache.rglob("*.parquet"))
    if not files:
        raise ValueError("Official source cache is empty")
    dump(args.output / "datasets" / args.dataset / "results.json",
        dict(status="completed", dataset=args.dataset, prepared_at=now(),
             official_source_sha256={str(path): digest(path) for path in files}))


def snapshot(args):
    import pandas as pd

    verify_sources(args.output)
    folder = args.output / "baseline"
    folder.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASELINE_URL, timeout=60) as response:
        payload = response.read()
        headers = dict(response.headers)
    path = folder / "published-results.csv"
    path.write_bytes(payload)
    raw = pd.read_csv(path)
    selected = raw.loc[raw.selected.astype(str).str.lower().eq("true") & raw.test_score.notna()].copy()
    if selected.duplicated(["model", "dataset", "task"]).any():
        raise ValueError("Published selected rows are not unique")
    if len(selected[["dataset", "task"]].drop_duplicates()) != 21:
        raise ValueError("Published task set is not the expected 21-task panel")
    counts = selected.groupby("model").size()
    incomplete = counts[counts.ne(21)].index.tolist()
    selected = selected.loc[~selected.model.isin(incomplete)]
    selected.to_csv(folder / "selected-panel.csv", index=False)
    dump(folder / "results.json", dict(status="completed", captured_at=now(),
        url=BASELINE_URL, sha256=hashlib.sha256(payload).hexdigest(), response_headers=headers,
        complete_methods=sorted(selected.model.unique()), excluded_incomplete_methods=incomplete,
        rows=len(selected)))


def task(args):
    import numpy as np
    from relarena.models.kurversc.model import KurveRSCSystem, _runtime_config
    from relarena.runner import run_system_experiment
    from relarena.results import summary_to_dataframe

    started = time.monotonic()
    clean_environment()
    identity = verify_sources(args.output)
    if args.profile != identity["run_profile"]:
        raise ValueError("Execution profile changed between admission and execution")
    system, effective_config = execution_profile(args.profile)
    if effective_config != identity["effective_runtime_config"]:
        raise ValueError("Effective execution recipe changed")
    if _runtime_config() != identity["registered_runtime_config"]:
        raise ValueError("Registered defaults changed between admission and execution")
    if identity.get("registered_system_attributes") != dict(
        schema_depth=KurveRSCSystem.schema_depth,
        rerank_practical_recent_ties=KurveRSCSystem.rerank_practical_recent_ties):
        raise ValueError("Registered system selection policy changed")
    folder = args.output / "tasks" / f"{args.dataset}_{args.task}"
    folder.mkdir(parents=True, exist_ok=True)
    # Only a cache-location override: no profile or model parameter override.
    os.environ["KURVERSC_SEARCH_CHECKPOINT_DIR"] = str(folder / "checkpoints")
    phases = {}

    class AuditedDefaultSystem(system):
        def _fit_arm(self, *values, **kwargs):
            phase = kwargs["phase"]
            phase_started = time.monotonic()
            result, problem = super()._fit_arm(*values, **kwargs)
            record = dict(phase=phase, elapsed_seconds=time.monotonic() - phase_started,
                recommended_config=asdict(result.recommended_config),
                plan_fingerprint=result.plan_fingerprint,
                full_validation_score=result.full_validation_score,
                training_frames=result.fitted_model.training_frames,
                screening_trials=len(result.trials), confirmation_trials=len(result.confirmation_trials),
                rerank_trials=len(result.rerank_trials))
            dump(folder / f"{phase}.json", record)
            dump(folder / f"{phase}-execution-plan.json", result.execution_plan)
            result.results.to_csv(folder / f"{phase}-trials.csv", index=False)
            result.rerank_results.to_csv(folder / f"{phase}-rerank.csv", index=False)
            phases[phase] = record
            return result, problem

    # Preparation has already verified upstream archives. download=False
    # prevents concurrent workers from re-extracting the same dataset zip.
    summary = run_system_experiment(AuditedDefaultSystem, args.dataset, args.task,
        seed=0, download=False, cache_predictions=True, evaluate_test=True)
    score = float(summary.result.test_score)
    if not math.isfinite(score) or set(phases) != {"inner", "outer"}:
        raise ValueError("Complete inner/outer protocol and finite official score required")
    if phases["inner"]["recommended_config"] != phases["outer"]["recommended_config"]:
        raise ValueError("Selected configuration changed in the official reporting arm")
    predictions = np.asarray(summary.result.test_pred)
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite official predictions")
    np.save(folder / "test-predictions.npy", predictions)
    summary_to_dataframe(summary).to_csv(folder / "official-results.csv", index=False)
    verify_sources(args.output)
    dump(folder / "results.json", dict(status="completed", completed_at=now(),
        model="kurversc", dataset=args.dataset, task=args.task, metric=summary.metric_name,
        seed=0, test_score=score, test_metrics=summary.result.test_metrics,
        official_time_total=summary.result.time_total, elapsed_seconds=time.monotonic() - started,
        registered_runtime_config=identity["registered_runtime_config"], phases=phases,
        run_profile=args.profile, effective_runtime_config=effective_config,
        registered_system_attributes=identity["registered_system_attributes"],
        predictions_sha256=digest(folder / "test-predictions.npy"),
        source_identity_sha256=digest(args.output / "source-identity.json"),
        reused_previous_score=False, optional_challengers=False))
    print(json.dumps(dict(event="official_task_completed", dataset=args.dataset,
                         task=args.task, metric=summary.metric_name, test_score=score)), flush=True)


def validate_complete_scores(pairs, frame):
    expected = set(map(tuple, pairs))
    keys = ["dataset", "task"]
    present = set(map(tuple, frame[keys].to_numpy()))
    if len(expected) != 21 or frame.duplicated(keys).any() or present != expected:
        raise ValueError("Full Elo requires exactly all 21 fresh task scores; no carry-forwards")
    if not frame.test_score.map(math.isfinite).all():
        raise ValueError("All official scores must be finite")


def collect(args):
    import numpy as np
    import pandas as pd
    from relarena.evaluation import compute_leaderboard

    identity = verify_sources(args.output)
    frames = []
    for dataset, name in identity["task_pairs"]:
        folder = args.output / "tasks" / f"{dataset}_{name}"
        report = json.loads((folder / "results.json").read_text())
        if (report["status"] != "completed" or report["reused_previous_score"]
                or report["optional_challengers"] or report["registered_runtime_config"] != identity["registered_runtime_config"]
                or report["run_profile"] != identity["run_profile"]
                or report["effective_runtime_config"] != identity["effective_runtime_config"]):
            raise ValueError("Only fresh same-profile official results may enter the final ranking")
        frames.append(pd.read_csv(folder / "official-results.csv"))
    scores = pd.concat(frames, ignore_index=True)
    validate_complete_scores(identity["task_pairs"], scores)
    panel = pd.read_csv(args.output / "baseline/selected-panel.csv")
    opponents = panel.loc[~panel.model.eq("kurversc")]
    matrix = pd.concat([opponents, scores], ignore_index=True)
    folder = args.output / "elo"
    folder.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output / "campaign-results.csv", index=False)
    matrix.to_csv(folder / "input.csv", index=False)
    board = compute_leaderboard(matrix).sort_values("elo", ascending=False, kind="stable")
    board.insert(0, "elo_rank", np.arange(1, len(board) + 1))
    board.to_csv(folder / "leaderboard.csv")
    dump(folder / "results.json", dict(status="completed", computed_at=now(),
        elo=float(board.loc["kurversc", "elo"]), rank=int(board.loc["kurversc", "elo_rank"]),
        tasks=21, carry_forwards=0, methods=len(board),
        run_profile=args.profile, effective_runtime_config=identity["effective_runtime_config"],
        baseline=json.loads((args.output / "baseline/results.json").read_text()),
        source_identity_sha256=digest(args.output / "source-identity.json"),
        leaderboard=board.reset_index().to_dict("records"),
        scope="21 fresh same-profile official test scores; published KurveRSC replaced, not added as a second opponent"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("launch", "prepare", "snapshot", "task", "collect", "single"), default="launch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--task")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--profile", choices=("registered", "recovered-reference"), default="registered")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.max_workers < 1 or (args.mode in {"prepare", "task", "single"} and args.dataset not in DATASETS):
        parser.error("Positive parallelism and an explicit v1 dataset required")
    if args.mode in {"task", "single"} and not args.task:
        parser.error("Task mode requires --task")
    {"launch": launch, "prepare": prepare, "snapshot": snapshot, "task": task, "collect": collect,
     "single": single}[args.mode](args)


if __name__ == "__main__":
    main()
