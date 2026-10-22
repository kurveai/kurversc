"""RAM-aware full-procedure benchmark with an attendance recovery gate.

Every task performs a fresh staged/CDF/peer graph search, phase-frozen temporal
policy selection, bounded optional signal selection, and official replay. No
historical graph, prediction, blend, or test score supplies a task result.
The attendance acceptance gate controls deployment of the remaining tasks;
it is never available to graph or learner selection.
"""

import argparse
from argparse import Namespace
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

from scripts import run_default_relarena_cloud as base
from scripts import run_fresh_cloud_campaign as fresh
from scripts.run_gap_followup_queue import digest, dump, validate_manifest


PROFILE = "staged-complete"
CONTRACT = dict(
    graph_profile="staged-retention", base_cdf=True, peer_context="both",
    peer_max_history_rows=50_000, search_parallelism=1,
    initial_production_training_frames=1,
    graph_estimator_mode="frame_ensemble", search_max_features=10_000,
    temporal_origins=3, temporal_seeds=[0, 1, 2], temporal_seed_parallelism=1,
    temporal_selection_required=True, ordered_median_challenger=True,
    optional_signal_selection_required=True, force_graph=False,
    reuse_previous_scores=False, task_hours=23.75,
)
RECOVERY_TASK = ("rel-event", "user-attendance")


def ram_estimate(dataset, task):
    # Reserve complete search + temporal-policy growth, not an RSS kill limit.
    estimates = dict(**{"rel-f1": 8, "rel-event": 80, "rel-trial": 64,
        "rel-stack": 180, "rel-avito": 220, "rel-amazon": 280, "rel-hm": 320})
    return max(estimates[dataset], base.ram_estimate(dataset, task))


def build_manifest(output, pairs, sources, max_workers, cpus):
    if set(pairs) != set(base.task_pairs()):
        raise ValueError("Exactly the 21 official tasks are required")

    def command(mode, *args):
        return ["{python}", "-u", "-m", "scripts.run_staged_complete_cloud",
            "--mode", mode, "--output", str(output), *args]

    jobs = [dict(name="prepare-" + dataset, task="prepare-" + dataset,
        command=command("prepare", "--dataset", dataset), ram_gib=2, threads=2,
        result=str(output / "datasets" / dataset / "results.json"))
        for dataset in base.DATASETS]
    jobs.append(dict(name="leaderboard-snapshot", command=command("snapshot"),
        ram_gib=2, threads=1, result=str(output / "baseline/results.json")))
    pairs = sorted(pairs, key=lambda pair: (
        pair != RECOVERY_TASK, ram_estimate(*pair), *pair))
    for dataset, task in pairs:
        slug = f"{dataset}_{task}"
        dependencies = ["prepare-" + dataset]
        if (dataset, task) != RECOVERY_TASK:
            dependencies.append("attendance-recovery-gate")
        jobs.append(dict(name=slug, task=f"{dataset}/{task}",
            command=command("task", "--dataset", dataset, "--task", task),
            ram_gib=ram_estimate(dataset, task), threads=8, depends=dependencies,
            result=str(output / "tasks" / slug / "results.json")))
    jobs.append(dict(name="attendance-recovery-gate",
        command=command("gate"), ram_gib=2, threads=1,
        depends=["rel-event_user-attendance", "leaderboard-snapshot"],
        result=str(output / "recovery-gate/results.json")))
    jobs.append(dict(name="final-official-elo", command=command("collect"),
        ram_gib=4, threads=1, depends=["leaderboard-snapshot",
            *[f"{d}_{t}" for d, t in pairs]], result=str(output / "elo/results.json")))
    manifest = dict(sources=[str(path) for path in sources], reserve_gib=128,
        task_hours=CONTRACT["task_hours"], max_workers=min(max_workers, max(1, cpus // 8)),
        max_threads=cpus, jobs=jobs, run_profile=PROFILE,
        scope="Fresh complete staged procedure; attendance must recover its client win before the other twenty tasks start")
    validate_manifest(manifest)
    return manifest


def launch(args):
    pairs = base.task_pairs()
    identity, sources, cpus = base.initialize(args.output, pairs)
    # Freeze every stage script and the task-list metadata used by the existing
    # campaign code, not just the one-call native adapter.
    sources = sorted(set(sources) | set((base.ROOT / "scripts").glob("*.py")) |
        {base.ROOT / "docs/checkpoints/20261007/frozen-graph-campaign-21.csv"})
    identity.update(run_profile=PROFILE, effective_runtime_config=CONTRACT,
        source_sha256={str(path): digest(path) for path in sources},
        scope="Fresh full staged/CDF/peer search and validation-selected learner; no remembered winners")
    dump(args.output / "source-identity.json", identity)
    manifest = build_manifest(args.output, pairs, sources, args.max_workers, cpus)
    dump(args.output / "manifest.json", manifest)
    command = [sys.executable, "-u", "-m", "scripts.run_gap_followup_queue",
        "--manifest", str(args.output / "manifest.json"),
        "--output", str(args.output / "queue"), "--detach"]
    completed = subprocess.run(command, cwd=base.ROOT, check=True,
        capture_output=True, text=True)
    print(completed.stdout, end="", flush=True)
    print(json.dumps(dict(profile=PROFILE, recovery_first=RECOVERY_TASK,
        max_workers=manifest["max_workers"], reserve_gib=128, output=str(args.output))), flush=True)


def verify_profile(output):
    identity = base.verify_sources(output)
    if identity["run_profile"] != PROFILE or identity["effective_runtime_config"] != CONTRACT:
        raise ValueError("The complete staged execution contract changed")
    return identity


def prepare(args):
    from relbench.datasets import download_dataset
    from relbench.tasks import download_task
    from relbench.utils import get_relbench_cache_dir

    identity = verify_profile(args.output)
    download_dataset(args.dataset)
    for dataset, task in identity["task_pairs"]:
        if dataset == args.dataset:
            download_task(dataset, task)
    files = sorted((Path(get_relbench_cache_dir()) / args.dataset).rglob("*.parquet"))
    if not files:
        raise ValueError("Official source cache is empty")
    dump(args.output / "datasets" / args.dataset / "results.json",
        dict(status="completed", official_source_sha256={str(p): digest(p) for p in files}))


def verify_stages(root, dataset, task):
    key, slug = f"{dataset}/{task}", f"{dataset}_{task}"
    folder = root / "workers" / slug / "temporal/tasks" / slug
    temporal = json.loads((folder / "selection/results.json").read_text())
    signals = json.loads((folder / "signal-challengers/selection.json").read_text())
    official = fresh.official_result(root, key)
    graph = json.loads((root / "graphs/results" / f"{slug}.json").read_text())
    if official is None:
        raise ValueError("Complete official signal replay is required")
    if graph["profile"] != "staged-retention" or graph["base_distribution_features"] is not True:
        raise ValueError("The stable graph search/CDF stage was omitted")
    if graph.get("peer_context_mode") != "both":
        raise ValueError("The stable peer-context stage was omitted")
    policy = temporal["selected_policy"]
    if temporal["decision"]["selected"] != policy["name"]:
        raise ValueError("The recorded temporal winner differs from its validation decision")
    if policy["name"] != "incumbent":
        if temporal["decision"]["candidates"][policy["name"]]["reason"] != "eligible":
            raise ValueError("Temporal challenger lacks validation admission")
    if signals["selected"] != signals["decision"]["selected"]:
        raise ValueError("The optional signal winner differs from its validation decision")
    if signals["selected"] != "incumbent":
        if signals["decision"]["candidates"][signals["selected"]]["reason"] != "eligible":
            raise ValueError("Signal challenger lacks validation admission")
    return official, dict(graph_config=graph["config"], temporal_policy=policy,
        temporal_decision=temporal["decision"]["reason"], signal_selected=signals["selected"])


def task(args):
    import os

    started = time.monotonic()
    identity = verify_profile(args.output)
    folder = args.output / "tasks" / f"{args.dataset}_{args.task}"
    root = folder / "procedure"
    root.mkdir(parents=True, exist_ok=True)
    key = f"{args.dataset}/{args.task}"
    # The outer RAM-aware queue owns admission, process-tree cancellation, and
    # the cumulative <24h task deadline. The existing engine owns stage resumes.
    procedure_identity = dict(signal_challengers=True, complete_profile=copy.deepcopy(CONTRACT))
    state_path = root / "campaign-status.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous["identity"] != procedure_identity or set(previous["tasks"]) != {key}:
            raise ValueError("Interrupted procedure has a different immutable contract")
    else:
        dump(state_path, dict(identity=procedure_identity,
            tasks={key: dict(status="running", consumed_seconds=0)}))
    os.environ["KURVERSC_DUCKDB_MEMORY_LIMIT"] = "none"
    os.environ["KURVERSC_DUCKDB_MAX_TEMP_DIRECTORY_SIZE"] = "none"
    values = Namespace(output=root, dataset=args.dataset, task=args.task,
        search_workers=1, seed_workers=1, task_hours=CONTRACT["task_hours"],
        signal_challengers=True)
    code = fresh.task_main(values)
    if code:
        raise RuntimeError(f"The complete procedure failed: {code}")
    official, stages = verify_stages(root, args.dataset, args.task)
    score = float(official["official_test_scores"]["candidate"])
    if not math.isfinite(score):
        raise ValueError("A finite official score is required")
    row = official_score_row(args.dataset, args.task, official,
        elapsed_seconds=time.monotonic() - started)
    pd.DataFrame([row]).to_csv(folder / "official-results.csv", index=False)
    verify_profile(args.output)
    dump(folder / "results.json", dict(status="completed", **row,
        run_profile=PROFILE, contract=CONTRACT, stages=stages,
        elapsed_seconds=time.monotonic() - started, reused_previous_score=False,
        source_identity_sha256=digest(args.output / "source-identity.json"),
        official_report=str(fresh.official_path(root, key))))
    print(json.dumps(dict(event="complete_task_scored", **row, stages=stages)), flush=True)


def recovery_passes(score, opponent, stages):
    if stages.get("temporal_policy") is None:
        raise ValueError("Missing validation-selected learner stage")
    if not math.isfinite(score) or not math.isfinite(opponent):
        raise ValueError("Finite acceptance scores are required")
    return score < opponent


def official_score_row(dataset, task, official, *, elapsed_seconds):
    # RelArena filters its leaderboard to selected rows. Omitting this flag
    # when concatenating with the published panel would drop our fresh scores.
    return dict(model="kurversc", dataset=dataset, task=task,
        metric=official["metric"],
        test_score=float(official["official_test_scores"]["candidate"]),
        selected=True, time_total=float(elapsed_seconds))


def gate(args):
    verify_profile(args.output)
    report = json.loads((args.output / "tasks/rel-event_user-attendance/results.json").read_text())
    panel = pd.read_csv(args.output / "baseline/selected-panel.csv")
    clients = panel.loc[panel.model.str.startswith("tabpfn-rel-client-")]
    model = sorted(clients.model.unique())[-1]
    opponent = float(clients.loc[clients.model.eq(model) &
        clients.dataset.eq("rel-event") & clients.task.eq("user-attendance"), "test_score"].iloc[0])
    passed = recovery_passes(report["test_score"], opponent, report["stages"])
    dump(args.output / "recovery-gate/results.json", dict(status="passed" if passed else "failed",
        score=report["test_score"], opponent=opponent, opponent_model=model,
        affects_model_selection=False,
        scope="Operational benchmark acceptance only; no result is supplied to graph or learner selection"))
    if not passed:
        raise RuntimeError("Attendance has not recovered its client win; the other tasks remain blocked")


def collect(args):
    import numpy as np
    from relarena.evaluation import compute_leaderboard

    identity = verify_profile(args.output)
    rows = []
    for dataset, task in identity["task_pairs"]:
        p = args.output / "tasks" / f"{dataset}_{task}" / "results.json"
        r = json.loads(p.read_text())
        if r["status"] != "completed" or r["run_profile"] != PROFILE or r["reused_previous_score"]:
            raise ValueError("All 21 fresh complete-profile scores are required")
        rows.append({key: r[key] for key in ["model", "dataset", "task", "metric", "test_score", "selected", "time_total"]})
    scores = pd.DataFrame(rows)
    base.validate_complete_scores(identity["task_pairs"], scores)
    panel = pd.read_csv(args.output / "baseline/selected-panel.csv")
    matrix = pd.concat([panel.loc[~panel.model.eq("kurversc")], scores], ignore_index=True)
    folder = args.output / "elo"
    folder.mkdir(parents=True, exist_ok=True)
    board = compute_leaderboard(matrix).sort_values("elo", ascending=False, kind="stable")
    board.insert(0, "elo_rank", np.arange(1, len(board) + 1))
    scores.to_csv(args.output / "campaign-results.csv", index=False)
    matrix.to_csv(folder / "input.csv", index=False)
    board.to_csv(folder / "leaderboard.csv")
    dump(folder / "results.json", dict(status="completed", run_profile=PROFILE,
        elo=float(board.loc["kurversc", "elo"]), rank=int(board.loc["kurversc", "elo_rank"]),
        tasks=21, carry_forwards=0, leaderboard=board.reset_index().to_dict("records")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["launch", "prepare", "snapshot", "task", "gate", "collect"], default="launch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--task")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.max_workers < 1:
        parser.error("Positive parallelism required")
    if args.mode == "task" and (args.dataset, args.task) not in base.task_pairs():
        parser.error("An explicit official task is required")
    if args.mode == "prepare" and args.dataset not in base.DATASETS:
        parser.error("An explicit official dataset is required")
    return dict(launch=launch, prepare=prepare, snapshot=base.snapshot,
        task=task, gate=gate, collect=collect)[args.mode](args)


if __name__ == "__main__":
    main()
