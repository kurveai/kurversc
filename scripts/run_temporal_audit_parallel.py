"""Parallelize independent seeds of the unchanged temporal-policy audit.

Each seed runs the original audit in a separate interpreter. A complete seed
barrier precedes collecting predictions and running the original all-seed
selector. This changes scheduling, not the fit or selection recipe.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if (
            hashlib.sha256(source.read_bytes()).digest()
            != hashlib.sha256(destination.read_bytes()).digest()
        ):
            raise ValueError(f"Conflicting completed artifact: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".copying")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def audit_arguments(args, *, output, seeds):
    values = [
        "--cache",
        str(args.cache),
        "--output",
        str(output),
        "--seeds",
        seeds,
        "--origins",
        str(args.origins),
        "--threads",
        str(args.threads),
    ]
    if args.phase_frozen_cache:
        values += ["--phase-frozen-cache", str(args.phase_frozen_cache)]
    return values


def run_seed(args, seed):
    output = args.output / "seed-jobs" / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    # Preserve completed fits from a serial run without importing its old
    # path-dependent manifest. The original audit checks truth alignment and
    # creates its own immutable manifest in the new seed-specific directory.
    for source in sorted(args.output.glob(f"fold*-seed{seed}.*")):
        if source.suffix in (".json", ".npz"):
            copy_verified(source, output / source.name)
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.benchmark_temporal_policy",
        *audit_arguments(args, output=output, seeds=str(seed)),
    ]
    with (output / "worker.log").open("a") as log:
        # Inherit the coordinator's process group so the task guard also
        # interrupts all seed subprocesses; never create orphan jobs.
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not (output / "results.json").exists():
        raise RuntimeError(f"Seed {seed} exited without a completed audit")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--origins", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--phase-frozen-cache", type=Path)
    args = parser.parse_args()
    seeds = list(map(int, args.seeds.split(",")))
    if len(set(seeds)) != len(seeds) or not seeds or args.workers < 1:
        parser.error("Distinct seeds and a positive worker count are required")
    args.output.mkdir(parents=True, exist_ok=True)
    execution = dict(
        workers=args.workers,
        seeds=seeds,
        threads=args.threads,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Independent seed processes; original all-seed audit after a completion barrier",
    )
    path = args.output / "parallel-execution.json"
    if path.exists() and json.loads(path.read_text()) != execution:
        raise ValueError("Parallel audit execution identity changed")
    path.write_text(json.dumps(execution, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=min(args.workers, len(seeds))) as pool:
        outputs = list(pool.map(lambda seed: run_seed(args, seed), seeds))
    for seed, output in zip(seeds, outputs):
        for source in sorted(output.glob(f"fold*-seed{seed}.*")):
            if source.suffix in (".json", ".npz"):
                copy_verified(source, args.output / source.name)
    # Original code performs every alignment check and computes the original
    # selector on the complete evidence. No first-finished winner is admitted.
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.benchmark_temporal_policy",
        *audit_arguments(args, output=args.output, seeds=args.seeds),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
