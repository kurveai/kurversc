"""Read-only resource observer and submission report for a frozen campaign.

Existing task-owned cgroup memory.peak is authoritative for full-task peak RAM.
CPU and task-directory disk measurements are sampled separately. Attaching to
an already running task never implies full coverage of its earlier CPU/disk
peaks. This observer has no authority to signal, retry or modify benchmark jobs.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def load(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def allocated_bytes(paths):
    existing = [str(p) for p in paths if p.exists()]
    if not existing:
        return 0
    result = subprocess.run(["du", "-s", "-B1", "--", *existing],
        capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Task disk sample unavailable: " + result.stderr[-300:])
    return sum(int(line.split()[0]) for line in result.stdout.splitlines())


def cpu_usage(group):
    path = group.resolve()
    if (not path.is_relative_to(Path("/sys/fs/cgroup"))
            or not path.name.startswith("kurversc-task-") or path.suffix != ".scope"):
        raise ValueError("CPU observation requires a dedicated task scope")
    try:
        stats = dict(line.split() for line in (path / "cpu.stat").read_text().splitlines())
        return int(stats["usage_usec"])
    except FileNotFoundError:
        return None


def host_inventory(args, identity):
    result = subprocess.run(["lscpu", "-J"], capture_output=True, text=True)
    disk = shutil.disk_usage(args.run)
    shared = {}
    for value in args.shared_directory:
        label, name = value.split("=", 1)
        shared[label] = dict(path=name, allocated_gib=allocated_bytes([Path(name)]) / 2**30)
    return dict(captured_at=now(), logical_cpus=identity["cpus"],
        ram_gib=identity["total_ram_gib"], filesystem_capacity_gib=disk.total / 2**30,
        filesystem_available_gib_at_observer_start=disk.free / 2**30,
        cpu=json.loads(result.stdout) if result.returncode == 0 else None,
        python=identity["python"], packages=identity["packages_complete"],
        kernel=os.uname().release, shared_directory_allocations=shared,
        gpu_used=False, cpu_pipeline=True,
        shared_storage_note="Existing raw-data, embedding and runtime caches; excludes old benchmark outputs. Allocated bytes can share hardlinks and are not a certified minimum disk requirement.")


def observe_attempt(path, observation, *, sample_disk, directories):
    metric = load(path)
    stamp, tick = now(), time.monotonic()
    if observation is None:
        observation = dict(first_observed_at=stamp,
            first_observed_task_elapsed_seconds=metric["elapsed_seconds"],
            cpu_sample_interval_seconds=1, disk_sample_interval_seconds=60,
            peak_sampled_busy_logical_cpus=None, peak_sampled_task_disk_bytes=None,
            resource_file=str(path), sampled_cpu_intervals=0, disk_samples=0)
    try:
        usage = cpu_usage(Path(metric["cgroup"]))
        if usage is not None:
            previous = observation.pop("_cpu_previous", None)
            if previous is not None:
                prior_tick, prior_usage = previous
                if tick > prior_tick and usage >= prior_usage:
                    cores = (usage - prior_usage) / 1e6 / (tick - prior_tick)
                    observation["peak_sampled_busy_logical_cpus"] = max(
                        observation["peak_sampled_busy_logical_cpus"] or 0., cores)
                    observation["sampled_cpu_intervals"] += 1
            observation["_cpu_previous"] = (tick, usage)
            observation["last_observed_lifetime_cpu_seconds"] = usage / 1e6
            observation["last_cpu_observation_at"] = stamp
    except (OSError, ValueError) as error:
        observation["last_cpu_sample_error"] = str(error)
    if sample_disk:
        try:
            size = allocated_bytes(directories)
            observation["peak_sampled_task_disk_bytes"] = max(
                observation["peak_sampled_task_disk_bytes"] or 0, size)
            observation.update(last_sampled_task_disk_bytes=size,
                last_disk_observation_at=stamp, disk_samples=observation["disk_samples"] + 1)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            observation["last_disk_sample_error"] = str(error)
    observation["last_observed_at"] = stamp
    observation["task_resource_report_finalized"] = "finished_at" in metric
    return metric, observation


def report(args, identity, manifest, observations, *, final=False):
    queue = load(args.run / "queue/state.json", {"jobs": {}})
    jobs = {j["task"]: j for j in manifest["jobs"] if "/" in j.get("task", "")}
    rows, lineage = [], {}
    for dataset, task in sorted(identity["task_pairs"]):
        key, slug = f"{dataset}/{task}", f"{dataset}_{task}"
        paths = sorted((args.run / "metrics" / slug).glob("attempt-*/resources.json"))
        metrics = [load(p) for p in paths]
        observed = [observations[str(p)] for p in paths if str(p) in observations]
        complete = bool(metrics) and all("finished_at" in d for d in metrics)
        status = queue["jobs"].get(slug, {}).get("status", "queued")
        max_ram = max((d["kernel_memory_peak_gib"] for d in metrics), default=None)
        max_rss = max((d["peak_process_tree_rss_gib"] for d in metrics), default=None)
        cpu_peaks = [d["peak_sampled_busy_logical_cpus"] for d in observed
            if d.get("peak_sampled_busy_logical_cpus") is not None]
        disk_peaks = [d["peak_sampled_task_disk_bytes"] / 2**30 for d in observed
            if d.get("peak_sampled_task_disk_bytes") is not None]
        row = dict(dataset=dataset, task=task, status=status, attempts=len(metrics),
            elapsed_hours=sum(d["elapsed_seconds"] for d in metrics) / 3600 if metrics else None,
            max_task_cgroup_ram_gib=max_ram, max_sampled_tree_rss_gib=max_rss,
            max_sampled_busy_logical_cpus=max(cpu_peaks, default=None),
            max_sampled_task_disk_gib=max(disk_peaks, default=None),
            observer_first_task_elapsed_seconds=min((d["first_observed_task_elapsed_seconds"] for d in observed), default=None),
            scheduler_ram_reservation_gib=jobs[key]["ram_gib"],
            scheduler_thread_slots=jobs[key]["threads"],
            duckdb_memory_limit=jobs[key].get("environment", {}).get("KURVERSC_DUCKDB_MEMORY_LIMIT"),
            resource_reports_finalized=complete, per_task_hard_ram_limit="none")
        rows.append(row)
        lineage[key] = dict(resources=[dict(path=str(p), sha256=digest(p) if "finished_at" in d else None)
            for p, d in zip(paths, metrics)],
            result=str(args.run / "tasks" / slug / "results.json"))
    completed = sum(r["status"] == "completed" and r["resource_reports_finalized"] for r in rows)
    value = dict(status="final" if final and completed == 21 else "provisional",
        generated_at=now(), fresh_run=str(args.run), completed_tasks=completed, expected_tasks=21,
        source_identity_sha256=digest(args.run / "source-identity.json"),
        host_inventory=str(args.output / "host-inventory.json"), tasks=rows, lineage=lineage,
        measurement_definitions=dict(
            ram="Exact kernel cgroup-v2 memory.peak for the full task, including descendants, charged file cache and kernel memory; running-task peaks are lower bounds until completion.",
            rss="One-second sampled sum of descendant resident memory; shared pages can be counted more than once.",
            cpu="Peak sampled cgroup CPU-time increase per wall second, expressed as busy logical CPUs. Eight scheduler thread slots are reservations, not CPU affinity caps.",
            disk="60-second sampled allocated storage in task-owned outputs, task TMPDIR/logs and metrics; excludes shared raw datasets, weights, runtime and earlier runs. Short-lived disk peaks may be missed.",
            coverage="Observer attachment time is recorded per task. CPU/disk peaks before attachment are unknown; full-task RAM peaks and elapsed times come from the original task monitor.",
            time="Full task wall time including search, materialization, temporal checks, fitting and official scoring; running tasks show elapsed time so far."),
        execution=dict(reserve_gib=manifest["reserve_gib"], max_task_workers=manifest["max_workers"],
            task_hours=manifest["task_hours"], automatic_retries=manifest.get("resource_pressure_retries", 0),
            gpu_used=False, node_ram_gib=identity["total_ram_gib"], logical_cpus=identity["cpus"]),
        minimum_hardware_claim="Observed usage and actual host configuration, not a certified minimum. Maintain headroom above measured peaks and account for shared input storage and concurrent jobs.")
    dump(args.output / "resources.json", value)
    csv_path = args.output / "resources.csv"
    with csv_path.with_suffix(".csv.tmp").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    csv_path.with_suffix(".csv.tmp").replace(csv_path)
    inventory = load(args.output / "host-inventory.json")
    cpu = {r["field"]: r["data"] for r in inventory.get("cpu", {}).get("lscpu", [])}
    lines = ["RELARENA SUBMISSION: PER-TASK RESOURCE UTILIZATION", "",
        f"Status: {value['status']}; {completed}/21 tasks completed in this fresh run.",
        f"Generated: {value['generated_at']}", f"Run: {args.run}",
        f"Source identity SHA256: {value['source_identity_sha256']}", "",
        f"Host: {cpu.get('Model name:', 'see host-inventory.json')}; {identity['cpus']} logical CPUs; {identity['total_ram_gib']:.2f} GiB RAM.",
        f"Filesystem capacity: {inventory['filesystem_capacity_gib']:.1f} GiB. CPU pipeline; no GPU used.",
        f"Concurrency: up to {manifest['max_workers']} tasks, RAM-aware; {manifest['reserve_gib']} GiB node reserve.",
        f"Task deadline: {manifest['task_hours']} hours; no hard per-task RAM cap; no automatic retry.", "",
        "Task                              Status        Hours   RAM GiB   RSS GiB  CPU cores*  Disk GiB*", "-" * 104]
    def formatted(value, digits=2):
        return "--" if value is None else f"{value:.{digits}f}"
    for r in rows:
        lines.append(f"{r['dataset']+'/'+r['task']:<33} {r['status']:<12} {formatted(r['elapsed_hours']):>6} "
            f"{formatted(r['max_task_cgroup_ram_gib']):>9} {formatted(r['max_sampled_tree_rss_gib']):>9} "
            f"{formatted(r['max_sampled_busy_logical_cpus']):>10} {formatted(r['max_sampled_task_disk_gib']):>10}")
    lines.extend(["", "RAM GiB is the full-task kernel memory.peak; 1 GiB = 2^30 bytes.",
        "Running-task RAM/time values are provisional; queued tasks have no measured values.",
        "RSS is a one-second sampled process-tree sum and may double-count shared pages.",
        "* CPU peaks are sampled at roughly one second; disk footprint at 60 seconds.",
        "CPU/disk coverage starts at observer attachment. See resources.csv for each task's attachment delay.",
        "Badge was already running when CPU/disk observation began; its earlier CPU/disk peaks are unknown.",
        "Task disk includes its owned outputs, TMPDIR/logs and metrics; shared inputs/runtime are separate.",
        "Scheduler thread slots and RAM reservations are not measured peaks or hard per-task limits.",
        "Observed peaks do not establish minimum hardware; keep RAM/disk headroom for reproduction.", "",
        "Shared storage baseline (allocated GiB, measured once; no old benchmark outputs):"])
    for label, entry in inventory["shared_directory_allocations"].items():
        lines.append(f"  {label}: {entry['allocated_gib']:.2f} GiB ({entry['path']})")
    lines.extend(["", "Raw original measurements: <run>/metrics/<dataset_task>/attempt-*/resources.json and samples.jsonl.",
        "Observer measurements: observations/*.json; machine/Python/package versions: host-inventory.json.",
        "This report contains fresh-run measurements only; previous campaign peaks are not substituted."])
    (args.output / "report.txt").write_text("\n".join(lines) + "\n")
    return value, queue


def main(args):
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError("Observer outputs must be separate from the frozen benchmark")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = load(args.run / "source-identity.json")
    source_sha = digest(args.run / "source-identity.json")
    manifest = load(args.run / "manifest.json")
    if len(identity["task_pairs"]) != 21:
        raise ValueError("Exactly 21 official tasks required")
    if not (args.output / "host-inventory.json").exists():
        dump(args.output / "host-inventory.json", host_inventory(args, identity))
    observations, last_disk = {}, {}
    for p in (args.output / "observations").glob("*.json"):
        d = load(p)
        observations[d["resource_file"]] = d
    last_report, began = 0., time.monotonic()
    while True:
        tick = time.monotonic()
        if tick - began > 7 * 86400:
            raise TimeoutError("Observation window ended before campaign completion")
        queue = load(args.run / "queue/state.json", {"jobs": {}})
        terminal = queue.get("status") in {"finished", "failed", "interrupted"}
        for path in args.run.glob("metrics/*/attempt-*/resources.json"):
            metric = load(path)
            key = str(path)
            old = observations.get(key)
            # Once a task report is finalized, its last observer sample suffices.
            if old and old.get("task_resource_report_finalized"):
                continue
            slug = path.parents[1].name
            disk_due = terminal or "finished_at" in metric or tick - last_disk.get(key, 0.) >= 60
            directories = [args.run / "tasks" / slug, args.run / "queue" / slug,
                args.run / "metrics" / slug]
            _, observed = observe_attempt(path, old, sample_disk=disk_due, directories=directories)
            observations[key] = observed
            if disk_due:
                last_disk[key] = tick
            if disk_due or tick - last_report >= 60:
                dump(args.output / "observations" / (slug + "-" + path.parent.name + ".json"),
                    {k:v for k,v in observed.items() if not k.startswith("_")})
        if args.once or terminal or tick - last_report >= 60:
            if digest(args.run / "source-identity.json") != source_sha:
                raise ValueError("Frozen run identity changed")
            value, queue = report(args, identity, manifest, observations, final=terminal)
            dump(args.output / "status.json", dict(status="completed" if value["status"] == "final" else
                "campaign_ended_incomplete" if terminal else "observing", checked_at=now(),
                completed_tasks=value["completed_tasks"], report=str(args.output / "report.txt"),
                benchmark_altered=False))
            last_report = tick
            if args.once or terminal:
                return
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-directory", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        dump(args.output / "status.json", dict(status="observer_failed", error=str(error), benchmark_altered=False))
        raise
