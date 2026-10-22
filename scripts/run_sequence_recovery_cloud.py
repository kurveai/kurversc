"""Optional production-matched sequence stage after the preserved full pipeline.

This is a separate candidate entry point, not a silent default change. The
existing graph search, temporal/readout choices, deadline and official scorer
remain intact. No task name or holdout score controls challenger eligibility.
"""

import json
import os
from pathlib import Path
import sys
import time

from kurversc.task_source import phase_source_root
from scripts.benchmark_production_sequence import CONTRACT
from scripts.run_gap_followup_queue import digest, dump


PROFILE = "staged-complete-production-sequence-v1"


def sequence_root(root, key):
    slug = key.replace("/", "_")
    return root / "workers" / slug / "temporal/tasks" / slug / "production-sequence"


def extend_recipe(staged, fresh):
    from scripts.run_kurversc_local_campaign import _run_guarded

    staged.PROFILE = PROFILE
    staged.CONTRACT = {**staged.CONTRACT, "production_sequence_readout": {**CONTRACT,
        "validation_workers": 1, "production_threads": 8}}
    old_task = fresh.task_main
    old_result, old_path, old_verify = fresh.official_result, fresh.official_path, staged.verify_stages

    def task_main(args):
        began = time.monotonic()
        key = f"{args.dataset}/{args.task}"
        budget = fresh.remaining_task_seconds(args, key)
        code = old_task(args)
        if code:
            return code
        root = sequence_root(args.output, key)
        root.mkdir(parents=True, exist_ok=True)
        if (root / "results.json").exists():
            return 0

        def stage(name, command, completion):
            if completion.exists():
                return
            remaining = budget-(time.monotonic()-began)
            if remaining <= 0:
                raise TimeoutError("Original complete-task deadline exhausted before sequence readout")
            with (root / f"{name}.log").open("a", buffering=1) as log:
                status, resources = _run_guarded(command, cwd=fresh.ROOT, environment=dict(os.environ),
                    log=log, minimum_available_memory_gb=0, poll_seconds=5, task_time_limit_seconds=remaining)
            dump(root / f"{name}-resources.json", resources)
            if status or not completion.exists():
                raise RuntimeError(f"Production sequence stage {name} incomplete: {status}")

        validation = root / "validation"
        stage("validation", [sys.executable, "-u", "-m", "scripts.benchmark_production_sequence",
            "--task-root", str(root.parent), "--source-root", str(phase_source_root(args.dataset)),
            "--output", str(validation), "--workers", "1", "--threads", "8"], validation / "selection.json")
        choice = json.loads((validation / "selection.json").read_text())
        admitted = choice["selected"] != "incumbent"
        if admitted:
            if not choice["complete_grid"] or choice["official_targets_used_for_selection"]:
                raise ValueError("An incomplete sequence choice cannot replace the incumbent")
            official = root / "official"
            stage("official", [sys.executable, "-u", "-m", "scripts.confirm_production_sequence",
                "--task-root", str(root.parent), "--source-root", str(phase_source_root(args.dataset, phase="outer")),
                "--validation", str(validation), "--output", str(official)], official / "results.json")
            report = json.loads((official / "results.json").read_text())
        else:
            report = {**old_result(args.output, key), "production_sequence_retained_incumbent": True}
        dump(root / "admission.json", dict(admitted=admitted, selected=choice["selected"],
            coefficients=choice["coefficients"], fallback=choice.get("fallback"),
            validation_selection_sha256=digest(validation / "selection.json"),
            official_targets_used_for_selection=False))
        dump(root / "results.json", report)
        return 0

    def official_result(root, key):
        path = sequence_root(root, key) / "results.json"
        return json.loads(path.read_text()) if path.exists() else old_result(root, key)

    def official_path(root, key):
        path = sequence_root(root, key) / "results.json"
        return path if path.exists() else old_path(root, key)

    def verify_stages(root, dataset, task):
        fresh.official_result = old_result
        try:
            previous, stages = old_verify(root, dataset, task)
        finally:
            fresh.official_result = official_result
        folder = sequence_root(root, f"{dataset}/{task}")
        admission = json.loads((folder / "admission.json").read_text())
        selected = json.loads((folder / "validation/selection.json").read_text())
        report = json.loads((folder / "results.json").read_text())
        if (admission["validation_selection_sha256"] != digest(folder / "validation/selection.json")
                or admission["selected"] != selected["selected"] or admission["official_targets_used_for_selection"]):
            raise ValueError("Production sequence validation admission changed")
        if admission["admitted"]:
            if (not selected["complete_grid"] or selected["selected"] != report["selected"]
                    or not report["native_exact"] or report["deployment_seed"] != 0):
                raise ValueError("Production sequence final fit differs from its admitted procedure")
            if report["query_keys_sha256"] != previous["query_keys_sha256"]:
                raise ValueError("Sequence readout changed the forecast request identities")
        elif (not report["production_sequence_retained_incumbent"]
                or json.dumps({k: report[k] for k in previous}, sort_keys=True, default=str)
                != json.dumps(previous, sort_keys=True, default=str)):
            raise ValueError("Sequence fallback did not preserve the prior procedure exactly")
        stages["production_sequence"] = admission
        return report, stages

    fresh.task_main, fresh.official_result, fresh.official_path = task_main, official_result, official_path
    staged.verify_stages = verify_stages
    return staged


def install_recipe():
    from scripts import run_production_finalist_cloud_v2 as base
    from scripts import run_fresh_cloud_campaign as fresh
    return extend_recipe(base.install_recipe(), fresh)


def main():
    from scripts import run_production_finalist_cloud_v2 as base
    from scripts import run_fresh_cloud_campaign as fresh
    original_install = base.install_recipe
    base.install_recipe = lambda: extend_recipe(original_install(), fresh)
    base.PROFILE, base.MODULE = PROFILE, "scripts.run_sequence_recovery_cloud"
    base.main()


if __name__ == "__main__":
    main()
