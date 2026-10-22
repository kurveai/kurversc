"""One frozen staged recipe for portable predictions and fresh benchmarks.

The bounded-view extension is validation-only, schema-derived, and preserves
already-admitted readouts. Source identities precede all fits. Resource settings
govern spilling, never feature/source sampling or a task's predictive budget.
"""

import json
import os
from pathlib import Path
import sys
import time

from kurversc.staged_runtime import staged_source_identity
from kurversc.task_source import phase_source_root
from scripts.engagement_local_comparison import freeze
from scripts.run_gap_followup_queue import digest, dump


PROFILE = "staged-complete-reproducible-v1"
BOUNDED_CONTRACT = dict(optional=True, maximum_added_columns=112,
    fixed_equal_three_view_mean=True, weights=[.125, .5, 1.], allowance=1024,
    complete_available_origins=True, seeds=[0, 1, 2], deployment_seed=0,
    native_threads=8, validation_workers=1, maximum_feature_cells=125_000_000,
    maximum_rows=250_000, no_graph_configurations_added=True,
    preserve_already_admitted_procedures=True, exact_native_replay_required=True,
    historical_winner_lookup=False, official_targets_used_for_selection=False)
STORAGE_POLICY = dict(duckdb_working_memory="min(128, floor(host GiB / 4)) GiB",
    task_rss_cap=None, source_sampling_changed=False, model_budget_changed=False)


def spill_threshold(total_bytes):
    return f"{max(1, min(128, int(total_bytes / 2**30 / 4)))}GiB"


def bounded_root(root, key):
    slug = key.replace("/", "_")
    return root / "workers" / slug / "temporal/tasks" / slug / "bounded-views"


def source_identity(code_root):
    """Make the packaged transitive source identity resolvable for replay."""
    mapping = staged_source_identity(code_root)
    paths = {}
    for name, sha in mapping.items():
        if name.startswith("relarena/"):
            import relarena
            file = Path(relarena.__file__).parent / name[len("relarena/"):]
        else:
            file = code_root / name
        if digest(file) != sha:
            raise ValueError("Complete staged code changed before native graph production")
        paths[str(file)] = sha
    import graphreduce
    for file in sorted(Path(graphreduce.__file__).parent.rglob("*.py")):
        paths[str(file)] = digest(file)
    return dict(profile=PROFILE, source_sha256=paths)


def extend_recipe(staged, fresh):
    from scripts.run_kurversc_local_campaign import _run_guarded
    import psutil

    staged.PROFILE = PROFILE
    staged.CONTRACT = {**staged.CONTRACT, "bounded_complementary_views": BOUNDED_CONTRACT,
        "execution_allocation": STORAGE_POLICY}
    old_task = fresh.task_main
    old_result, old_path, old_verify = fresh.official_result, fresh.official_path, staged.verify_stages

    def task_main(args):
        began = time.monotonic()
        key = f"{args.dataset}/{args.task}"
        budget = fresh.remaining_task_seconds(args, key)
        if budget <= 0:
            raise TimeoutError("Original complete-task deadline exhausted before bounded-view stage")
        args.output.mkdir(parents=True, exist_ok=True)
        freeze(args.output / "source-identity.json", source_identity(fresh.ROOT))
        engine = spill_threshold(psutil.virtual_memory().total)
        os.environ["KURVERSC_DUCKDB_MEMORY_LIMIT"] = engine
        freeze(args.output / "execution-allocation.json", dict(duckdb_spill_threshold=engine,
            host_bytes=psutil.virtual_memory().total, task_rss_cap=None,
            source_sampling_changed=False, model_budget_changed=False))
        code = old_task(args)
        if code:
            return code
        root = bounded_root(args.output, key)
        root.mkdir(parents=True, exist_ok=True)
        if (root / "results.json").exists():
            return 0

        def stage(name, command, completion):
            if completion.exists():
                return
            remaining = budget-(time.monotonic()-began)
            if remaining <= 0:
                raise TimeoutError("Original complete-task deadline exhausted during bounded-view stage")
            with (root / f"{name}.log").open("a", buffering=1) as log:
                status, resources = _run_guarded(command, cwd=fresh.ROOT, environment=dict(os.environ),
                    log=log, minimum_available_memory_gb=0, poll_seconds=5, task_time_limit_seconds=remaining)
            dump(root / f"{name}-resources.json", resources)
            if status or not completion.exists():
                raise RuntimeError(f"Bounded-view stage {name} incomplete: {status}")

        validation = root / "validation"
        stage("validation", [sys.executable, "-u", "-m", "scripts.run_bounded_view_validation_20261020",
            "--task-root", str(root.parent), "--source-root", str(phase_source_root(args.dataset)),
            "--output", str(validation), "--workers", "1"], validation / "results.json")
        choice = json.loads((validation / "results.json").read_text())
        admitted = choice["selected"] != "incumbent"
        if admitted:
            if not choice["complete_grid"] or choice["official_test_accessed"] is not False:
                raise ValueError("Incomplete or target-seen bounded-view choice cannot replace the incumbent")
            official = root / "official"
            stage("official", [sys.executable, "-u", "-m", "scripts.confirm_bounded_view_recovery_20261020",
                "--task-root", str(root.parent), "--run-root", str(args.output),
                "--source-root", str(phase_source_root(args.dataset, phase="outer")),
                "--validation", str(validation), "--output", str(official)], official / "results.json")
            report = json.loads((official / "results.json").read_text())
        else:
            report = {**old_result(args.output, key), "bounded_views_retained_incumbent": True}
        dump(root / "admission.json", dict(admitted=admitted, selected=choice["selected"],
            fallback=choice.get("fallback"), validation_sha256=digest(validation / "results.json"),
            official_targets_used_for_selection=False))
        dump(root / "results.json", report)
        return 0

    def official_result(root, key):
        path = bounded_root(root, key) / "results.json"
        return json.loads(path.read_text()) if path.exists() else old_result(root, key)

    def official_path(root, key):
        path = bounded_root(root, key) / "results.json"
        return path if path.exists() else old_path(root, key)

    def verify_stages(root, dataset, task):
        fresh.official_result = old_result
        try:
            previous, stages = old_verify(root, dataset, task)
        finally:
            fresh.official_result = official_result
        folder = bounded_root(root, f"{dataset}/{task}")
        admission = json.loads((folder / "admission.json").read_text())
        selected = json.loads((folder / "validation/results.json").read_text())
        report = json.loads((folder / "results.json").read_text())
        if (admission["validation_sha256"] != digest(folder / "validation/results.json")
                or admission["selected"] != selected["selected"] or admission["official_targets_used_for_selection"]):
            raise ValueError("Bounded-view validation admission changed")
        if admission["admitted"]:
            if (not selected["complete_grid"] or selected["selected"] != report["selected"]
                    or not report["native_exact"] or report["deployment_seed"] != 0
                    or report["query_keys_sha256"] != previous["query_keys_sha256"]):
                raise ValueError("Bounded-view final fit differs from the admitted native-keyed procedure")
        elif (not report["bounded_views_retained_incumbent"]
                or json.dumps({k: report[k] for k in previous}, sort_keys=True, default=str)
                != json.dumps(previous, sort_keys=True, default=str)):
            raise ValueError("Bounded-view fallback changed the previous native procedure")
        stages["bounded_views"] = admission
        return report, stages

    fresh.task_main, fresh.official_result, fresh.official_path = task_main, official_result, official_path
    staged.verify_stages = verify_stages
    return staged


def install_recipe(*, install_base=None):
    from scripts import run_streamed_sequence_cloud as streamed
    from scripts import run_fresh_cloud_campaign as fresh
    return extend_recipe(streamed.install_recipe(install_base=install_base), fresh)


def main():
    from scripts import run_production_finalist_cloud_v2 as base
    original_install = base.install_recipe
    base.install_recipe = lambda: install_recipe(install_base=original_install)
    base.PROFILE, base.MODULE = PROFILE, "scripts.run_reproducible_staged_cloud"
    base.main()


if __name__ == "__main__":
    main()
