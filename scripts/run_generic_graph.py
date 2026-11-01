"""Schema-generic production-faithful finalist comparison; no recovery stages."""
import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import time
from scripts.run_gap_followup_queue import digest,dump

def finalized_graph_row(original, decision, *, validation_report):
    """Never attach the old graph's official score to a new selected graph."""
    row = dict(original)
    row.update(config=json.dumps(decision.get("config", json.loads(original["config"])), sort_keys=True),
        production_finalist_decision=decision["selected"],
        production_finalist_validation=str(validation_report),
        audit_source="Fresh staged search followed by bounded production-faithful recent-origin finalist comparison")
    if decision["selected"] != "incumbent":
        for key in ("test_score", "improvement", "beats_published_kurversc", "plan_fingerprint",
                    "inner_validation_score", "inner_full_validation_score"):
            row[key] = None
        row["graph_official_score_pending"] = True
    return row


def graph_stage(arguments):
    from scripts.run_fresh_cloud_campaign import ROOT
    from scripts.run_kurversc_local_campaign import _run_guarded
    import pandas as pd

    def value(flag):
        return arguments[arguments.index(flag)+1]

    output = Path(value("--output")).resolve()
    dataset, task = value("--dataset"), value("--task")
    graph_root = output.parent.parent
    check = graph_root / "production-finalist-check"
    original_output = check / "search-stage" / output.name
    original_result = original_output.with_suffix(".json")
    started = time.monotonic()
    check.mkdir(parents=True, exist_ok=True)
    if not original_result.exists():
        search = list(arguments)
        search[search.index("--output")+1] = str(original_output)
        # The enclosing full-task process guard owns the cumulative deadline.
        code = subprocess.call([sys.executable, "-u", "-m",
            "scripts.run_kurversc_selected_relarena", *search], cwd=ROOT, env=dict(os.environ))
        if code or not original_result.exists():
            raise RuntimeError(f"Fresh graph search incomplete: {code}")
    validation = check / "validation"
    completion = validation / "results.json"
    if not completion.exists():
        with (check / "validation.log").open("a", buffering=1) as log:
            code, usage = _run_guarded([sys.executable, "-u", "-m", "scripts.benchmark_production_finalists_v2",
                "--graph-root", str(graph_root), "--search-result", str(original_result),
                "--dataset", dataset, "--task", task, "--output", str(validation)],
                cwd=ROOT, environment=dict(os.environ), log=log, minimum_available_memory_gb=0,
                poll_seconds=5, task_time_limit_seconds=4*3600)
        dump(check / "validation-resources.json", usage)
        if code or not completion.exists():
            raise RuntimeError(f"Production-faithful finalist validation incomplete: {code}")
    decision = json.loads(completion.read_text())
    if "checkpoint" in decision and digest(decision["checkpoint"]) != decision["checkpoint_sha256"]:
        raise ValueError("Validation-selected fresh checkpoint changed")
    row = finalized_graph_row(json.loads(original_result.read_text()), decision, validation_report=completion)
    row["production_finalist_graph_stage_seconds"] = time.monotonic()-started
    # This is the first publication of the selected graph to downstream stages.
    # A crash before this point leaves resumable search/check caches, not an
    # apparently completed graph result that silently skipped the comparison.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    pd.DataFrame([row]).to_csv(temporary, index=False)
    temporary.replace(output)
    dump(output.with_suffix(".json"), row)
    print(json.dumps(dict(event="production_finalist_graph_selected", dataset=dataset,
        task=task, selected=decision["selected"], validation=str(completion))), flush=True)


if __name__ == "__main__":
    arguments=sys.argv[1:]
    if arguments and arguments[0]=="--":arguments=arguments[1:]
    graph_stage(arguments)
