"""Public, task-scoped entry point for the complete staged prediction recipe."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .task_context import TaskContext
from .task_source import CONTEXT_ENV, ContextTaskSource


@dataclass(frozen=True)
class StagedTaskResult:
    output: Path

    @property
    def predictions(self):
        import pandas as pd
        return pd.read_parquet(self.output/"predictions.parquet")

    @property
    def report(self):
        return json.loads((self.output/"portable-results.json").read_text())


def _runtime_entry():
    root = Path(__file__).resolve().parents[1]
    if (root/"scripts/run_portable_staged.py").exists():
        return "scripts.run_portable_staged",root
    if importlib.util.find_spec("kurversc._staged") is not None:
        return "kurversc._staged.run_portable_staged",root
    raise ImportError("The complete staged runtime is missing; install a wheel built from the current source")


def run_staged_task(context, *, output,task_hours=23.75,minimum_available_memory_gib=8.,poll_seconds=2.):
    """Search, select, refit and forecast an explicit task in one owned process.

    ``context`` is a TaskContext or its checksummed context.json. Reuse that
    same bundle and output directory to resume; the original deadline does
    not reset. No hidden targets or benchmark scoring occurs. Predictions
    retain the caller's original entity keys and request order.

    This synchronous call supervises its own process group only. A resource
    stop preserves completed checkpoints and raises; it does not change model
    parameters, trim the graph, or claim a partial prediction as completed.
    """
    if not 0<task_hours<24 or minimum_available_memory_gib<0 or not 0<poll_seconds<=60:
        raise ValueError("Require 0 < task_hours < 24, a nonnegative RAM reserve and 0 < polling <= 60s")
    try:
        import psutil
    except ImportError as error:
        raise ImportError("Install kurversc[staged] for the supervised complete pipeline") from error
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    path = context.export(output/"input-context") if isinstance(context,TaskContext) else Path(context).resolve()
    # Export already records preparation time. Start this part's clock after
    # export so that preparing an in-memory context is not counted twice.
    started_epoch = time.time()
    source = ContextTaskSource(path)
    module,root = _runtime_entry()
    state_path = output/"portable-state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["task_hours"]!=task_hours:
            raise ValueError("Cannot reset the task deadline with a different budget")
    else:
        original = started_epoch-source.manifest.get("preprocessing_seconds",0.)
        state = dict(task_hours=task_hours,first_started_epoch=original,deadline_epoch=original+task_hours*3600)
        state_path.write_text(json.dumps(state,sort_keys=True)+"\n")
    if state["deadline_epoch"]<=time.time():
        raise TimeoutError("The original complete-procedure deadline is exhausted")
    # A frozen public recipe must not inherit a research shell's hidden tree,
    # frame, winner or selector overrides. Resource admission is explicit.
    environment = {k:v for k,v in os.environ.items() if not k.startswith("KURVERSC_")}
    environment[CONTEXT_ENV] = str(path)
    environment["PYTHONPATH"] = str(root)+os.pathsep+environment.get("PYTHONPATH","")
    command = [sys.executable,"-u","-m",module,"--context",str(path),"--output",str(output),"--hours",str(task_hours)]
    reason = None
    with (output/"execution.log").open("a",buffering=1) as log:
        child = subprocess.Popen(command,cwd=root,env=environment,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                if time.time()>=state["deadline_epoch"]:
                    reason = "total_task_time_budget"
                    break
                if psutil.virtual_memory().available/1024**3<minimum_available_memory_gib:
                    reason = "available_memory_reserve"
                    break
                time.sleep(min(poll_seconds,max(.01,state["deadline_epoch"]-time.time())))
        finally:
            # Portable stage subprocesses stay in this one owned process
            # group. Clean up descendants even if the root exited first.
            try:
                os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            if child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
    if reason:
        (output/"resource-stop.json").write_text(json.dumps(dict(reason=reason,
            checkpoints_preserved=True,original_deadline_epoch=state["deadline_epoch"]))+"\n")
        if reason=="total_task_time_budget":
            raise TimeoutError("The original complete-procedure deadline was reached; checkpoints preserved")
        raise MemoryError("Available RAM reached the declared reserve; only this task was stopped")
    if child.returncode or not (output/"portable-results.json").exists():
        raise RuntimeError(f"Complete staged task failed (exit={child.returncode}); see {output/'execution.log'}")
    return StagedTaskResult(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--hours",type=float,default=23.75)
    parser.add_argument("--minimum-available-memory-gib",type=float,default=8.)
    args = parser.parse_args()
    result = run_staged_task(args.context,output=args.output,task_hours=args.hours,
        minimum_available_memory_gib=args.minimum_available_memory_gib)
    print(json.dumps(dict(status=result.report["status"],predictions=str(result.output/"predictions.parquet"))))


if __name__=="__main__":
    main()
