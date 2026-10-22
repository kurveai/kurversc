"""Explicit execution allocation or validation-pinned learner replay.

Legacy invocation preserves its original behavior. Thread count is NOT
guaranteed numerically inert: strict replay can instead preserve the model
parameters recorded by validation, independently of the process CPU budget.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import kurversc.core as core
from scripts import benchmark_supported_official as benchmark


def learner_replay_parameters(*, thread_budget, validation_manifest=None):
    """Never silently turn an execution allocation into a changed frozen fit."""
    if isinstance(thread_budget,bool) or not isinstance(thread_budget,int) or thread_budget<1:
        raise ValueError("A positive learner thread budget is required")
    if validation_manifest is None:
        return dict(thread_count=thread_budget)
    params = dict(validation_manifest.get("model_params") or {})
    threads = params.get("thread_count")
    if isinstance(threads,bool) or not isinstance(threads,int) or threads<1:
        raise ValueError("Strict replay requires an explicit validation thread count")
    if threads>thread_budget:
        raise ValueError("CPU budget cannot reproduce the recorded learner thread count")
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--duckdb-threads", type=int, default=32)
    parser.add_argument("--replay-validation-model-params",action="store_true",
        help="Opt-in: keep validation's direct CatBoost fit parameters; threads is only a resource ceiling")
    args, remaining = parser.parse_known_args()
    if min(args.threads, args.duckdb_threads) < 1:
        parser.error("Thread counts must be positive")
    validation = None
    if args.replay_validation_model_params:
        if "--policy-audit" not in remaining:
            parser.error("Strict learner replay requires --policy-audit")
        evidence_path = Path(remaining[remaining.index("--policy-audit")+1])
        audit = json.loads(evidence_path.read_text())
        if audit["selected_policy"]["readout"]!="direct":
            parser.error("Strict parameter replay currently covers direct CatBoost readouts only")
        validation = json.loads((evidence_path.parent/"manifest.json").read_text())
    replay = learner_replay_parameters(thread_budget=args.threads,validation_manifest=validation)
    output = Path(remaining[remaining.index("--output") + 1])
    output.mkdir(parents=True, exist_ok=True)
    execution = dict(
        catboost_threads=replay["thread_count"],
        duckdb_threads=args.duckdb_threads,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Frozen graph/support/seed/readout; thread allocation can change CatBoost numerical fits",
    )
    if validation is not None:
        execution.update(replay_validation_model_params=True,learner_model_params=replay,
            validation_manifest_sha256=hashlib.sha256((evidence_path.parent/"manifest.json").read_bytes()).hexdigest(),
            learner_cpu_budget=args.threads)
    (output / "cloud-execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    original_fit, original_scope = core.fit_final_catboost, core._connection_scope
    original_benchmark_fit = benchmark.fit_final_catboost

    def fit(*positional, **kwargs):
        kwargs["model_params"] = {
            **(kwargs.get("model_params") or {}),
            **replay,
        }
        return original_fit(*positional, **kwargs)

    @contextmanager
    def connection(*positional, **kwargs):
        with original_scope(*positional, **kwargs) as con:
            con.sql(f"SET threads={args.duckdb_threads}")
            yield con

    core.fit_final_catboost = fit
    benchmark.fit_final_catboost = fit
    core._connection_scope = connection
    try:
        sys.argv = ["benchmark_supported_official", *remaining]
        benchmark.main()
    finally:
        core.fit_final_catboost = original_fit
        core._connection_scope = original_scope
        benchmark.fit_final_catboost = original_benchmark_fit


if __name__ == "__main__":
    main()
