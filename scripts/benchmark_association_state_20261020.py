"""Matched validation of bounded relational co-occurrence state, not labels.

Reuse the complete production-native audit and existing 300-tree frame policy.
Only <=32 training-only association coordinates are appended. Unsupported
query neighborhoods retain native forecasts exactly; no graph lattice grows.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kurversc.temporal_policy import TemporalFitPolicy
from scripts import benchmark_latest_completed_peers_20261020 as benchmark
from scripts.association_state_compression_20261020 import AssociationState
from scripts.benchmark_recovered_readouts import phase_frames
from scripts.engagement_local_comparison import freeze as freeze_file
from scripts.run_gap_followup_queue import digest, dump


RECIPES = tuple(f"association_state_{w:g}" for w in benchmark.WEIGHTS)
ORIGINAL_CASE, ORIGINAL_FIT = benchmark.case, benchmark.fit_prediction


def prepare(args, raw, manifest):
    meta = raw["metadata"]
    source = benchmark.ValidationOnlySource(meta["dataset"], meta["task"])
    inner, task = source.inner_split(), source.task
    problem = benchmark.kurversc.relbench_problem_from_objects(task, inner.db_state, inner.train_table, inner.eval_table,
        dataset_name=meta["dataset"], task_name=meta["task"], sample_rows=1000,
        search_training_frames=1, max_train_timestamps=3, schema_depth=4,
        max_base_table_occurrences=1, include_target_history=False, random_state=0)
    root, tables = benchmark._normalize_tables(problem.parent_node, problem.tables)
    policy = TemporalFitPolicy(**manifest["policy"])
    for fold in reversed(manifest["folds"]):
        train, held, phase = phase_frames(args.task_root, raw, policy, fold)
        print(json.dumps(dict(phase="association_memberships_and_basis", fold=fold)), flush=True)
        provider = AssociationState()
        extra, support = provider.fit_transform(train, held, tables=tables,
            relationships=problem.relationships, root_name=root, phase_boundary=phase["fit_origin"])
        if list(extra[0]) != list(extra[1]) or not 0 < len(extra[0].columns) <= 32:
            raise ValueError("Bounded association schema changed")
        folder = args.output / f"fold{fold}"
        folder.mkdir(exist_ok=True)
        paths = []
        for name, frame in zip(("train", "validation"), extra):
            path = folder / f"{name}-extra.parquet"
            frame.to_parquet(path, index=False)
            paths.append(path)
        freeze_file(folder / "producer.json", dict(fold=fold, phase=phase, provider=provider.audit,
            supported_rows=[int(mask.sum()) for mask in support],
            features_sha256={str(p): digest(p) for p in paths}, original_native_payload_unchanged=True))
        del train, held, extra, provider
        gc.collect()


def case(task_root, production, output, fold, seed):
    with np.load(Path(production) / f"fold{fold}/seed{seed}.npz") as a:
        native = a["control"].copy()

    def fit(x, y, times, vx, **kwargs):
        prediction, audit = ORIGINAL_FIT(x, y, times, vx, **kwargs)
        columns = [c for c in x if c.startswith("association_state_p")]
        if columns:
            supported = np.any(vx[columns].to_numpy()!=0, axis=1)
            prediction[~supported] = native[~supported]
            audit = dict(audit, unsupported_incidence_native_fallback=True,
                supported_query_rows=int(supported.sum()))
        return prediction, audit

    previous = benchmark.fit_prediction, benchmark.RECIPES
    benchmark.fit_prediction, benchmark.RECIPES = fit, RECIPES
    try:
        return ORIGINAL_CASE(task_root, production, output, fold, seed)
    finally:
        benchmark.fit_prediction, benchmark.RECIPES = previous


def run(args):
    previous = benchmark.prepare, benchmark.case, benchmark.freeze, benchmark.RECIPES

    def freeze(path, identity):
        identity = dict(identity, profile="bounded_training_only_association_state_v1", recipes=RECIPES,
            maximum_new_columns=32, maximum_members_per_example_per_hop=32, targets_used_by_representation=False,
            native_fallback_on_unknown_objects=True, new_graph_configurations=0,
            sources={**identity["sources"], str(Path(__file__)): digest(__file__),
                str(Path(__import__(AssociationState.__module__, fromlist=["_"]).__file__)): digest(
                    Path(__import__(AssociationState.__module__, fromlist=["_"]).__file__))})
        return previous[2](path, identity)

    benchmark.prepare, benchmark.case, benchmark.freeze, benchmark.RECIPES = prepare, case, freeze, RECIPES
    try:
        return benchmark.run(args)
    finally:
        benchmark.prepare, benchmark.case, benchmark.freeze, benchmark.RECIPES = previous


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root", "source-root", "production", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.workers <= 3:
        p.error("One to three independent case workers required")
    run(args)
