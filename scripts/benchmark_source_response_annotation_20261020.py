"""Unchanged-production matched test of a frozen ordinary-event annotation.

No predictive-task labels train the source annotator. The native graph, source
rows, CatBoost budget, temporal policy and confidence gate are unchanged.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kurversc.temporal_policy import TemporalFitPolicy
from scripts import benchmark_latest_completed_peers_20261020 as benchmark
from scripts.benchmark_recovered_readouts import phase_frames
from scripts.engagement_local_comparison import freeze as freeze_file
from scripts.run_gap_followup_queue import digest
from scripts.schema_semantic_recurrence import declared_root
from scripts.source_response_annotation_20261020 import SourceResponseAnnotation


RECIPES = tuple(f"source_annotation_{w:g}" for w in benchmark.WEIGHTS)
ORIGINAL_CASE, ORIGINAL_FIT = benchmark.case, benchmark.fit_prediction


def prepare(args, raw, manifest):
    meta, policy = raw["metadata"], TemporalFitPolicy(**manifest["policy"])
    root, _, _ = declared_root(args.source_root / "tasks" / meta["task"] / "train.parquet", meta)
    for fold in reversed(manifest["folds"]):
        train, held, phase = phase_frames(args.task_root, raw, policy, fold)
        print(json.dumps(dict(phase="source_annotation", fold=fold)), flush=True)
        provider = SourceResponseAnnotation(args.source_root / "db", root,
            training_cutoff=pd.to_datetime(train.__event_cutoff).min(), horizon=meta["label_horizon"],
            temporary=args.output / f"fold{fold}/source-spill")
        try:
            extras = [provider.transform(f[["__event_entity", "__event_cutoff"]],
                phase_boundary=phase["fit_origin"]) for f in (train, held)]
            if list(extras[0]) != list(extras[1]) or len(extras[0].columns) != 30:
                raise ValueError("Frozen source annotation schema changed")
            folder = args.output / f"fold{fold}"
            folder.mkdir(exist_ok=True)
            paths = []
            for name, extra in zip(("train", "validation"), extras):
                path = folder / f"{name}-extra.parquet"
                extra.to_parquet(path, index=False)
                paths.append(path)
            freeze_file(folder / "producer.json", dict(fold=fold, phase=phase, provider=provider.audit,
                features_sha256={str(p): digest(p) for p in paths}, official_test_accessed=False))
        finally:
            provider.close()


def case(task_root, production, output, fold, seed):
    with np.load(Path(production) / f"fold{fold}/seed{seed}.npz") as arrays:
        native = arrays["control"].copy()

    def fit(x, y, times, vx, **kwargs):
        prediction, audit = ORIGINAL_FIT(x, y, times, vx, **kwargs)
        if "source_annotation_w0_count" in x:
            supported = vx.source_annotation_w0_count.fillna(0).to_numpy()>0
            prediction[~supported] = native[~supported]
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
        identity = dict(identity, profile="frozen_source_response_annotation_v1", recipes=RECIPES,
            maximum_new_columns=30, ordinary_binary_mark_supervision_only=True,
            source_model_training_rows=200_000, source_model_iterations=100,
            source_model_fit_before_first_consumed_training_anchor=True,
            native_predictions_on_unsupported_queries=True, new_graph_configurations=0,
            sources={**identity["sources"], **{str(p): digest(p) for p in (Path(__file__),
                Path(__import__(SourceResponseAnnotation.__module__, fromlist=["_"]).__file__))}})
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
