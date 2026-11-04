"""Generic finalist selection and native evaluation mechanics."""
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
from pathlib import Path
import pickle
import sys
from scripts.generic_staged_helpers import freeze,digest,dump

SUPPORTED_FINALIST_METRICS = ("mae", "roc_auc")


def finalist_loss(metric, score):
    """Put both supported validation metrics on a lower-is-better loss scale.

    MAE is already a loss. For binary ranking, 1 - AUC is the probability of
    misordering a positive/negative pair, counting tied predictions by half.
    A relative loss cap therefore has the same meaning for both task types.
    """
    score = float(score)
    if not math.isfinite(score):
        raise ValueError("A finite finalist validation score is required")
    if metric == "mae" and score >= 0:
        return score
    if metric == "roc_auc" and 0 <= score <= 1:
        return 1. - score
    raise ValueError("Finalist checks require MAE or binary ROC AUC with a valid score")


def finalist_case_score(metric, truth, prediction):
    """Use the native metric and sampling-uncertainty estimator for a fold."""
    import numpy as np
    import pandas as pd
    from kurversc.core import _prediction_standard_error, _score_prediction_chunks
    from kurversc.modeling import _binary_classes

    truth, prediction = np.asarray(truth), np.asarray(prediction, dtype=float)
    if (truth.ndim != 1 or prediction.shape != truth.shape or not truth.size
            or not np.isfinite(prediction).all() or pd.isna(truth).any()):
        raise ValueError("Complete finite aligned validation vectors required")
    if metric == "mae":
        truth = pd.to_numeric(pd.Series(truth)).to_numpy(dtype=float)
        if not np.isfinite(truth).all():
            raise ValueError("Finite mature validation targets required")
        task, classes = "regression", ()
    elif metric == "roc_auc":
        classes = tuple(_binary_classes(pd.Series(truth)))
        if len(classes) != 2:
            raise ValueError("Both binary classes are required for finalist ROC AUC")
        if pd.api.types.is_numeric_dtype(truth.dtype) and not np.isfinite(truth).all():
            raise ValueError("Finite mature validation targets required")
        task = "classification"
    else:
        raise ValueError("Finalist checks support MAE and binary ROC AUC")
    actual, score = _score_prediction_chunks([truth], [prediction], task=task, target_classes=classes)
    if actual != metric:
        raise ValueError("Native validation metric differs from the declared task metric")
    error = _prediction_standard_error([truth], [prediction], task=task, target_classes=classes)
    return score, error


def small_finalist(trials, incumbent, *, relative_recent_loss_cap=.01):
    if not 0 <= relative_recent_loss_cap <= .01:
        raise ValueError("A bounded recent relative validation-loss cap is required")
    completed = [(path, t) for path, t in trials if t.status == "completed"
        and t.stage == "rerank" and t.metric in SUPPORTED_FINALIST_METRICS and t.execution_plan
        and len(t.cutoff_scores) >= 2
        and all(v is not None and math.isfinite(v) for v in t.cutoff_scores)]
    same = [t for _, t in completed if t.config == incumbent]
    if not same:
        raise ValueError("The deployed graph must have complete full-fidelity evidence")
    metric = same[0].metric
    completed = [(p, t) for p, t in completed if t.metric == metric]
    width = same[0].feature_count
    newest = min(finalist_loss(metric, t.cutoff_scores[-1]) for _, t in completed)
    candidates = [(p, t) for p, t in completed if t.config != incumbent
        and t.feature_count < width
        and finalist_loss(metric, t.cutoff_scores[-1]) <= newest * (1 + relative_recent_loss_cap) + 1e-12]
    return min(candidates, key=lambda pair: (pair[1].feature_count,
        pair[1].config.complexity, json.dumps(asdict(pair[1].config), sort_keys=True),
        str(pair[0])), default=None)


@contextmanager
def normalized_engine_limits():
    # Same normalization used by the full staged executor; process-local only.
    import kurversc.core as core
    original = core._connection_scope

    @contextmanager
    def scope(connection, *, memory_limit=None, max_temp_directory_size=None):
        def normalize(v):
            return None if isinstance(v, str) and v.lower() in {"none", "null", "unlimited"} else v
        with original(connection, memory_limit=normalize(memory_limit),
                      max_temp_directory_size=normalize(max_temp_directory_size)) as con:
            yield con

    core._connection_scope = scope
    try:
        yield
    finally:
        core._connection_scope = original


def materialize(args):
    from scripts import benchmark_sparse_training as sparse
    chosen = json.loads((args.output / "plan.json").read_text())
    path = Path(chosen["checkpoint"])
    if digest(path) != chosen["checkpoint_sha256"]:
        raise ValueError("Selected finalist bytes changed")
    with path.open("rb") as handle:
        trial = pickle.load(handle)["trial"]
    original = sparse._checkpoint

    def checkpoint(dataset, task):
        if (dataset, task) != (args.dataset, args.task):
            raise ValueError("Unexpected dataset/task")
        return path, trial

    sparse._checkpoint = checkpoint
    sys.argv = [sparse.__file__, "--dataset", args.dataset, "--task", args.task,
        "--fold", "current", "--output", str(args.output / "task/cache"),
        "--atomic-date-replay", "--materialize-only", "--policy-origins", "3",
        "--frame-counts", "1,12", "--row-floors", "1000", "--max-frames", "64",
        "--duckdb-memory-limit", "none", "--duckdb-temp-limit", "none"]
    try:
        with normalized_engine_limits():
            sparse.main()
    finally:
        sparse._checkpoint = original


def native_audit(args):
    from kurversc.temporal_policy import TemporalFitPolicy
    from scripts import benchmark_temporal_policy as audit
    original = audit.bounded_temporal_policies
    # Audit only the already-declared native procedure: do not spend another
    # full temporal-policy search or silently select a different comparator.
    audit.bounded_temporal_policies = lambda task: (TemporalFitPolicy("incumbent"),)
    task_root = args.output / "task"
    sys.argv = [audit.__file__, "--cache", str(task_root / "cache"),
        "--output", str(task_root / "raw-audit"), "--seeds", "0,1,2",
        "--origins", "3", "--threads", "4", "--phase-frozen-cache", str(task_root / "snapshots")]
    try:
        audit.main()
    finally:
        audit.bounded_temporal_policies = original
    raw = json.loads((task_root / "raw-audit/manifest.json").read_text())
    freeze(task_root / "selection/manifest.json", raw)
    freeze(task_root / "selection/results.json", dict(selected_policy=asdict(TemporalFitPolicy("incumbent")),
        reason="Unchanged native frame-ensemble comparator, not a new temporal selection",
        official_test_accessed=False, model_defaults_changed=False))


