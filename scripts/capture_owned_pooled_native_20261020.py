"""Capture a freshly generated graph's official native path before scoring.

Decode only the exact checksummed trial under the explicitly supplied NEW run
root. Never scan or deserialize historical/untrusted checkpoint collections.
The ordinary official harness runs unchanged; hidden-target scoring is stopped.
"""

from contextlib import contextmanager
import copy
import gc
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

from scripts.run_gap_followup_queue import digest


class CaptureComplete(Exception):
    pass


def load_owned_trial(run_root, metadata):
    root = Path(run_root).resolve()
    checkpoint = Path(metadata["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(root) or not (root / "source-identity.json").is_file():
        raise ValueError("Only this explicitly supplied fresh run's owned checkpoint may be decoded")
    if digest(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Owned graph checksum changed; refuse to decode")
    # Trusted run-owned model cache, not an uploaded or discovered pickle.
    with checkpoint.open("rb") as handle:
        trial = pickle.load(handle)["trial"]  # noqa: S301 - exact owned producer + checksum boundary above
    schema = trial.execution_plan["kurversc_feature_schema"]
    reference = metadata["feature_schema"]
    if (trial.status != "completed" or list(schema["columns"]) != list(reference["columns"])
            or set(schema["categorical_columns"]) != set(reference["categorical_columns"])
            or set(schema["datetime_columns"]) != set(reference["datetime_columns"])):
        raise ValueError("Fresh owned graph/schema does not match validation")
    return checkpoint, trial


@contextmanager
def capture_frames(schema, *, target, entity, root):
    import kurversc.core as core
    previous = core._materialize_at_cutoff
    training, forecasts = {}, {}
    active = 0

    def materialize(*args, **kwargs):
        nonlocal active
        active += 1
        try:
            frame = previous(*args, **kwargs)
            if active != 1:
                return frame
            clock = str(pd.Timestamp(kwargs["cut_date"]))
            key = entity if entity in frame else f"{root}_{entity}"
            if key not in frame or set(schema["columns"])-set(frame):
                raise ValueError("Fresh native frame lost its frozen schema/routing key")
            data = frame[schema["columns"]].copy()
            data["__event_entity"], data["__event_cutoff"] = frame[key].to_numpy(), clock
            order = [c for c in frame if c.startswith("__kurversc_prediction_order__")]
            if order:
                if len(order) != 1:
                    raise ValueError("Exactly one forecast order marker required")
                data["__order"] = frame[order[0]].to_numpy()
                slot, identity = forecasts, (clock, tuple(data.__order.to_numpy()))
            else:
                if target not in frame:
                    raise ValueError("Mature feature capture lacks its training target")
                data[target] = frame[target].to_numpy()
                spec = args[6] if len(args)>6 else kwargs.get("label_spec")
                if spec is not None and getattr(spec, "available_at", None):
                    data["__kurversc_label_available_at"] = frame[spec.available_at].to_numpy()
                slot, identity = training, clock
            if identity in slot:
                pd.testing.assert_frame_equal(slot[identity], data, check_exact=True)
            else:
                slot[identity] = data
            return frame
        finally:
            active -= 1

    core._materialize_at_cutoff = materialize
    try:
        yield training, forecasts
    finally:
        core._materialize_at_cutoff = previous


def materialize(args, metadata):
    from kurversc.task_source import load_task_source
    from kurversc.staged_adapter import KURVERSC_DEFAULTS
    from scripts import benchmark_supported_official as harness
    from scripts import run_official_cloud as cloud

    if not args.task_root.resolve().is_relative_to(args.run_root.resolve()):
        raise ValueError("Task artifacts must belong to the explicitly supplied fresh run")
    checkpoint, trial = load_owned_trial(args.run_root, metadata)
    source = load_task_source(metadata["dataset"], metadata["task"], download=False)
    task = source.task
    original_checkpoint, original_argv = harness._checkpoint, sys.argv
    original_predict = harness.AuditWinnerSystem._predict
    predictions_completed = 0
    defaults = copy.deepcopy(KURVERSC_DEFAULTS)
    task_class = type(task)
    evaluator_owned = "evaluate" in task_class.__dict__
    original_evaluate = task_class.evaluate

    def owned_checkpoint(dataset, task_name):
        if (dataset, task_name) != (metadata["dataset"], metadata["task"]):
            raise ValueError("Unexpected task requested the fresh graph")
        return checkpoint, trial

    def intercept(original):
        def evaluate(self, prediction, target_table=None, *positional, **kwargs):
            if target_table is None:
                raise CaptureComplete("Stopped before any hidden-target score access")
            return original(self, prediction, target_table, *positional, **kwargs)
        return evaluate

    harness._checkpoint = owned_checkpoint
    task_class.evaluate = intercept(original_evaluate)
    try:
        with capture_frames(metadata["feature_schema"], target=task.target_col,
                entity=task.entity_col, root=task.entity_table) as (training, forecasts):
            def predict(self, *positional, **kwargs):
                nonlocal predictions_completed
                values = original_predict(self, *positional, **kwargs)
                predictions_completed += 1
                if predictions_completed == 1:
                    # The harness first fits/predicts the frame-ensemble
                    # comparator, then separately fits the selected pooled
                    # candidate. Same-cutoff reductions from these DIFFERENT
                    # fits need not be bitwise equal. Never conflate them;
                    # retain only the actual candidate's full source frames.
                    training.clear()
                    forecasts.clear()
                return values
            harness.AuditWinnerSystem._predict = predict
            sys.argv = ["official_native_capture", "--dataset", metadata["dataset"], "--task", metadata["task"],
                "--policy-audit", str(args.task_root / "selection/results.json"),
                "--output", str(args.output / "native-capture"), "--threads", "8", "--duckdb-threads", "8",
                "--duckdb-memory-limit", "32GB", "--duckdb-temp-limit", "256GB"]
            try:
                cloud.main()
            except CaptureComplete:
                pass
            else:
                raise ValueError("Official native capture did not reach the explicit pre-score stop")
    finally:
        harness._checkpoint = original_checkpoint
        harness.AuditWinnerSystem._predict = original_predict
        if evaluator_owned:
            task_class.evaluate = original_evaluate
        else:
            delattr(task_class, "evaluate")
        sys.argv = original_argv
        KURVERSC_DEFAULTS.clear()
        KURVERSC_DEFAULTS.update(defaults)
    if predictions_completed != 2:
        raise ValueError("The exact control-then-pooled-candidate lifecycle was not captured")
    if not training or not forecasts:
        raise ValueError("Native path did not expose complete training/forecast frames")
    train = pd.concat([training[k] for k in sorted(training)], ignore_index=True)
    query = pd.concat(forecasts.values(), ignore_index=True).sort_values("__order", kind="stable").reset_index(drop=True)
    if train.duplicated(["__event_entity", "__event_cutoff"]).any() or query.__order.duplicated().any():
        raise ValueError("Fresh native capture changed entity/time/request identity")
    outer = source.outer_split()
    if query.__order.tolist() != list(range(len(outer.eval_table.df))):
        raise ValueError("Native forecast requests are incomplete or reordered")
    from kurversc.prediction_provenance import artifact_keys, masked_query_keys, align_prediction
    keys = masked_query_keys(query, entity="__event_entity", timestamp="__event_cutoff", target=metadata["target"])
    with np.load(args.output / "native-capture/predictions.npz") as arrays:
        actual_native = align_prediction(arrays["candidate"], source_keys=artifact_keys(arrays), query_keys=keys)
    gc.collect()
    return source, outer, train, query, actual_native
