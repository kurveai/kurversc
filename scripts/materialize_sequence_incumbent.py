"""Capture a selected graph's real unsampled fit/refit lifecycle and requests.

The public adapter supports explicit task contexts; the import fallback only
supports replaying older frozen runtimes which predate that adapter boundary.
No dataset name, target value, historical score, or graph recipe is consulted.
"""

import copy
import gc
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from scripts.benchmark_avito_quality_official import capture_prediction
from scripts.benchmark_avito_recovered_quality import capture_fit_inputs
from scripts.benchmark_frozen_event_summaries import _capture
from scripts.benchmark_signal_challengers import prepared
from scripts.run_gap_followup_queue import digest


def materialize(args, meta, policy, model_params):
    from kurversc.plan_dates import atomic_execution_plan
    try:
        from kurversc.staged_adapter import resolve_cache_config, KURVERSC_DEFAULTS, _SelectedGraph
        from kurversc.task_source import load_task_source
    except ModuleNotFoundError as exc:
        if exc.name not in {"kurversc.staged_adapter", "kurversc.task_source"}:
            raise
        from relarena.cache import resolve_cache_config
        from relarena.models.kurversc.model import KURVERSC_DEFAULTS, _SelectedGraph
        from relarena.dataset import RelBenchDatasetTask as load_task_source
    from scripts.run_kurversc_selected_relarena import AuditWinnerSystem

    if (policy.name != "incumbent" or policy.mode != "frame_ensemble" or policy.blend_weight != 1):
        raise ValueError("Faithful lifecycle capture supports the unmodified frame-ensemble incumbent")
    checkpoint = Path(meta["checkpoint"])
    if digest(checkpoint) != meta["checkpoint_sha256"]:
        raise ValueError("Validation-selected graph changed before prediction")
    trial = pickle.loads(checkpoint.read_bytes())["trial"]
    trial.execution_plan = atomic_execution_plan(trial.execution_plan)
    source = load_task_source(meta["dataset"], meta["task"], download=False)
    outer, task = source.outer_split(), source.task
    original_defaults = copy.deepcopy(KURVERSC_DEFAULTS)
    try:
        KURVERSC_DEFAULTS.update(full_training_frames=1, full_validation_frames=None,
            base_distribution_features=True, temporal_estimator_mode="frame_ensemble",
            duckdb_memory_limit="6GB", duckdb_max_temp_directory_size="80GB", model_params=model_params)
        system = AuditWinnerSystem(cache=resolve_cache_config(None, on_miss="raise"), run_identity=source.run_identity())
        system.peer_context_mode, system.peer_max_history_rows = "both", 50_000
        schema = meta["feature_schema"]
        with capture_fit_inputs() as inputs, _capture(schema["columns"], task.target_col,
                task.entity_col, task.entity_table) as (frames, _):
            fitted, problem = system._fit_arm(task, outer.db_state, outer.train_table, outer.val_table,
                seed=0, phase="outer", selection=_SelectedGraph(config=trial.config,
                    execution_plan=copy.deepcopy(trial.execution_plan)))
        actual, lifecycle = [], []
        for number, (x, y, kwargs, params) in enumerate(inputs):
            matches = []
            for frame in frames:
                if len(frame) != len(x) or not np.array_equal(frame[task.target_col], y):
                    continue
                try:
                    pd.testing.assert_frame_equal(prepared(frame, schema).reset_index(drop=True),
                        x.reset_index(drop=True), check_exact=True)
                except AssertionError:
                    continue
                matches.append(frame)
            if not matches:
                raise ValueError("An actual estimator member was sampled or has no materialized source frame")
            for other in matches[1:]:
                pd.testing.assert_frame_equal(matches[0].reset_index(drop=True), other.reset_index(drop=True), check_exact=True)
            actual.append(matches[0])
            lifecycle.append(dict(component=number, rows=len(x), cutoff=str(matches[0].__event_cutoff.iloc[0]),
                kwargs=kwargs, model_params=params))
        if not actual:
            raise ValueError("Native estimator inputs were not captured")
        train = pd.concat(actual, ignore_index=True)
        if train.duplicated(["__event_entity", "__event_cutoff"]).any():
            raise ValueError("Duplicated native training identities")
        with capture_prediction(schema, task.entity_col, task.entity_table) as chunks:
            prediction = system._predict(task, outer.eval_table, result=fitted, problem=problem)
        forecast = pd.concat(chunks, ignore_index=True)
        if sorted(forecast.__order.tolist()) != list(range(len(outer.eval_table.df))):
            raise ValueError("Masked request order changed")
        forecast = forecast.sort_values("__order", kind="stable").reset_index(drop=True)
        del fitted
        gc.collect()
    finally:
        KURVERSC_DEFAULTS.clear()
        KURVERSC_DEFAULTS.update(original_defaults)
    return source, outer, train, forecast, prediction, lifecycle
