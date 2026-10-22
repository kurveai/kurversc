"""Official replay of a validation-qualified, generic count-median challenger.

Record the original production sampler's actual member inputs and masked
forecast order. Refit full-budget controls and auxiliary heads separately.
Both the native vector and already-admitted count projection must reproduce
before the single validation-locked candidate is scored. No graph lookup,
test-selected blend, source-storage substitution, or core default change.
"""

import argparse
import copy
import gc
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from kurversc.modeling import fit_final_catboost
from kurversc.prediction_provenance import (
    align_prediction, artifact_keys, key_arrays, key_fingerprint, masked_query_keys,
)
from scripts.benchmark_signal_challengers import prepared
from scripts.count_challenger_protocol import count_stage_eligible, validated_count_recipe
from scripts.count_median_faithful_inputs import (
    capture_masked_frames, capture_sampled_inputs, canonical_vectors, selected_members,
)
from scripts.discrete_target_projection import TargetDomainProjection
from scripts.engagement_local_comparison import freeze
from scripts.poisson_count_median import PoissonCountMedian
from scripts.run_gap_followup_queue import digest, dump


def verify(files):
    if any(digest(path) != expected for path, expected in files.items()):
        raise ValueError("Frozen count-challenger source or evidence changed")


def materialize(raw, output):
    from kurversc.plan_dates import atomic_execution_plan
    from kurversc.staged_adapter import resolve_cache_config
    from kurversc.task_source import load_task_source as RelBenchDatasetTask
    from kurversc.staged_adapter import KURVERSC_DEFAULTS, _SelectedGraph
    from scripts.run_kurversc_selected_relarena import AuditWinnerSystem

    meta = raw["metadata"]
    with Path(meta["checkpoint"]).open("rb") as stream:
        trial = pickle.load(stream)["trial"]
    trial.execution_plan = atomic_execution_plan(trial.execution_plan)
    previous = copy.deepcopy(KURVERSC_DEFAULTS)
    try:
        source = RelBenchDatasetTask(meta["dataset"], meta["task"], download=False)
        outer, task = source.outer_split(), source.task
        source._db = outer.db_state
        if hasattr(source._dataset,"get_db"):
            source._dataset.get_db.cache_clear()
        expected = masked_query_keys(outer.eval_table.df, entity=task.entity_col,
            timestamp=task.time_col, target=meta["target"])
        expected.to_parquet(output / "expected-query-keys.parquet", index=False)
        observed = pd.concat([outer.train_table.df, outer.val_table.df], ignore_index=True)
        projector = TargetDomainProjection().fit(observed[meta["target"]], observed[task.time_col],
            origin=outer.cutoff, horizon=meta["label_horizon"])
        if not projector.integer_counts:
            raise ValueError("Mature production targets must preserve the integral domain")
        del observed
        # These one-tree models ONLY drive the original materializer/sampler.
        # Their outputs are discarded; every scored member is refit below.
        KURVERSC_DEFAULTS.update(full_training_frames=1, full_validation_frames=None,
            base_distribution_features=True, temporal_estimator_mode="frame_ensemble",
            duckdb_memory_limit=None, duckdb_max_temp_directory_size=None,
            model_params=dict(iterations=1, thread_count=1))
        system = AuditWinnerSystem(cache=resolve_cache_config(None, on_miss="raise"),
            run_identity=source.run_identity())
        system.peer_context_mode, system.peer_max_history_rows = "both", 50_000
        print(json.dumps(dict(phase="capture_actual_sampled_native_members")), flush=True)
        with capture_sampled_inputs(output) as (members, identities):
            fitted, problem = system._fit_arm(task, outer.db_state, outer.train_table, outer.val_table,
                seed=0, phase="outer", selection=_SelectedGraph(config=trial.config,
                    execution_plan=copy.deepcopy(trial.execution_plan)))
        if list(fitted.fitted_model.feature_columns) != meta["feature_schema"]["columns"]:
            raise ValueError("Native predictor order changed")
        selected = selected_members(fitted.fitted_model.estimator, identities)
        with capture_masked_frames(output, meta["feature_schema"], entity=task.entity_col,
                                   root=task.entity_table) as frames:
            system._predict(task, outer.eval_table, result=fitted, problem=problem)
        info = dict(selected_members=selected, members=members, forecasts=frames,
            projection=projector.audit, original_sampler_used=True,
            one_tree_predictions_discarded=True, source_storage_unchanged=True)
        dump(output / "materialized-inputs.json", info)
        source._db = None
        if hasattr(source._dataset,"get_db"):
            source._dataset.get_db.cache_clear()
        del source, outer, fitted, problem, system
        gc.collect()
        return task, projector, info
    finally:
        KURVERSC_DEFAULTS.clear()
        KURVERSC_DEFAULTS.update(previous)


def run(args):
    started = time.monotonic()
    raw = json.loads((args.task_root / "raw-audit/manifest.json").read_text())
    policy = json.loads((args.task_root / "selection/results.json").read_text())["selected_policy"]
    recovery = args.task_root / "recovered-readouts"
    recovered = json.loads((recovery / "selection.json").read_text())
    if not count_stage_eligible(raw, policy, recovered):
        raise ValueError("This count procedure must not replace a different admitted readout")
    result = json.loads((args.validation / "results.json").read_text())
    manifest = json.loads((args.validation / "manifest.json").read_text())
    choice = validated_count_recipe(result, manifest, raw["metadata"])
    if choice is None:
        raise ValueError("The challenger was not admitted; preserve the incumbent")
    paths = [Path(__file__), Path(__file__).with_name("count_challenger_protocol.py"),
        args.validation / "manifest.json", args.validation / "results.json",
        args.task_root / "raw-audit/manifest.json", args.task_root / "selection/results.json",
        recovery / "selection.json", recovery / "results.json", recovery / "predictions.npz",
        Path("scripts/count_median_faithful_inputs.py"), Path("scripts/poisson_count_median.py"),
        Path("scripts/discrete_target_projection.py"), Path("kurversc/modeling.py")]
    files = {str(p.resolve()): digest(p) for p in paths}
    files.update(manifest["sources"])
    files.update(manifest["inputs"])
    verify(files)
    args.output.mkdir(parents=True, exist_ok=True)
    freeze(args.output / "selection.json", {**choice, "files":files,
        "metadata":raw["metadata"], "policy":policy, "model_defaults_changed":False,
        "scope":"Validation-only locked conditional-count challenger, before official replay"})
    lock = digest(args.output / "selection.json")
    if (args.output / "results.json").exists():
        existing = json.loads((args.output / "results.json").read_text())
        if existing["selection_sha256"] != lock:
            raise ValueError("Existing result has a different validation lock")
        return existing
    task, projector, info = materialize(raw, args.output)
    meta = raw["metadata"]
    expected = pd.read_parquet(args.output / "expected-query-keys.parquet")
    native_predictions, count_predictions, routing = [], [], None
    params = dict(raw["model_params"], thread_count=8)
    for index, member in enumerate(info["selected_members"]):
        record = info["members"][member]
        folder = args.output / f"member{member}"
        if (digest(folder / "features.parquet") != record["features_sha256"]
                or digest(folder / "labels.parquet") != record["labels_sha256"]):
            raise ValueError("Actual sampled inputs changed")
        x, y = pd.read_parquet(folder / "features.parquet"), pd.read_parquet(folder / "labels.parquet").target
        if len(x) != record["rows"] or list(x) != record["columns"]:
            raise ValueError("Original row sampling or predictor order changed")
        print(json.dumps(dict(phase="refit_native_and_count_member", member=member, rows=len(x))), flush=True)
        native, _, _ = fit_final_catboost(x, y, task="regression", categorical=record["categorical"],
            random_state=record["seed"], model_params=params)
        head = PoissonCountMedian(seed=record["seed"], model_params=raw["model_params"]).fit(
            x, y, categorical=record["categorical"])
        repeated = (PoissonCountMedian(seed=record["seed"], model_params=raw["model_params"]).fit(
            x, y, categorical=record["categorical"]) if index == 0 else None)
        restored = pickle.loads(pickle.dumps(head))
        del x, y
        gc.collect()
        frames, vectors = [], dict(native=[], count=[])
        for forecast in info["forecasts"]:
            path = Path(forecast["path"])
            if digest(path) != forecast["sha256"]:
                raise ValueError("Masked forecast feature cache changed")
            frame = pd.read_parquet(path)
            frames.append(frame[["__event_entity", "__event_cutoff", "__query_order"]].copy())
            vx = prepared(frame, meta["feature_schema"])
            prediction = head.predict(vx)
            np.testing.assert_array_equal(prediction, restored.predict(vx))
            if repeated is not None:
                np.testing.assert_array_equal(prediction, repeated.predict(vx))
            vectors["native"].append(native.predict(vx))
            vectors["count"].append(prediction)
            del frame, vx
            gc.collect()
        routing, aligned = canonical_vectors(frames, vectors, expected_rows=len(expected))
        native_predictions.append(aligned["native"])
        count_predictions.append(aligned["count"])
        del native, head, repeated, restored
        gc.collect()
    keys = masked_query_keys(routing, entity="__event_entity", timestamp="__event_cutoff", target=meta["target"])
    pd.testing.assert_frame_equal(keys, expected, check_exact=True)
    native = np.mean(native_predictions, axis=0)
    count = np.mean(count_predictions, axis=0)
    with np.load(recovery / "predictions.npz") as arrays:
        baseline_keys = artifact_keys(arrays)
        raw_control = align_prediction(arrays["control"], source_keys=baseline_keys, query_keys=keys)
        strongest = align_prediction(arrays["candidate"], source_keys=baseline_keys, query_keys=keys)
    np.testing.assert_allclose(native, raw_control, atol=1e-9, rtol=0)
    projected = projector.predict_values(native, recipe="integer_counts")
    np.testing.assert_array_equal(projected, strongest)
    w = choice["weight"]
    candidate = projector.predict_values((1-w)*native+w*count, recipe="integer_counts")
    verify(files)
    if digest(args.output / "selection.json") != lock:
        raise ValueError("Candidate changed after masked official queries were opened")
    np.savez_compressed(args.output / "predictions.npz", control=strongest, candidate=candidate,
        raw_native=native, count_head=count, **key_arrays(keys))
    dump(args.output / "control-replay-audit.json", dict(native_max_error=float(np.max(np.abs(native-raw_control))),
        strongest_integer_control_exact=True, query_keys_sha256=key_fingerprint(keys),
        independent_count_repeat=True, serialized_count_repeat=True, selection_sha256=lock))
    scores = {name:float(task.evaluate(vector, target_table=None)[meta["metric"]])
        for name, vector in (("control", strongest), ("candidate", candidate))}
    completed = dict(dataset=meta["dataset"], task=meta["task"], metric=meta["metric"],
        selected=choice["selected"], official_test_scores=scores,
        query_keys_sha256=key_fingerprint(keys), selection_sha256=lock,
        deployment_seed=0, official_targets_used_for_selection=False,
        default_changed=False, elapsed_seconds=time.monotonic()-started)
    dump(args.output / "results.json", completed)
    print(json.dumps(dict(phase="validated_count_official_complete", **completed)), flush=True)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root", "validation", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
