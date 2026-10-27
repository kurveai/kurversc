"""Opt-in post-promotion signal selection and official replay for any task.

The incumbent graph and temporal selector are not modified. Candidate choices
are frozen from inner temporal predictions before any outer/test inputs are
opened. No archived winner or score is used to choose a challenger.
"""

import argparse
import copy
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import pickle
import resource
import sys
import time

import numpy as np
import pandas as pd

from kurversc.modeling import prepare_prediction_features
from kurversc.prediction_provenance import (
    align_prediction, artifact_keys, key_arrays, key_fingerprint, masked_query_keys,
)
from kurversc.signal_challengers import (BoundedTextReadout, ChallengerUnavailable,
    challenger_menu, challenger_policy, combine_predictions, mature_support_mask)
from kurversc.signal_challengers import apply_challenger_weight, challenger_fit_key
from kurversc.temporal_policy import (InsufficientPolicySupport, TemporalFitPolicy,
    fit_temporal_policy, select_temporal_policy, _score)
from scripts.schema_exposure_challenger import SchemaExposureQuality, discover_exposure_path


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def prepared(frame, schema):
    return prepare_prediction_features(frame, columns=list(schema["columns"]),
        categorical_columns=set(schema["categorical_columns"]),
        datetime_columns=set(schema["datetime_columns"]))


def routing_headers(frame, schema):
    """Own the nonpayload columns without retaining a wide numeric buffer."""
    payload = set(schema["columns"])
    return frame.loc[:, [name for name in frame if name not in payload]].copy()


def values(model, features, task):
    return np.asarray(model.predict_proba(features)[:, 1] if task == "classification"
                      else model.predict(features)).reshape(-1)


def incumbent_evidence(raw, fold_number, seed, policy):
    name = policy.name.split("__blend_")[0]
    path = raw / f"fold{fold_number}-{name}-seed{seed}.npz"
    with np.load(path) as arrays:
        row = {key: arrays[key].copy() for key in ("truth", "prediction", "groups")}
    if policy.blend_weight < 1:
        with np.load(raw / f"fold{fold_number}-incumbent-seed{seed}.npz") as base:
            np.testing.assert_array_equal(base["truth"], row["truth"])
            np.testing.assert_array_equal(base["groups"], row["groups"])
            row["prediction"] = combine_predictions(base["prediction"], row["prediction"], policy.blend_weight)
    return row


def fit_candidate(spec, base, x, y, times, vx, *, origin, metadata, categorical,
                  root, seed, threads, extra=None, heldout_extra=None):
    task = "classification" if metadata["metric"] == "roc_auc" else "regression"
    policy = challenger_policy(base, spec)
    if spec.kind == "text":
        mask, support = mature_support_mask(times, origin=origin,
            horizon=metadata["label_horizon"], refit_frames=metadata["validation_cutoffs"], policy=base,
            available_at=x.attrs.get("kurversc_label_available_at"))
        text_columns = [x.columns[i] for i in categorical]
        # The text selector consumes categorical fields only. Project those
        # fields before the row mask instead of copying every numeric feature
        # for the complete mature history.
        head = BoundedTextReadout(task=task, random_state=0).fit(x.loc[:, text_columns].loc[mask], y.loc[mask],
            categorical=text_columns, root_prefix=root)
        return head.predict_values(vx), {**head.audit, "support": support}
    if spec.kind == "exposure":
        if extra is None or heldout_extra is None:
            raise ChallengerUnavailable("No eligible exposure path")
        routing = dict(x.attrs)
        x, vx = pd.concat([x, extra], axis=1, copy=False), pd.concat([vx, heldout_extra], axis=1, copy=False)
        x.attrs.update(routing)
    params = dict(thread_count=threads)
    if spec.regularized:
        params.update(depth=5, l2_leaf_reg=30)
    model, audit = fit_temporal_policy(x, y, times, origin=origin,
        label_horizon=metadata["label_horizon"], refit_frames=metadata["validation_cutoffs"],
        policy=policy, task=task, categorical=categorical, random_state=seed,
        row_limit=100_000, model_params=params)
    prediction = values(model, vx, task)
    del model
    gc.collect()
    return prediction, audit


def validate(args):
    selection = json.loads((args.task_root / "selection/results.json").read_text())
    policy = TemporalFitPolicy(**selection["selected_policy"])
    selection_manifest = json.loads((args.task_root / "selection/manifest.json").read_text())
    metadata = selection_manifest["metadata"]
    raw = args.task_root / "raw-audit"
    manifest = dict(dataset=metadata["dataset"], task=metadata["task"],
        checkpoint_sha256=metadata["checkpoint_sha256"], base_policy=asdict(policy),
        menu=[asdict(p) for p in challenger_menu(exposure_blends=getattr(args, "exposure_blends", False))], seeds=[0, 1, 2],
        threads=args.threads, deployment_seed=0,
        implementation_sha256=implementation_digest(),
        scope="Opt-in signals; unchanged incumbent; inner temporal evidence only; no test-based selection")
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Different immutable challenger experiment in output")
    dump(path, manifest)
    if (args.output / "selection.json").exists():
        return json.loads((args.output / "selection.json").read_text()), metadata
    if not (raw / "manifest.json").exists():
        result = dict(selected="incumbent", decision=dict(selected="incumbent",
            reason="insufficient_temporal_evidence", candidates={}), records=[], manifest=manifest)
        dump(args.output / "selection.json", result)
        return result, metadata
    original = json.loads((raw / "manifest.json").read_text())
    if metadata["fold"] != "current" or not metadata["atomic_date_replay"]:
        raise ValueError("Corrected current point-in-time cache required")
    cache = args.task_root / "cache"
    frame = pd.concat([pd.read_parquet(cache / n) for n in ("history.parquet", "validation.parquet")], ignore_index=True)
    schema = metadata["feature_schema"]
    x = prepared(frame, schema)
    frame = routing_headers(frame, schema)
    gc.collect()
    y, times = pd.to_numeric(frame[metadata["target"]]), pd.to_datetime(frame["__event_cutoff"])
    categorical = [i for i, name in enumerate(x) if name in schema["categorical_columns"]]
    from kurversc.task_source import load_task_source as RelBenchDatasetTask
    source = RelBenchDatasetTask(metadata["dataset"], metadata["task"], download=False)
    inner, root = source.inner_split(), source.task.entity_table
    exposure_path = discover_exposure_path(inner.db_state.table_dict, root)
    dump(args.output / "applicability.json", dict(exposure_path=asdict(exposure_path) if exposure_path else None,
        root=root, text_scope="Training-selected parent text columns only"))
    provider = None
    extra = None
    if exposure_path:
        provider = SchemaExposureQuality(inner.db_state.table_dict, root,
            horizon=source.task.timedelta, threads=args.threads, path=exposure_path,
            temp_directory=args.output / "spill")
        extra = provider.transform(frame, phase_boundary=inner.cutoff)
        dump(args.output / "exposure-training-audit.json", provider.last_audit)
    task = "classification" if metadata["metric"] == "roc_auc" else "regression"
    evidence, records = [], []
    try:
        for number, fold in enumerate(original["folds"]):
            folder = args.task_root / "snapshots" / f"fold{number}"
            snapshot_manifest = json.loads((folder / "manifest.json").read_text())
            if snapshot_manifest["origin"] != fold["origin"] or snapshot_manifest["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
                raise ValueError("Retrospective snapshot identity mismatch")
            origin = snapshot_manifest["snapshot_cutoff"]
            if pd.Timestamp(origin) > pd.Timestamp(fold["origin"]) or pd.Timestamp(snapshot_manifest["latest_observed_label_complete_at"]) > pd.Timestamp(origin):
                raise ValueError("Future state or immature targets reached a snapshot")
            heldout = pd.read_parquet(folder / "validation.parquet")
            positions = heldout["__position"].to_numpy(dtype=int, copy=True)
            if len(set(positions)) != len(positions):
                raise ValueError("Duplicated validation positions")
            np.testing.assert_array_equal(heldout[metadata["target"]], y.iloc[positions])
            vx = prepared(heldout, schema)
            heldout = routing_headers(heldout, schema)
            gc.collect()
            heldout_extra = provider.transform(heldout, phase_boundary=origin) if provider else None
            if provider:
                dump(args.output / f"fold{number}-exposure-audit.json", provider.last_audit)
            for seed in (0, 1, 2):
                control = incumbent_evidence(raw, number, seed, policy)
                np.testing.assert_array_equal(control["truth"], y.iloc[positions])
                np.testing.assert_array_equal(control["groups"], heldout["__event_entity"])
                evidence.append(dict(origin=fold["origin"], seed=seed, policy="incumbent", **control))
                records.append(dict(origin=fold["origin"], seed=seed, policy="incumbent",
                    score=_score(control["truth"], control["prediction"], task)))
                raw_heads = {}
                for spec in challenger_menu(exposure_blends=getattr(args, "exposure_blends", False)):
                    basename = f"fold{number}-{spec.name}-seed{seed}"
                    arrays_path, record_path = args.output / f"{basename}.npz", args.output / f"{basename}.json"
                    try:
                        if arrays_path.exists() and record_path.exists():
                            with np.load(arrays_path) as saved:
                                prediction = saved["prediction"].copy()
                                np.testing.assert_array_equal(saved["truth"], control["truth"])
                                np.testing.assert_array_equal(saved["groups"], control["groups"])
                            record = json.loads(record_path.read_text())
                            if spec.weight == 1:
                                raw_heads[challenger_fit_key(spec)] = prediction, record["audit"]
                        else:
                            fit_key = challenger_fit_key(spec)
                            if fit_key not in raw_heads:
                                raw_heads[fit_key] = fit_candidate(spec, policy, x, y, times, vx,
                                    origin=origin, metadata=metadata, categorical=categorical,
                                    root=root, seed=seed, threads=args.threads,
                                    extra=extra, heldout_extra=heldout_extra)
                            prediction, audit = raw_heads[fit_key]
                            prediction = apply_challenger_weight(spec, control["prediction"], prediction)
                            record = dict(origin=fold["origin"], seed=seed, policy=spec.name,
                                score=_score(control["truth"], prediction, task), audit=audit)
                            np.savez_compressed(arrays_path, truth=control["truth"], prediction=prediction, groups=control["groups"])
                            dump(record_path, record)
                        evidence.append(dict(origin=fold["origin"], seed=seed, policy=spec.name,
                            truth=control["truth"], prediction=prediction, groups=control["groups"]))
                    except (ChallengerUnavailable, InsufficientPolicySupport) as exc:
                        record = dict(origin=fold["origin"], seed=seed, policy=spec.name, unavailable=str(exc))
                        dump(record_path, record)
                    records.append(record)
                    dump(args.output / "progress.json", dict(records=records))
                    print(json.dumps({k:v for k,v in record.items() if k != "audit"}), flush=True)
                    gc.collect()
    finally:
        if provider:
            provider.close()
    decision = select_temporal_policy(evidence, task=task, deployment_seed=0)
    result = dict(selected=decision["selected"], decision=decision, records=records, manifest=manifest)
    dump(args.output / "selection.json", result)
    return result, metadata


def materialize_outer(args, metadata, spec, base_policy):
    """Frozen graph materialization, with one-tree drivers whose scores are discarded."""
    import kurversc
    from kurversc.plan_dates import atomic_execution_plan
    from kurversc.training_support import available_root_training, select_training_support
    from kurversc.staged_adapter import resolve_cache_config
    from kurversc.task_source import load_task_source as RelBenchDatasetTask
    from kurversc.staged_adapter import KURVERSC_DEFAULTS, _SelectedGraph, _runtime_config
    from scripts.benchmark_avito_quality_official import capture_prediction
    from scripts.benchmark_frozen_event_summaries import _capture
    from scripts.run_kurversc_selected_relarena import AuditWinnerSystem

    checkpoint = Path(metadata["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata["checkpoint_sha256"]:
        raise ValueError("Graph checkpoint changed after validation")
    trial = pickle.loads(checkpoint.read_bytes())["trial"]
    trial.execution_plan = atomic_execution_plan(trial.execution_plan)
    source = RelBenchDatasetTask(metadata["dataset"], metadata["task"], download=False)
    outer, task = source.outer_split(), source.task
    KURVERSC_DEFAULTS.update(full_training_frames=1, full_validation_frames=None,
        base_distribution_features=True, temporal_estimator_mode="frame_ensemble",
        duckdb_memory_limit=None, duckdb_max_temp_directory_size=None,
        model_params={"iterations":1, "thread_count":1})
    system = AuditWinnerSystem(cache=resolve_cache_config(None, on_miss="raise"), run_identity=source.run_identity())
    system.peer_context_mode, system.peer_max_history_rows = "both", 50_000
    _, problem = system._fit_arm(task, outer.db_state, outer.train_table, outer.val_table,
        seed=0, phase="outer", selection=_SelectedGraph(config=trial.config,
            execution_plan=copy.deepcopy(trial.execution_plan)))
    root = outer.db_state.table_dict[task.entity_table]
    policy = challenger_policy(base_policy, spec)
    available = available_root_training(outer.train_table.df, root.df, label_key=task.entity_col,
        root_key=root.pkey_col, timestamp=task.time_col, root_timestamp=root.time_col)
    support = select_training_support(available, timestamp=task.time_col,
        evaluation_cutoff=pd.to_datetime(outer.val_table.df[task.time_col]).min(),
        label_horizon=task.timedelta, minimum_rows=policy.minimum_training_rows,
        minimum_frames=policy.training_frames, max_frames=policy.max_training_frames)
    if not support.complete or len(support.cutoffs) < policy.training_frames:
        raise InsufficientPolicySupport("Final training support is incomplete")
    labels = problem.label_node
    history = outer.train_table.df.loc[pd.to_datetime(outer.train_table.df[task.time_col]).isin(support.cutoffs)].copy()
    history[labels.split] = labels.train_value
    validation = labels.source.loc[labels.source[labels.split] != labels.train_value].copy()
    observed = pd.concat([history, validation], ignore_index=True)
    first_test = pd.to_datetime(outer.eval_table.df[task.time_col]).min()
    if (pd.to_datetime(observed[task.time_col]) + task.timedelta > first_test).any():
        raise ValueError("Immature final labels")
    problem = replace(problem, label_node=replace(labels, source=observed))
    options = _runtime_config()
    options.update(full_training_frames=len(support.cutoffs), temporal_estimator_mode="pooled",
        model_params={"iterations":1, "thread_count":1})
    schema = metadata["feature_schema"]
    with _capture(schema["columns"], task.target_col, task.entity_col, task.entity_table) as (frames, _):
        fitted = kurversc.fit(**problem.fit_kwargs(), **options, random_state=0,
            preselected_config=trial.config, preselected_execution_plan=copy.deepcopy(trial.execution_plan))
    train = pd.concat(frames, ignore_index=True)
    with capture_prediction(schema, task.entity_col, task.entity_table) as chunks:
        system._predict(task, outer.eval_table, result=fitted, problem=problem)
    forecast = pd.concat(chunks, ignore_index=True)
    if sorted(forecast["__order"].tolist()) != list(range(len(outer.eval_table.df))):
        raise ValueError("Masked test query order changed")
    forecast = forecast.sort_values("__order", kind="stable").reset_index(drop=True)
    del fitted
    gc.collect()
    return source, outer, train, forecast


def official(args, selected, metadata):
    # All choices are immutable before this function exposes the outer split.
    selected_path = args.output / "selection.json"
    choice_digest = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    baseline_root = args.task_root / "official"
    baseline = json.loads((baseline_root / "results.json").read_text())
    with np.load(baseline_root / "predictions.npz") as saved:
        control = saved["candidate"].copy()  # The unchanged full current system.
        prediction_keys = artifact_keys(saved)
    if baseline.get("query_keys_sha256") != key_fingerprint(prediction_keys):
        raise ValueError("Official baseline key provenance changed")
    candidate, application = control.copy(), "incumbent_retained"
    if selected["selected"] != "incumbent":
        # The immutable validated menu, not current defaults, owns inference.
        from kurversc.signal_challengers import SignalChallenger
        specs = [SignalChallenger(**row) for row in selected["manifest"]["menu"]]
        spec = next(p for p in specs if p.name == selected["selected"])
        if selected["decision"]["candidates"][spec.name]["reason"] != "eligible":
            raise ValueError("Unadmitted signal challenger")
        base = TemporalFitPolicy(**selected["manifest"]["base_policy"])
        source, outer, train, forecast = materialize_outer(args, metadata, spec, base)
        query_keys = masked_query_keys(forecast, entity="__event_entity",
                                      timestamp="__event_cutoff", target=metadata["target"])
        control = align_prediction(control, source_keys=prediction_keys, query_keys=query_keys)
        prediction_keys = query_keys
        schema = metadata["feature_schema"]
        x, vx = prepared(train, schema), prepared(forecast, schema)
        train, forecast = routing_headers(train, schema), routing_headers(forecast, schema)
        gc.collect()
        y, times = pd.to_numeric(train[metadata["target"]]), pd.to_datetime(train["__event_cutoff"])
        categorical = [i for i, name in enumerate(x) if name in schema["categorical_columns"]]
        extra, heldout_extra, provider = None, None, None
        try:
            if spec.kind == "exposure":
                path = json.loads((args.output / "applicability.json").read_text())["exposure_path"]
                from scripts.schema_exposure_challenger import ExposurePath
                provider = SchemaExposureQuality(outer.db_state.table_dict, source.task.entity_table,
                    horizon=source.task.timedelta, threads=8, path=ExposurePath(**path),
                    temp_directory=args.output / "official-spill")
                extra = provider.transform(train, phase_boundary=outer.cutoff)
                dump(args.output / "official-exposure-training-audit.json", provider.last_audit)
                heldout_extra = provider.transform(forecast, phase_boundary=outer.cutoff)
                dump(args.output / "official-exposure-forecast-audit.json", provider.last_audit)
            candidate, audit = fit_candidate(spec, base, x, y, times, vx,
                origin=pd.to_datetime(outer.eval_table.df[source.task.time_col]).min(),
                metadata=metadata, categorical=categorical, root=source.task.entity_table,
                seed=0, threads=8, extra=extra, heldout_extra=heldout_extra)
            candidate = apply_challenger_weight(spec, control, candidate)
            dump(args.output / "official-fit-audit.json", audit)
        finally:
            if provider:
                provider.close()
        application = spec.name
    if hashlib.sha256(selected_path.read_bytes()).hexdigest() != choice_digest:
        raise ValueError("Selection changed during official replay")
    if control.shape != candidate.shape or not np.isfinite(candidate).all():
        raise ValueError("Invalid official prediction shape or values")
    prediction_keys.to_parquet(args.output / "masked-query-keys.parquet", index=False)
    np.savez_compressed(args.output / "predictions.npz", control=control, candidate=candidate,
                        **key_arrays(prediction_keys))
    # Scores cannot change the chosen model. Every scored vector is saved first.
    if application == "incumbent_retained":
        score = float(baseline["official_test_scores"]["candidate"])
    else:
        score = float(source.task.evaluate(candidate, target_table=None)[metadata["metric"]])
    result = dict(dataset=metadata["dataset"], task=metadata["task"], metric=metadata["metric"],
        arm=application, official_test_scores=dict(control=baseline["official_test_scores"]["candidate"], candidate=score),
        baseline_temporal_policy=baseline["arm"], signal_selection=selected["decision"],
        selection_sha256=choice_digest, optional_challengers=True, deployment_seed=0,
        query_keys_sha256=key_fingerprint(prediction_keys),
        scope="Validation-selected optional signal challenger; unchanged incumbent fallback; no test-based choice")
    return result


def implementation_digest():
    root = Path(__file__).resolve().parents[1]
    names = ["scripts/benchmark_signal_challengers.py", "scripts/schema_exposure_challenger.py",
        "scripts/avito_exposure_signals.py", "kurversc/signal_challengers.py", "kurversc/temporal_policy.py",
        "kurversc/prediction_provenance.py"]
    return hashlib.sha256(b"".join((root / p).read_bytes() for p in names)).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--exposure-blends", action="store_true",
        help="Test bounded partial exposure readouts, reusing the existing same-support fits; opt-in")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive thread budget required")
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    selected, metadata = validate(args)
    result = official(args, selected, metadata)
    result.update(elapsed_seconds=time.monotonic() - started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3 if sys.platform == "darwin" else 1024**2))
    dump(args.output / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
