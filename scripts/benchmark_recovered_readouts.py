"""Complete, schema-generic post-promotion readouts missing from the suite recipe.

Domain projection, object history, recurrence support, joint semantic correction,
and zero-aware count readout are optional, validation-gated procedures. No task
winner lookup or archived graph/prediction supplies a result. Every official
vector carries its own entity/time keys and choices are frozen before scoring.
"""

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import sys
import time

# A direct execution overlay must win over the legacy working-directory
# namespace without replacing the pinned KurveRSC/GraphReduce packages.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from kurversc.task_source import CONTEXT_ENV, phase_source_root
import os
import pyarrow.parquet as pq

from kurversc.prediction_provenance import align_prediction, artifact_keys, key_arrays, key_fingerprint, masked_query_keys
from kurversc.signal_challengers import ChallengerUnavailable, SignalChallenger
from kurversc.temporal_policy import InsufficientPolicySupport, TemporalFitPolicy, fit_temporal_policy, policy_support, _score
from scripts.activity_timeline_challenger import table_metadata
from scripts.benchmark_large_gap_challengers import fit_prediction
from scripts.benchmark_phase_frozen_breadth_v2 import ObjectBreadth, task_root_table
from scripts.benchmark_signal_challengers import incumbent_evidence, materialize_outer, prepared, values
from scripts.discrete_target_projection import TargetDomainProjection
from scripts.engagement_local_comparison import freeze
from scripts.frozen_auc_bootstrap import frozen_auc_select_temporal_policy
from scripts.run_gap_followup_queue import digest, dump
from scripts.schema_semantic_recurrence import MAX_SEMANTIC_ROWS, MAX_SEMANTIC_DOCUMENTS, fit_seed, prepare_phase
from scripts.benchmark_training_row_budget import file_sha256
from scripts.select_activity_readout import cache_locations


RECIPES = ("nonnegative", "integer_counts", "object_full", "object_half", "recurrence3",
           "joint_semantic32", "zero_aware_0.125", "zero_aware_0.5", "zero_aware_1")
ALLOWANCE = dict(domain=90, objects=46, recurrence=146, semantic=146, zero=200)
MAX_OPTIONAL_FEATURE_CELLS = 500_000_000


def optional_zero_prediction(*args, **kwargs):
    """A globally supported target can lack support in one sampled cohort.

    This invalidates only the optional zero-aware candidate, not the already
    admitted comparator or the rest of the readout evaluation. Other errors
    remain fatal; do not turn an arbitrary learner bug into a silent fallback.
    """
    try:
        return fit_prediction(*args, **kwargs, zero=True)
    except ValueError as error:
        if str(error) != "Nonnegative numeric targets with >=20 positive and zero observations required":
            raise
        return None, dict(unavailable=str(error), support_scope="actual sampled estimator cohort")


def baseline_evidence(root, fold, seed, policy):
    selected = json.loads((root / "signal-challengers/selection.json").read_text())["selected"]
    if selected == "incumbent":
        return incumbent_evidence(root / "raw-audit", fold, seed, policy)
    path = root / "signal-challengers" / f"fold{fold}-{selected}-seed{seed}.npz"
    with np.load(path) as a:
        return {k: a[k].copy() for k in ("truth", "groups", "prediction")}


def matched_gate(records, *, metric, recipes, allowance):
    origins = sorted({r["origin"] for r in records})
    expected = {(o, s) for o in origins for s in (0, 1, 2)}
    observed = [(r["origin"], r["seed"]) for r in records]
    if len(observed) != len(set(observed)):
        raise ValueError("Duplicate readout validation cases")
    if len(origins) < 2 or set(observed) != expected:
        return dict(selected="incumbent", reason="incomplete_available_origin_three_seed_grid", candidates={})
    task = "classification" if metric == "roc_auc" else "regression"
    survivors = [n for n in recipes if all(n in r["vectors"] and
        ((1 if task == "classification" else -1) *
         (_score(r["truth"], r["vectors"][n], task) - _score(r["truth"], r["control"], task))) > 0 for r in records)]
    if not survivors:
        return dict(selected="incumbent", reason="failed_complete_positive_gate", candidates={})
    evidence = [dict(origin=r["origin"], seed=r["seed"], policy=n, truth=r["truth"], groups=r["groups"],
                     prediction=r["control"] if n == "incumbent" else r["vectors"][n])
                for r in records for n in ["incumbent", *survivors]]
    return frozen_auc_select_temporal_policy(evidence, task=task, minimum_origins=len(origins),
        minimum_seeds=3, deployment_seed=0, bootstrap_repeats=1600, alpha=.05 * len(survivors) / allowance)


def phase_frames(root, raw, policy, fold, *, frames=None,
                 maximum_feature_cells=MAX_OPTIONAL_FEATURE_CELLS, maximum_rows=None):
    cache, snapshots = cache_locations(root)
    meta = raw["metadata"]
    snapshot = json.loads((snapshots / f"fold{fold}/manifest.json").read_text())
    origin = snapshot["snapshot_cutoff"]
    if pd.Timestamp(snapshot["latest_observed_label_complete_at"]) > pd.Timestamp(origin):
        raise ValueError("Immature targets in a frozen source phase")
    availability = meta.get("label_available_at")
    headers = pd.read_parquet(cache / "history.parquet", columns=["__event_cutoff",*([availability] if availability else [])])
    support_policy = policy if frames is None else TemporalFitPolicy("three_initial", training_frames=frames)
    support = policy_support(headers, timestamp="__event_cutoff", origin=origin,
        label_horizon=meta["label_horizon"], refit_frames=meta["validation_cutoffs"], policy=support_policy)
    # Keep the on-disk timestamp representation (some old caches use strings).
    visible = pd.to_datetime(headers.__event_cutoff).isin(pd.to_datetime(support["cutoffs"]))
    count = int(visible.sum())
    if maximum_rows is not None and count > maximum_rows:
        raise ChallengerUnavailable(f"Optional mature support has {count} rows; capacity is {maximum_rows}")
    held_path = snapshots / f"fold{fold}/validation.parquet"
    cells = (count + pq.ParquetFile(held_path).metadata.num_rows) * len(meta["feature_schema"]["columns"])
    fit_available = cells <= maximum_feature_cells
    if frames is not None and not fit_available:
        raise ChallengerUnavailable(f"Optional learner payload has {cells} cells; capacity is {maximum_feature_cells}")
    columns = [*(meta["feature_schema"]["columns"] if fit_available else []),
               meta["target"], "__event_entity", "__event_cutoff",*([availability] if availability else [])]
    filter_cutoffs = headers.loc[visible, "__event_cutoff"].drop_duplicates().tolist()
    train = pd.read_parquet(cache / "history.parquet", columns=columns,
                           filters=[("__event_cutoff", "in", filter_cutoffs)])
    if availability:
        dates = pd.to_datetime(train.__event_cutoff)
        boundary = support["refit"][0] if support["refit"] else pd.Timestamp(origin)
        ready = pd.to_datetime(train[availability])
        train = train.loc[ready.le(pd.Timestamp(origin)) & (~dates.isin(support["initial"]) | ready.le(boundary))]
    held = pd.read_parquet(held_path, columns=columns)
    if (pd.to_datetime(train.__event_cutoff) + pd.Timedelta(meta["label_horizon"]) > pd.Timestamp(origin)).any():
        raise ValueError("Immature readout training observations")
    return train.reset_index(drop=True), held.reset_index(drop=True), dict(origin=snapshot["origin"], fit_origin=origin,
        optional_fit_available=fit_available, optional_feature_cells=cells)


def object_payload(train, held, *, meta, source_root, origin, output):
    source_meta = table_metadata(source_root / "tasks" / meta["task"] / "train.parquet")
    if len(source_meta["fkey_col_to_pkey_table"] or {}) != 1:
        raise ChallengerUnavailable("Single predictive root required for object history")
    root = task_root_table(meta, source_meta)
    try:
        provider = ObjectBreadth(source_root / "db", root, horizon=meta["label_horizon"],
            training_cutoff=pd.to_datetime(train.__event_cutoff).min(), temporary=output / "object-spill", threads=2)
    except ValueError as error:
        if "No training-visible high-cardinality object foreign key" not in str(error):
            raise
        raise ChallengerUnavailable(str(error)) from error
    try:
        return [provider.transform(f, phase_boundary=origin) for f in (train, held)]
    finally:
        provider.close()


def projected_vectors(records, projection):
    return [{**r, "control":r["vectors"].get(projection, r["control"]) if projection != "incumbent" else r["control"],
             "vectors":{k:r["projector"].predict_values(v, recipe=projection) if projection != "incumbent" else v
                        for k,v in r["vectors"].items() if k.startswith("zero_aware_")}}
            for r in records]


def validate(args):
    raw = json.loads((args.task_root / "raw-audit/manifest.json").read_text())
    meta = raw["metadata"]
    if meta["fold"] != "current" or not meta["atomic_date_replay"]:
        raise ValueError("Corrected phase-frozen graph required")
    policy = TemporalFitPolicy(**json.loads((args.task_root / "selection/results.json").read_text())["selected_policy"])
    sources = {str(p):digest(p) for p in [__file__, "scripts/schema_semantic_recurrence.py",
        "scripts/zero_aware_quantile_readout.py", "scripts/discrete_target_projection.py",
        "scripts/phase_frozen_object_breadth.py", "scripts/activity_timeline_challenger.py",
        "scripts/diagnose_event_recurrence.py", "scripts/concurrent_portable_bagged_offset.py",
        "scripts/frozen_parent_semantic_embedding.py", "scripts/crossfit_semantic_propensity.py",
        "kurversc/prediction_provenance.py", "kurversc/temporal_policy.py", "kurversc/modeling.py"]}
    # Include the actually imported provider when execution fixes are deployed
    # as a separately versioned overlay rather than editing a frozen checkout.
    provider_source = Path(importlib.import_module(ObjectBreadth.__module__).__file__).resolve()
    sources[str(provider_source)] = digest(provider_source)
    cache, snapshots = cache_locations(args.task_root)
    input_files = {str(p): file_sha256(p) for p in [cache / "history.parquet", cache / "validation.parquet",
        *sorted(snapshots.glob("fold*/manifest.json")), *sorted(snapshots.glob("fold*/validation.parquet"))]}
    identity = dict(metadata=meta, policy=asdict(policy), recipes=list(RECIPES), allowance=ALLOWANCE,
        sources=sources, baseline_selection_sha256=digest(args.task_root / "signal-challengers/selection.json"),
        raw_manifest_sha256=digest(args.task_root / "raw-audit/manifest.json"), workers=args.workers,
        input_files=input_files,
        capacity=dict(optional_feature_cells=MAX_OPTIONAL_FEATURE_CELLS,
                      semantic_rows=MAX_SEMANTIC_ROWS, semantic_documents=MAX_SEMANTIC_DOCUMENTS),
        deployment_seed=0, official_targets_used_for_selection=False)
    freeze(args.output / "manifest.json", identity)
    result_path = args.output / "selection.json"
    if result_path.exists():
        return json.loads(result_path.read_text()), meta
    task = "classification" if meta["metric"] == "roc_auc" else "regression"
    rows, applicability = [], []
    folds = list(range(len(raw["folds"])))
    if not 1 <= len(folds) <= 3:
        raise ValueError("One to three available frozen validation origins required")
    for fold in reversed(folds):
        folder = args.output / f"fold{fold}"
        folder.mkdir(exist_ok=True)
        train, held, phase = phase_frames(args.task_root, raw, policy, fold,
            maximum_feature_cells=MAX_OPTIONAL_FEATURE_CELLS if len(folds) >= 2 else 0)
        fit_available = phase["optional_fit_available"]
        x, vx = (prepared(train, meta["feature_schema"]), prepared(held, meta["feature_schema"])) if fit_available else (None, None)
        y, times = train[meta["target"]], pd.to_datetime(train.__event_cutoff)
        projector = TargetDomainProjection().fit(y, times, origin=phase["fit_origin"], horizon=meta["label_horizon"])
        obj = semantic = None
        if fit_available:
            try:
                obj = object_payload(train, held, meta=meta, source_root=args.source_root,
                                     origin=phase["fit_origin"], output=folder)
            except ChallengerUnavailable as error:
                applicability.append(dict(fold=fold, family="objects", reason=str(error)))
        else:
            applicability.append(dict(fold=fold, family="additional_payload_learners",
                reason="optional_feature_cell_capacity", cells=phase["optional_feature_cells"],
                incumbent_and_domain_projection_preserved=True))
        if task == "classification" and len(folds) >= 2:
            try:
                tx, tv, sp = phase_frames(args.task_root, raw, policy, fold, frames=3, maximum_rows=MAX_SEMANTIC_ROWS)
                phase_path, phase_manifest = folder / "semantic-phase.pkl", folder / "semantic-cache.json"
                if phase_path.exists():
                    cached = json.loads(phase_manifest.read_text())
                    if cached["sha256"] != file_sha256(phase_path) or cached["identity_sha256"] != digest(args.output / "manifest.json"):
                        raise ValueError("Semantic preparation cache changed")
                    with phase_path.open("rb") as handle:
                        semantic = pickle.load(handle)
                else:
                    print(json.dumps(dict(phase="schema_semantic_preparation", fold=fold, rows=len(tx))), flush=True)
                    semantic = prepare_phase(tx, tv, metadata=meta, source_root=args.source_root,
                        origin=sp["fit_origin"], output=folder / "semantic", workers=args.workers)
                    with phase_path.open("wb") as handle:
                        pickle.dump(semantic, handle)
                    dump(phase_manifest, dict(sha256=file_sha256(phase_path), identity_sha256=digest(args.output / "manifest.json")))
                np.testing.assert_array_equal(tv[meta["target"]], held[meta["target"]])
                np.testing.assert_array_equal(tv.__event_entity, held.__event_entity)
                del tx, tv
            except (ChallengerUnavailable, InsufficientPolicySupport) as error:
                applicability.append(dict(fold=fold, family="semantic_recurrence", reason=str(error)))
        zero_possible = (fit_available and task == "regression" and (np.asarray(y) >= 0).all() and
            (np.asarray(y) == 0).sum() >= 20 and (np.asarray(y) > 0).sum() >= 20 and
            np.unique(np.asarray(y)[np.asarray(y) > 0]).size >= 2)
        for seed in (0, 1, 2):
            path = folder / f"seed{seed}.npz"
            report_path = folder / f"seed{seed}.json"
            baseline = baseline_evidence(args.task_root, fold, seed, policy)
            np.testing.assert_array_equal(baseline["truth"], held[meta["target"]])
            np.testing.assert_array_equal(baseline["groups"], held.__event_entity)
            audit = {}
            if path.exists():
                with np.load(path) as a:
                    for key in ("truth", "groups"):
                        np.testing.assert_array_equal(a[key], baseline[key])
                    np.testing.assert_array_equal(a["control"], baseline["prediction"])
                    vectors = {k:a[k].copy() for k in a.files if k not in {"truth", "groups", "control", "forecast_cutoffs"}}
                previous_report = json.loads(report_path.read_text())
                audit = previous_report["audit"]
                if previous_report.get("manifest_sha256") != digest(args.output / "manifest.json"):
                    audit = {**audit, "reused_checkpoint":dict(
                        source_manifest_sha256=previous_report.get("manifest_sha256"),
                        predictions_sha256=digest(path))}
            else:
                vectors = {}
                if seed == 0 and fit_available:
                    native, _ = fit_prediction(x, y, times, vx, raw=raw, meta=meta, policy=policy, phase=phase, seed=seed)
                    reference = incumbent_evidence(args.task_root / "raw-audit", fold, seed, policy)
                    np.testing.assert_allclose(native, reference["prediction"], rtol=0, atol=1e-9)
                if task == "regression":
                    for n, available in (("nonnegative", projector.nonnegative), ("integer_counts", projector.integer_counts)):
                        if available:
                            vectors[n] = projector.predict_values(baseline["prediction"], recipe=n)
                if obj is not None:
                    xx, vxx = [pd.concat([a,b], axis=1) for a,b in ((x,obj[0]), (vx,obj[1]))]
                    pred, audit["objects"] = fit_prediction(xx, y, times, vxx, raw=raw, meta=meta, policy=policy, phase=phase, seed=seed)
                    vectors["object_full"], vectors["object_half"] = pred, .5 * (baseline["prediction"] + pred)
                    if fold == max(folds) and seed == 0:
                        repeated, _ = fit_prediction(xx, y, times, vxx, raw=raw, meta=meta, policy=policy, phase=phase, seed=seed)
                        np.testing.assert_allclose(pred, repeated, rtol=0, atol=1e-9)
                        audit["objects_independent_replay_max_error"] = float(np.max(np.abs(pred-repeated)))
                    del xx, vxx
                if zero_possible:
                    zero, audit["zero"] = optional_zero_prediction(x, y, times, vx, raw=raw, meta=meta, policy=policy, phase=phase, seed=seed)
                    if zero is not None and fold == max(folds) and seed == 0:
                        repeated, _ = optional_zero_prediction(x, y, times, vx, raw=raw, meta=meta, policy=policy, phase=phase, seed=seed)
                        np.testing.assert_allclose(zero, repeated, rtol=0, atol=1e-9)
                        audit["zero_independent_replay_max_error"] = float(np.max(np.abs(zero-repeated)))
                    if zero is not None:
                        for w in (.125,.5,1.):
                            vectors[f"zero_aware_{w:g}"] = (1-w) * baseline["prediction"] + w * zero
                if semantic is not None:
                    print(json.dumps(dict(phase="schema_semantic_fit", fold=fold, seed=seed)), flush=True)
                    head, corrected, audit["semantic"] = fit_seed(semantic, seed, repeat=fold == max(folds) and seed == 0)
                    vectors["recurrence3"] = .5 * (baseline["prediction"] + head)
                    vectors["joint_semantic32"] = .5 * (baseline["prediction"] + corrected)
                np.savez_compressed(path, truth=baseline["truth"], groups=baseline["groups"], control=baseline["prediction"],
                    forecast_cutoffs=pd.to_datetime(held.__event_cutoff).to_numpy(), **vectors)
            scores = {n:_score(baseline["truth"], p, task) for n,p in {"control":baseline["prediction"], **vectors}.items()}
            dump(report_path, dict(fold=fold, seed=seed, origin=phase["origin"], scores=scores, audit=audit,
                                  vectors_sha256=digest(path), manifest_sha256=digest(args.output / "manifest.json")))
            rows.append(dict(fold=fold, seed=seed, origin=phase["origin"], truth=baseline["truth"], groups=baseline["groups"],
                             control=baseline["prediction"], vectors=vectors, projector=projector))
            print(json.dumps(dict(phase="recovered_readout_case", fold=fold, seed=seed, scores=scores)), flush=True)
            dump(args.output / "progress.json", dict(cases=len(rows), required_cases=3*len(folds), latest=scores))
        del train, held, x, vx, obj, semantic
        gc.collect()
    decisions = {}
    # Each family challenges the strongest ALREADY ADMITTED comparator. This
    # prevents a gain against a weaker raw head from overwriting a better one.
    chosen = "incumbent"
    decisions["domain"] = matched_gate(rows, metric=meta["metric"], recipes=["nonnegative", "integer_counts"], allowance=ALLOWANCE["domain"])
    project = decisions["domain"]["selected"]
    if project != "incumbent":
        chosen = project
    active = projected_vectors(rows, project) if task == "regression" else rows
    decisions["zero"] = matched_gate(active, metric=meta["metric"], recipes=[n for n in RECIPES if n.startswith("zero_aware_")], allowance=ALLOWANCE["zero"])
    if decisions["zero"]["selected"] != "incumbent":
        chosen = decisions["zero"]["selected"]
    # Keep all optional candidates available, but compare with current selected
    # predictions rather than selecting a weaker head from another family.
    for family, names in (("recurrence", ["recurrence3"]), ("objects", ["object_full", "object_half"]), ("semantic", ["joint_semantic32"])):
        comparison = []
        for r in rows:
            control = r["control"] if chosen == "incumbent" else r["vectors"].get(chosen)
            if chosen.startswith("zero_aware_") and project != "incumbent":
                control = r["projector"].predict_values(control, recipe=project)
            if control is None:
                raise ValueError("Incomplete strongest-control prediction grid")
            candidates = {n:(r["projector"].predict_values(r["vectors"][n], recipe=project)
                            if project != "incumbent" else r["vectors"][n]) for n in names if n in r["vectors"]}
            comparison.append({**r, "control":control, "vectors":candidates})
        decisions[family] = matched_gate(comparison, metric=meta["metric"], recipes=names, allowance=ALLOWANCE[family])
        if decisions[family]["selected"] != "incumbent":
            chosen = decisions[family]["selected"]
    if any(digest(p) != sha for p,sha in sources.items()):
        raise ValueError("Readout implementation changed during selection")
    result = dict(selected=chosen, projection=project, decisions=decisions, complete_grid=len(rows) == 3*len(folds),
        origins=len(folds), seeds=[0,1,2],
        applicability=applicability, source_sha256=sources, metadata=meta, policy=asdict(policy),
        baseline_signal=json.loads((args.task_root / "signal-challengers/selection.json").read_text())["selected"],
        official_targets_used_for_selection=False, deployment_seed=0)
    dump(result_path, result)
    return result, meta


def official(args, selected, meta):
    source_root = (phase_source_root(meta["dataset"],phase="outer")
        if os.environ.get(CONTEXT_ENV) else args.source_root)
    path = args.output / "results.json"
    if path.exists():
        return json.loads(path.read_text())
    identity = digest(args.output / "selection.json")
    baseline_root = args.task_root / "signal-challengers"
    baseline = json.loads((baseline_root / "results.json").read_text())
    with np.load(baseline_root / "predictions.npz") as a:
        keys, control = artifact_keys(a), a["candidate"].copy()
    if baseline.get("query_keys_sha256") != key_fingerprint(keys):
        raise ValueError("Cannot replay an unkeyed or changed official baseline")
    candidate, choice = control.copy(), selected["selected"]
    source = None
    if choice != "incumbent":
        policy = TemporalFitPolicy(**selected["policy"])
        semantic_choice = choice in {"recurrence3", "joint_semantic32"}
        spec = (SignalChallenger("three_initial", "support", training_frames=3) if semantic_choice
                else SignalChallenger("recovered_readout", "support"))
        source, outer, train, forecast = materialize_outer(args, meta, spec, policy)
        new_keys = masked_query_keys(forecast, entity="__event_entity", timestamp="__event_cutoff", target=meta["target"])
        control = align_prediction(control, source_keys=keys, query_keys=new_keys)
        keys = new_keys
        task = "classification" if meta["metric"] == "roc_auc" else "regression"
        x, vx = prepared(train, meta["feature_schema"]), prepared(forecast, meta["feature_schema"])
        y, times = train[meta["target"]], pd.to_datetime(train.__event_cutoff)
        projection = TargetDomainProjection().fit(y, times, origin=outer.cutoff, horizon=meta["label_horizon"])
        if choice in {"nonnegative", "integer_counts"}:
            candidate = projection.predict_values(control, recipe=choice)
        elif choice.startswith("zero_aware_"):
            raw = json.loads((args.task_root / "raw-audit/manifest.json").read_text())
            head, _ = fit_prediction(x, y, times, vx, raw=raw, meta=meta, policy=policy,
                                    phase=dict(fit_origin=str(outer.cutoff)), seed=0, zero=True)
            weight = float(choice.removeprefix("zero_aware_"))
            candidate = (1-weight)*control + weight*head
        elif choice.startswith("object_"):
            blocks = object_payload(train, forecast, meta=meta, source_root=source_root, origin=outer.cutoff, output=args.output / "official")
            raw = json.loads((args.task_root / "raw-audit/manifest.json").read_text())
            xx, vxx = [pd.concat([a,b], axis=1) for a,b in ((x,blocks[0]), (vx,blocks[1]))]
            head, _ = fit_prediction(xx, y, times, vxx, raw=raw, meta=meta, policy=policy,
                                    phase=dict(fit_origin=str(outer.cutoff)), seed=0)
            candidate = head if choice == "object_full" else .5 * (control + head)
        else:
            phase = prepare_phase(train, forecast, metadata=meta, source_root=source_root, origin=outer.cutoff,
                                  output=args.output / "official-semantic", workers=args.workers)
            head, corrected, _ = fit_seed(phase, 0, repeat=True)
            selected_head = head if choice == "recurrence3" else corrected
            candidate = .5 * (control + selected_head)
        if selected["projection"] != "incumbent" and choice not in {"nonnegative", "integer_counts"}:
            candidate = projection.predict_values(candidate, recipe=selected["projection"])
    if digest(args.output / "selection.json") != identity:
        raise ValueError("Readout choice changed after exposure to masked official queries")
    keys.to_parquet(args.output / "masked-query-keys.parquet", index=False)
    np.savez_compressed(args.output / "predictions.npz", control=control, candidate=candidate,
                        **key_arrays(keys))
    # Exactly the validation-selected vector is fixed before scoring; never
    # replace a bad held-out result with a different research score or recipe.
    score = baseline["official_test_scores"]["candidate"] if choice == "incumbent" else float(source.task.evaluate(candidate, target_table=None)[meta["metric"]])
    result = dict(dataset=meta["dataset"], task=meta["task"], metric=meta["metric"], selected=choice,
        official_test_scores=dict(control=baseline["official_test_scores"]["candidate"], candidate=score),
        query_keys_sha256=key_fingerprint(keys), selection_sha256=identity,
        deployment_seed=0, official_targets_used_for_selection=False, reused_previous_score=False)
    dump(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root", "source-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("Bounded 1-8 independent teacher workers required")
    args.output.mkdir(parents=True, exist_ok=True)
    selected, meta = validate(args)
    if not args.validation_only:
        print(json.dumps(official(args, selected, meta)), flush=True)


if __name__ == "__main__":
    main()
