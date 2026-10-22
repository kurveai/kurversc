"""Validation-only cohort audit of dormant positives and matched negatives.

No official split, feature selection, model fitting, or default mutation. The
outcome is used to diagnose errors, never to generate predictive covariates.
Event timelines use declared reverse-FK channels and strict request cutoffs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts.activity_timeline_challenger import literal, quote
from scripts.select_activity_readout import cache_locations


def positive_pair_errors(truth, prediction):
    """Number of misordered negative pairs per positive, half credit for ties."""
    truth, prediction = np.asarray(truth), np.asarray(prediction, float)
    if truth.shape != prediction.shape or not np.isin(truth, [0, 1]).all():
        raise ValueError("Aligned binary truth and prediction required")
    if not np.isfinite(prediction).all() or not np.any(truth == 0):
        raise ValueError("Finite scores and negative examples required")
    negative = np.sort(prediction[truth == 0])
    left = np.searchsorted(negative, prediction, side="left")
    right = np.searchsorted(negative, prediction, side="right")
    errors = len(negative) - right + .5 * (right - left)
    return np.where(truth == 1, errors, 0.)


def cohort_summary(frame, mask, prediction_name):
    selected = frame.loc[mask]
    positive = selected.truth.eq(1)
    all_positive = frame.truth.eq(1)
    errors = f"{prediction_name}_pair_errors"
    total_error = frame[errors].sum()
    return dict(rows=len(selected), positives=int(positive.sum()),
        positive_rate=float(positive.mean()) if len(selected) else None,
        share_of_all_positives=float(positive.sum()/all_positive.sum()),
        share_of_all_pair_errors=float(selected[errors].sum()/total_error) if total_error else 0.,
        mean_positive_score=float(selected.loc[positive, prediction_name].mean()) if positive.any() else None,
        mean_negative_score=float(selected.loc[~positive, prediction_name].mean()) if (~positive).any() else None)


def matched_examples(frame, *, horizon, count=12):
    """Match on past recency/counts, not predictions or heldout outcomes as features.

    Labels identify retrospective positive/negative examples for inspection
    only. No selected examples or distances enter model training.
    """
    dormant = frame.rc_all_recency.ge(horizon)
    positive = frame.loc[dormant & frame.truth.eq(1)].nlargest(count, "recurrence_pair_errors")
    negative = frame.loc[dormant & frame.truth.eq(0)]
    if positive.empty or negative.empty:
        return []
    columns = ["rc_all_recency", "rc_all_days"]
    coordinates = np.log1p(negative[columns].fillna(0).to_numpy(float))
    tree = cKDTree(coordinates)
    distances, positions = tree.query(np.log1p(positive[columns].fillna(0).to_numpy(float)), k=1)
    return [dict(positive_row=int(p), negative_row=int(negative.index[n]),
        log_distance=float(d)) for p, n, d in zip(positive.index, positions, distances)]


def event_timelines(channels, examples, cutoff, *, temporary):
    """Small, past-only active-day timelines; no raw text or future outcomes."""
    Path(temporary).mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(config=dict(memory_limit="1GB", threads=1,
        temp_directory=str(temporary), max_temp_directory_size="8GB"))
    output = {}
    try:
        roots = pd.DataFrame({"entity": sorted(set(str(e) for e in examples))})
        connection.register("audit_roots", roots)
        stamp = f"TIMESTAMP {literal(pd.Timestamp(cutoff))}"
        for i, channel in enumerate(channels):
            query = f"""SELECT entity,CAST(event_time AS DATE) active_day,COUNT(*) events
                FROM (SELECT CAST({quote(channel['entity'])} AS VARCHAR) entity,
                    {quote(channel['timestamp'])}::TIMESTAMP event_time
                    FROM read_parquet({literal(channel['path'])})) e
                JOIN audit_roots USING(entity) WHERE event_time<{stamp}
                GROUP BY entity,active_day ORDER BY entity,active_day"""
            rows = connection.execute(query).df()
            for entity, group in rows.groupby("entity", sort=False):
                output.setdefault(entity, {})[f"channel{i}"] = [
                    dict(day=str(day), events=int(events)) for day, events
                    in zip(group.active_day.dt.date, group.events)]
    finally:
        connection.close()
    return output


def safe_record(row):
    record = {}
    for name, value in row.items():
        if name in ("__event_entity", "__event_cutoff"):
            continue
        record[name] = None if pd.isna(value) else float(value)
    record["entity_fingerprint"] = hashlib.sha256(str(row.__event_entity).encode()).hexdigest()[:12]
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--matched-control", type=Path, required=True)
    parser.add_argument("--recurrence-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.examples <= 30:
        raise ValueError("Bounded one-to-thirty retrospective examples required")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.task_root/"selection/manifest.json").read_text())["metadata"]
    channels = json.loads((args.matched_control/"manifest.json").read_text())["channels"]
    selection = json.loads(args.recurrence_selection.read_text())
    paths = {item["fold"]: Path(item["path"]) for item in selection["identities"] if item["seed"] == 0}
    _, snapshots = cache_locations(args.task_root)
    h = pd.Timedelta(metadata["label_horizon"]).total_seconds()/86400
    output = dict(scope="Retrospective validation diagnosis only, no official test access or predictive-feature selection",
        prediction_horizon_days=h, channels=channels, folds=[])
    for fold in map(int, args.folds.split(",")):
        root = paths[fold]
        snapshot = json.loads((snapshots/f"fold{fold}/manifest.json").read_text())
        validation = pd.read_parquet(snapshots/f"fold{fold}/validation.parquet",
            columns=[metadata["target"], "__event_entity", "__event_cutoff"])
        with np.load(root/"predictions.npz") as saved:
            np.testing.assert_array_equal(saved["truth"], validation[metadata["target"]])
            np.testing.assert_array_equal(saved["groups"], validation.__event_entity)
            predictions = {name: saved[key].copy() for name, key in
                (("activity_state", "control"), ("recurrence", "candidate"))}
        frame = pd.read_parquet(root/"validation-recurrence.parquet")
        if len(frame) != len(validation):
            raise ValueError("Past covariate and vector alignment differs")
        for name in ("__event_entity", "__event_cutoff"):
            frame[name] = validation[name].to_numpy()
        frame["truth"] = validation[metadata["target"]].to_numpy()
        for name, values in predictions.items():
            frame[name] = values
            frame[f"{name}_pair_errors"] = positive_pair_errors(frame.truth, values)
        dormant = frame.rc_all_recency.ge(h)
        summaries = []
        for name, mask in (("all", np.ones(len(frame), bool)), ("dormant_horizon", dormant),
            ("dormant_730d", frame.rc_all_recency.ge(730))):
            summaries.append(dict(cohort=name, **{p: cohort_summary(frame, mask, p) for p in predictions}))
        strata = []
        dimensions = dict(
            lifetime_active_days=("rc_all_days", [0, 1, 5, 20, 100, np.inf]),
            prior_long_returns=("rc_all_returns_90", [-1, 0, 1, 3, 10, np.inf]),
            silence_to_usual_gap=("rc_all_normalized_recency_median", [0, 1, 3, 10, 100, np.inf]),
            past_gap_support_beyond_current_silence=("rc_all_gaps_beyond_current_age", [-1, 0, 1, 3, 10, np.inf]))
        for label, (column, bounds) in dimensions.items():
            buckets = pd.cut(frame[column], bounds, include_lowest=True)
            for bucket in buckets.cat.categories:
                mask = dormant & buckets.eq(bucket)
                strata.append(dict(dimension=label, bucket=str(bucket),
                    **{p: cohort_summary(frame, mask, p) for p in predictions}))
        contrasts = []
        # All columns below are covariates already available to the recurrence head.
        for column in [c for c in frame if c.startswith("rc_")]:
            a, b = frame.loc[dormant & frame.truth.eq(1), column], frame.loc[dormant & frame.truth.eq(0), column]
            contrasts.append(dict(feature=column, positive_median=float(a.median()) if a.notna().any() else None,
                negative_median=float(b.median()) if b.notna().any() else None,
                positive_mean=float(a.mean()) if a.notna().any() else None,
                negative_mean=float(b.mean()) if b.notna().any() else None))
        matches = matched_examples(frame, horizon=h, count=args.examples)
        ids = [frame.loc[m[key], "__event_entity"] for m in matches for key in ("positive_row", "negative_row")]
        timelines = event_timelines(channels, ids, snapshot["origin"], temporary=args.output/"spill")
        examples = []
        for match in matches:
            example = dict(log_matching_distance=match["log_distance"])
            for label in ("positive", "negative"):
                row = frame.loc[match[f"{label}_row"]]
                example[label] = dict(covariates=safe_record(row),
                    past_event_days=timelines.get(str(row.__event_entity), {}))
            examples.append(example)
        record = dict(fold=fold, origin=snapshot["origin"], cohorts=summaries,
            dormant_strata=strata, covariate_contrasts=contrasts, matched_examples=examples)
        (args.output/f"fold{fold}.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        frame.to_parquet(args.output/f"fold{fold}-covariates.parquet", index=False)
        output["folds"].append(dict(fold=fold, origin=snapshot["origin"], cohorts=summaries))
        print(json.dumps(dict(fold=fold, phase="completed", cohorts=summaries)), flush=True)
    (args.output/"results.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
