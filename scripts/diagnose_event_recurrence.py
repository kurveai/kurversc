"""Isolated validation diagnostic: long-tail, channel-conditioned recurrence.

No production provider, family, graph, selector, or defaults are changed.
Every ordinary event is strictly earlier than its request's timestamp.
The fixed feature menu uses declared event channels, not task/schema names.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
import psutil

from scripts.activity_timeline_challenger import EventChannel, literal, quote


class RecurrenceDiagnostic:
    def __init__(self, channels, *, horizon, temporary, threads=2):
        if not 1 <= len(channels) <= 6:
            raise ValueError("Between one and six frozen event channels required")
        self.channels = tuple(channels)
        self.horizon_days = pd.Timedelta(horizon).total_seconds() / 86400
        if self.horizon_days <= 0:
            raise ValueError("Positive prediction horizon required")
        Path(temporary).mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(config=dict(memory_limit="4GB", threads=threads,
            temp_directory=str(temporary), max_temp_directory_size="40GB"))
        pieces = [f"SELECT CAST({quote(c.entity)} AS VARCHAR) entity, "
            f"{quote(c.timestamp)}::TIMESTAMP event_time, "
            f"{quote(c.available_at or c.timestamp)}::TIMESTAMP visible_at, {i} channel "
            f"FROM read_parquet({literal(c.path)}) WHERE {quote(c.entity)} IS NOT NULL "
            f"AND {quote(c.timestamp)} IS NOT NULL" for i, c in enumerate(channels)]
        self.connection.execute("CREATE VIEW event_rows AS " + " UNION ALL ".join(pieces))

    def transform(self, requests, *, phase_boundary):
        times = pd.to_datetime(requests["__event_cutoff"])
        if times.isna().any() or times.max() > pd.Timestamp(phase_boundary):
            raise ValueError("Requests exceed the authorized phase or lack timestamps")
        if requests["__event_entity"].isna().any():
            raise ValueError("Missing root entity")
        chunks = []
        for cutoff in sorted(times.unique()):
            positions = np.flatnonzero(times.eq(cutoff).to_numpy())
            ids = requests.iloc[positions]["__event_entity"].astype(str).to_numpy()
            self.connection.register("requested_roots", pd.DataFrame({"entity": ids}).drop_duplicates())
            stamp = f"TIMESTAMP {literal(pd.Timestamp(cutoff))}"
            fields = ['COUNT(*) AS "days"', "SUM(events) events", "MAX(age_days) first_age",
                "MIN(age_days) recency", "AVG(gap_days) gap_mean", "MEDIAN(gap_days) gap_median",
                "QUANTILE_CONT(gap_days, .9) gap_p90", "MAX(gap_days) gap_max",
                "STDDEV_POP(gap_days) gap_std", "COUNT(gap_days) gap_support",
                "AVG(gap_days) FILTER (WHERE recent_rank<=10) recent_gap_mean",
                "AVG(gap_days) FILTER (WHERE recent_rank BETWEEN 11 AND 30) older_gap_mean",
                "MAX(gap_days) FILTER (WHERE recent_rank=1) previous_gap",
                "COUNT(*) FILTER (WHERE gap_days>=90) returns_90",
                "COUNT(*) FILTER (WHERE gap_days>=365) returns_365",
                "COUNT(*) FILTER (WHERE age_days<365) days_365",
                "COUNT(*) FILTER (WHERE age_days<730) days_730",
                "COUNT(*) FILTER (WHERE gap_days>=current_age) gaps_beyond_current_age",
                f"COUNT(*) FILTER (WHERE gap_days>=current_age AND "
                f"gap_days<current_age+{self.horizon_days!r}) gaps_next_horizon"]
            query = f"""
                WITH past AS (
                    SELECT e.* FROM event_rows e JOIN requested_roots r USING(entity)
                    WHERE event_time<{stamp} AND visible_at<={stamp}
                ), expanded AS (
                    SELECT * FROM past UNION ALL
                    SELECT entity,event_time,visible_at,-1 channel FROM past
                ), days AS (
                    SELECT entity,channel,CAST(event_time AS DATE) event_day,
                        COUNT(*) events,MAX(event_time) last_event
                    FROM expanded GROUP BY entity,channel,event_day
                ), gaps AS (
                    SELECT *,DATE_DIFF('day',LAG(event_day) OVER
                        (PARTITION BY entity,channel ORDER BY event_day),event_day) gap_days,
                        EPOCH({stamp}-last_event)/86400.0 age_days,
                        ROW_NUMBER() OVER(PARTITION BY entity,channel ORDER BY event_day DESC) recent_rank,
                        EPOCH({stamp}-MAX(last_event) OVER(PARTITION BY entity,channel))/86400.0 current_age
                    FROM days
                )
                SELECT entity,channel,{','.join(fields)} FROM gaps GROUP BY entity,channel
            """
            rows = self.connection.execute(query).df()
            blocks = []
            for channel in (-1, *range(len(self.channels))):
                block = rows.loc[rows.channel.eq(channel)].drop(columns="channel").set_index("entity")
                block = block.reindex(ids).reset_index(drop=True)
                counts = ["days", "events", "gap_support", "returns_90", "returns_365", "days_365",
                          "days_730", "gaps_beyond_current_age", "gaps_next_horizon"]
                block[counts] = block[counts].fillna(0)
                block["normalized_recency_median"] = block.recency / (1 + block.gap_median)
                block["normalized_recency_p90"] = block.recency / (1 + block.gap_p90)
                block["gap_exceedance"] = block.gaps_beyond_current_age / block.gap_support.replace(0, np.nan)
                # Conditional empirical gap sketch, NOT a fitted survival model;
                # expose denominators because closed gaps are sparse/censored.
                block["conditional_gap_fraction"] = block.gaps_next_horizon / block.gaps_beyond_current_age.replace(0, np.nan)
                block["cadence_change"] = (1 + block.older_gap_mean) / (1 + block.recent_gap_mean)
                block["pre730_days"] = block.days - block.days_730
                name = "all" if channel == -1 else f"channel{channel}"
                blocks.append(block.add_prefix(f"rc_{name}_"))
            result = pd.concat(blocks, axis=1)
            result["__position"] = positions
            chunks.append(result)
        if not chunks:
            raise ValueError("Nonempty requests required")
        return pd.concat(chunks).sort_values("__position").drop(columns="__position").reset_index(drop=True).astype("float32")

    def close(self):
        self.connection.close()


def main():
    from kurversc.temporal_policy import TemporalFitPolicy, fit_temporal_policy, _score
    from scripts.benchmark_signal_challengers import dump, incumbent_evidence, prepared
    from scripts.select_activity_readout import cache_locations

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--matched-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if psutil.virtual_memory().available < 12 * 2**30:
        raise RuntimeError("Need 12 GiB available; no unrelated job will be stopped")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    reference = json.loads((args.matched_control / "manifest.json").read_text())
    metadata = json.loads((args.task_root / "selection/manifest.json").read_text())["metadata"]
    if reference["representation"] != "full" or reference["metadata"] != {k:v for k,v in metadata.items() if k != "feature_schema"}:
        raise ValueError("Matched control must use the same full frozen graph")
    params = dict(iterations=reference["iterations"],depth=reference["depth"],
                  l2_leaf_reg=reference["l2_leaf_reg"],thread_count=args.threads)
    if reference["class_weight"] != "None":
        params["auto_class_weights"] = reference["class_weight"]
    base = TemporalFitPolicy(**json.loads((args.task_root / "selection/results.json").read_text())["selected_policy"])
    cache, snapshots = cache_locations(args.task_root)
    snapshot = json.loads((snapshots / f"fold{args.fold}/manifest.json").read_text())
    origin = snapshot["snapshot_cutoff"]
    if origin != snapshot["origin"]:
        raise ValueError("Diagnostic requires the exact single-origin frozen cache")
    audit = json.loads((args.task_root / "raw-audit" / f"fold{args.fold}-incumbent-seed0.json").read_text())["audit"]
    schema, target = metadata["feature_schema"], metadata["target"]
    columns = [*schema["columns"], target, "__event_entity", "__event_cutoff"]
    train = pd.read_parquet(cache / "history.parquet", columns=columns,
        filters=[("__event_cutoff", "in", audit["initial_cutoffs"] + audit["observed_refit_cutoffs"])])
    heldout = pd.read_parquet(snapshots / f"fold{args.fold}/validation.parquet", columns=columns)
    control = incumbent_evidence(args.task_root / "raw-audit", args.fold, args.seed, base)
    np.testing.assert_array_equal(control["truth"], heldout[target])
    np.testing.assert_array_equal(control["groups"], heldout.__event_entity)
    x, vx = prepared(train, schema), prepared(heldout, schema)
    for name in ("activity", "label_state"):
        tx = pd.read_parquet(args.matched_control / f"fold{args.fold}-{name}-training.parquet")
        v = pd.read_parquet(args.matched_control / f"fold{args.fold}-{name}-validation.parquet")
        if len(tx) != len(x) or len(v) != len(vx):
            raise ValueError("Activity/state cache alignment differs")
        x, vx = pd.concat([x, tx], axis=1), pd.concat([vx, v], axis=1)
    channels = [EventChannel(Path(c["path"]), c["entity"], c["timestamp"]) for c in reference["channels"]]
    manifest = dict(metadata=metadata, origin=origin, fold=args.fold, seed=args.seed,
        control_source=str(args.matched_control), model_params=params, channels=reference["channels"],
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        feature_budget=25 * (len(channels) + 1),
        scope="Single validation diagnostic; support, graph and learner budget matched; no test/default changes")
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(manifest,default=str)):
        raise ValueError("Output has different immutable provenance")
    dump(path,manifest)
    provider = RecurrenceDiagnostic(channels,horizon=metadata["label_horizon"],temporary=args.output / "spill",threads=args.threads)
    try:
        blocks = []
        for label, frame in (("train",train),("validation",heldout)):
            print(json.dumps(dict(phase="recurrence_features",split=label,rows=len(frame))),flush=True)
            block = provider.transform(frame,phase_boundary=origin)
            block.to_parquet(args.output / f"{label}-recurrence.parquet",index=False)
            blocks.append(block)
    finally:
        provider.close()
    x, vx = pd.concat([x,blocks[0]],axis=1),pd.concat([vx,blocks[1]],axis=1)
    if len(blocks[0].columns) > manifest["feature_budget"]:
        raise ValueError("Recurrence feature budget exceeded")
    print(json.dumps(dict(phase="fit",added_features=len(blocks[0].columns),model_params=params)),flush=True)
    categorical = [i for i,c in enumerate(x) if c in schema["categorical_columns"]]
    model, fit_audit = fit_temporal_policy(x,train[target],pd.to_datetime(train.__event_cutoff),origin=origin,
        label_horizon=metadata["label_horizon"],refit_frames=metadata["validation_cutoffs"],
        policy=TemporalFitPolicy("incumbent"),task="classification",categorical=categorical,
        random_state=args.seed,row_limit=100000,model_params=params)
    prediction = model.predict_proba(vx)[:,1]
    with np.load(args.matched_control / f"fold{args.fold}-activity_state-seed{args.seed}.npz") as a:
        np.testing.assert_array_equal(a["truth"],control["truth"])
        np.testing.assert_array_equal(a["groups"],control["groups"])
        matched = .5*(a["prediction"]+control["prediction"])
    candidate = .5*(prediction+control["prediction"])
    scores = dict(matched_control=_score(control["truth"],matched,"classification"),
        recurrence=_score(control["truth"],candidate,"classification"))
    np.savez_compressed(args.output / "predictions.npz",truth=control["truth"],groups=control["groups"],
        candidate=candidate,control=matched)
    report = dict(manifest=manifest,fit_audit=fit_audit,scores=scores,
        gain=scores["recurrence"]-scores["matched_control"],added_features=list(blocks[0]),
        top_importance=[sorted(zip(x.columns,m.feature_importances_),key=lambda r:-r[1])[:25] for m in model.models],
        elapsed_seconds=time.monotonic()-start)
    dump(args.output / "results.json",report)
    print(json.dumps(dict(phase="completed",scores=scores,gain=report["gain"],elapsed_seconds=report["elapsed_seconds"])),flush=True)
    del model
    gc.collect()


if __name__ == "__main__":
    main()
