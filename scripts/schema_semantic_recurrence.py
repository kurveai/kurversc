"""Schema-routed recurrence/semantic readout for the complete benchmark procedure.

No task-name routing or archived predictions. Caller supplies one frozen graph,
mature label routing, and declared source metadata. Keys route OOF partitions;
they never enter the learner payload. Prediction arrays keep explicit query keys.
"""

from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from kurversc.modeling import fit_final_catboost, sample_training_rows
from kurversc.signal_challengers import ChallengerUnavailable
from scripts.activity_timeline_challenger import (
    ActivityTimeline, EventChannel, discover_channels, literal, mature_label_state, quote, table_metadata,
)
from scripts.benchmark_engagement_offset_ranker import correction_frame
from scripts.benchmark_phase_frozen_breadth_v2 import task_root_table
from scripts.benchmark_reliability_blending import group_masks
from scripts.benchmark_signal_challengers import prepared
from scripts.concurrent_portable_bagged_offset import ConcurrentPortableBaggedOffsetRanker
from scripts.crossfit_semantic_propensity import CrossfitSemanticPropensity, bounded_joint_annotation
from scripts.diagnose_event_recurrence import RecurrenceDiagnostic
from scripts.engagement_hypothesis_features import parent_text_columns
from scripts.engagement_local_comparison import freeze
from scripts.frozen_parent_semantic_embedding import FrozenParentSemanticEmbedding, MODEL, REVISION
from scripts.gap_text_readout import FieldTextReadout
from scripts.run_gap_followup_queue import digest, dump
from scripts.stable_peer_moments import stable_peer_moments
from scripts.benchmark_training_row_budget import file_sha256


TEACHER_PARAMS = dict(iterations=600, depth=5, l2_leaf_reg=30.,
                      auto_class_weights="SqrtBalanced", thread_count=2)
MAX_SEMANTIC_ROWS = 500_000
MAX_SEMANTIC_DOCUMENTS = 200_000


def declared_root(source, metadata=None):
    info = table_metadata(source)
    mapping = info["fkey_col_to_pkey_table"] or {}
    if len(mapping) != 1 or not info["time_col"]:
        raise ChallengerUnavailable("Exactly one declared predictive root and clock required")
    key, root = next(iter(mapping.items()))
    if root == "entity_table":
        if metadata is None:
            raise ChallengerUnavailable("Task's declared physical root metadata is required")
        root = task_root_table(metadata, info)
    return root, key, info["time_col"]


def scalar_payload(train, forecast, *, source_root, target, horizon, origin, output, metadata=None):
    root, entity, clock = declared_root(source_root / "tasks" / target[0] / "train.parquet", metadata)
    channels = discover_channels(source_root / "db", root)
    if not channels:
        raise ChallengerUnavailable("No declared timestamped reverse-FK event channels")
    # Existing timeline provider supports ordinary integral entity keys. Refuse
    # another key domain explicitly rather than coercing string IDs to numbers.
    for frame in (train, forecast):
        ids = pd.to_numeric(frame.__event_entity, errors="coerce")
        if ids.isna().any() or (ids != np.floor(ids)).any():
            raise ChallengerUnavailable("Timeline requires integral structural entity IDs")
    observed_path = source_root / "tasks" / target[0] / "train.parquet"
    observed_paths = [source_root / "tasks" / target[0] / f"{split}.parquet" for split in ("train", "val")]
    if not all(p.exists() for p in observed_paths):
        raise ValueError("Canonical observed training/validation tables are required")
    source_info = table_metadata(observed_path)
    availability = source_info.get("label_available_at")
    observed = pd.concat([pd.read_parquet(p, columns=[entity, clock, target[1],*([availability] if availability else [])])
                          for p in observed_paths], ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    # Freeze availability separately from each request's own query clock. This
    # also protects outer predictions when queries have timestamps > origin.
    import duckdb
    directory = output / "authorized-channels"
    directory.mkdir(exist_ok=True)
    frozen = []
    with duckdb.connect(config=dict(memory_limit="4GB", threads=2)) as connection:
        for number, channel in enumerate(channels):
            path = directory / f"channel{number}.parquet"
            if path.exists():
                raise ValueError("Never reuse channel exports without a frozen source identity")
            extra = f", {quote(channel.available_at)}" if channel.available_at and channel.available_at!=channel.timestamp else ""
            visibility = f" AND {quote(channel.available_at)}<=TIMESTAMP {literal(pd.Timestamp(origin))}" if channel.available_at else ""
            connection.execute(f"COPY (SELECT {quote(channel.entity)}, {quote(channel.timestamp)}{extra} "
                f"FROM read_parquet({literal(channel.path)}) WHERE {quote(channel.timestamp)} "
                f"< TIMESTAMP {literal(pd.Timestamp(origin))}{visibility}) TO {literal(path)} (FORMAT PARQUET)")
            frozen.append(EventChannel(path, channel.entity, channel.timestamp,channel.available_at))
    providers = [ActivityTimeline(frozen, temp_directory=output / "activity-spill", threads=2),
                 RecurrenceDiagnostic(frozen, horizon=horizon,
                                      temporary=output / "recurrence-spill", threads=2)]
    try:
        # A later forecast timestamp changes recency, not source availability.
        # Validation uses the exact origin; outer execution uses phase-truncated
        # event files supplied by the caller, never an unfrozen full database.
        blocks = []
        for name, frame in (("training", train), ("validation", forecast)):
            boundary = max(pd.Timestamp(origin), pd.to_datetime(frame.__event_cutoff).max())
            activity = providers[0].transform(frame, phase_boundary=boundary)
            observed_available = observed.loc[pd.to_datetime(observed[clock]) + pd.Timedelta(horizon) < pd.Timestamp(origin)]
            if availability:
                observed_available = observed_available.loc[pd.to_datetime(observed_available[availability]).le(pd.Timestamp(origin))]
            labels = mature_label_state(observed_available, frame, entity=entity, timestamp=clock,
                target=target[1], horizon=horizon, phase_boundary=boundary,available_at=availability)
            recurrence = providers[1].transform(frame, phase_boundary=boundary)
            block = pd.concat([activity, labels, recurrence], axis=1)
            if block.columns.duplicated().any() or len(block) != len(frame):
                raise ValueError("State feature payload collision or lost query order")
            block.to_parquet(output / f"state-{name}.parquet", index=False)
            blocks.append(block)
    finally:
        for provider in providers:
            provider.close()
    dump(output / "source-audit.json", dict(root=root, entity_key=entity, clock=clock,
        channels=[dict(path=str(c.path), entity=c.entity, timestamp=c.timestamp) for c in channels],
        source_sha256={str(p): file_sha256(p) for p in [*observed_paths, *[c.path for c in channels]]},
        origin=str(origin), horizon=str(horizon), scalar_columns=list(blocks[0])))
    return root, blocks


def teacher_fit(x, routing, mask, *, target, schema, seed, row_limit, params):
    y = routing.loc[mask, target].reset_index(drop=True)
    if set(pd.unique(y)) != {0, 1}:
        raise ChallengerUnavailable("Each bounded mature teacher cohort requires both classes")
    tx = x.loc[mask].reset_index(drop=True)
    tx, y = sample_training_rows(tx, y, limit=row_limit, task="classification", random_state=seed)
    model, _, _ = fit_final_catboost(tx, y, task="classification",
        categorical=[i for i, c in enumerate(tx) if c in schema["categorical_columns"]],
        random_state=seed, model_params=params)
    return model


def prepare_phase(train, forecast, *, metadata, source_root, origin, output,
                  workers=4, row_limit=100_000, encoder=None, teacher_params=None,
                  maximum_rows=MAX_SEMANTIC_ROWS, maximum_documents=MAX_SEMANTIC_DOCUMENTS):
    schema, target = metadata["feature_schema"], metadata["target"]
    if metadata["metric"] != "roc_auc" or not 1 <= workers <= 8 or row_limit < 1:
        raise ChallengerUnavailable("Bounded ordinary binary phase required")
    if maximum_rows < 1 or maximum_documents < 2:
        raise ValueError("Positive semantic row/document capacity required")
    if len(train) > maximum_rows:
        raise ChallengerUnavailable(f"Semantic support has {len(train)} rows; capacity is {maximum_rows}")
    if (pd.to_datetime(train.__event_cutoff) + pd.Timedelta(metadata["label_horizon"]) > pd.Timestamp(origin)).any():
        raise ValueError("Immature labels reached a semantic recurrence fit")
    if train.duplicated(["__event_entity", "__event_cutoff"]).any():
        raise ValueError("Unique entity/time training observations required")
    routing = train[[target, "__event_entity", "__event_cutoff"]].reset_index(drop=True)
    cuts = sorted(pd.to_datetime(routing.__event_cutoff).unique())
    if not 3 <= len(cuts) <= 4:
        raise ChallengerUnavailable("Three initial cohorts plus at most one observed refit required")
    root, _, _ = declared_root(source_root / "tasks" / metadata["task"] / "train.parquet", metadata)
    raw, vraw = prepared(train, schema), prepared(forecast, schema)
    first = pd.to_datetime(routing.__event_cutoff).eq(cuts[0]).to_numpy()
    columns = parent_text_columns(raw.loc[first], schema, root)
    if not columns:
        raise ChallengerUnavailable("No training-discovered parent text payload")
    # Check capacity before any event SQL, teacher fit, or pretrained weight
    # loading. This is only an optional readout limit, never a graph/feature cap.
    documents = set(FrozenParentSemanticEmbedding.documents(raw, columns))
    documents.update(FrozenParentSemanticEmbedding.documents(vraw, columns))
    if len(documents - {""}) > maximum_documents:
        raise ChallengerUnavailable(f"Semantic documents exceed capacity {maximum_documents}")
    del documents
    root, (state, vstate) = scalar_payload(train, forecast, source_root=source_root,
        target=(metadata["task"], target), horizon=metadata["label_horizon"], origin=origin, output=output,
        metadata=metadata)
    # Use the same stable numerical representation as the reproduced winner.
    x, audit = stable_peer_moments(pd.concat([raw, state], axis=1), categorical=schema["categorical_columns"])
    vx, vaudit = stable_peer_moments(pd.concat([vraw, vstate], axis=1), categorical=schema["categorical_columns"])
    params = dict(TEACHER_PARAMS if teacher_params is None else teacher_params)
    masks = group_masks(routing.__event_entity)
    oof = np.empty(len(routing))
    text = np.empty(len(routing))
    def fit_group(item):
        number, (fit, query) = item
        if set(routing.__event_entity[fit]) & set(routing.__event_entity[query]):
            raise ValueError("An OOF teacher saw its forecast entity")
        vectors = []
        for cutoff in cuts:
            mask = fit & pd.to_datetime(routing.__event_cutoff).eq(cutoff).to_numpy()
            model = teacher_fit(x, routing, mask, target=target, schema=schema,
                                seed=0, row_limit=row_limit, params=params)
            vectors.append(model.predict_proba(x.loc[query], thread_count=2)[:, 1])
            del model
        lexical = FieldTextReadout().fit(raw.loc[fit], routing.loc[fit, target], columns)
        return number, query, np.mean(vectors, axis=0), lexical.predict_proba(raw.loc[query])[:, 1]
    with ThreadPoolExecutor(max_workers=min(workers, 3)) as pool:
        for number, query, pred, word in pool.map(fit_group, enumerate(masks)):
            oof[query], text[query] = pred, word
    lexical = FieldTextReadout().fit(raw, routing[target], columns)
    held_text = lexical.predict_proba(vraw)[:, 1]
    encoder = FrozenParentSemanticEmbedding(dimensions=32) if encoder is None else encoder
    encoder.fit(raw.loc[first], columns=columns)
    projected, vprojected = encoder.transform(raw), encoder.transform(vraw)
    annotation = CrossfitSemanticPropensity().fit(projected, routing[target],
        entities=routing.__event_entity, timestamps=routing.__event_cutoff,
        origin=origin, horizon=metadata["label_horizon"])
    semantic = bounded_joint_annotation(projected, annotation.oof)
    vsemantic = bounded_joint_annotation(vprojected, annotation.transform(vprojected))
    correction = pd.concat([correction_frame(state, oof, text), semantic], axis=1)
    output.mkdir(parents=True, exist_ok=True)
    freeze(output / "phase-audit.json", dict(cuts=list(map(str, cuts)), origin=str(origin),
        model_params=params, row_limit=row_limit, maximum_rows=maximum_rows,
        maximum_documents=maximum_documents, root=root, text_columns=columns,
        scalar_columns=list(state), semantic=encoder.audit, annotation=annotation.audit,
        graph_numerical_audit=audit, forecast_numerical_audit=vaudit,
        own_entity_labels_excluded=True, target_not_in_predictors=True,
        query_keys_sha256=__import__("hashlib").sha256(pd.util.hash_pandas_object(
            forecast[["__event_entity", "__event_cutoff"]], index=False).values.tobytes()).hexdigest()))
    np.savez_compressed(output / "oof-teachers.npz", base=oof, text=text, held_text=held_text)
    return dict(x=x, vx=vx, routing=routing, state=state, vstate=vstate,
        oof=oof, text=text, held_text=held_text, semantic=semantic, vsemantic=vsemantic,
        correction=correction, cuts=cuts, params=params, schema=schema, target=target,
        row_limit=row_limit, origin=origin, horizon=metadata["label_horizon"], workers=workers)


def fit_seed(phase, seed, *, repeat=False):
    x, vx, routing = phase["x"], phase["vx"], phase["routing"]
    def member(cutoff):
        mask = pd.to_datetime(routing.__event_cutoff).eq(cutoff).to_numpy()
        model = teacher_fit(x, routing, mask, target=phase["target"], schema=phase["schema"],
            seed=seed, row_limit=phase["row_limit"], params=phase["params"])
        pred = model.predict_proba(vx, thread_count=2)[:, 1]
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(pred, restored.predict_proba(vx, thread_count=2)[:, 1])
        return pred
    with ThreadPoolExecutor(max_workers=phase["workers"]) as pool:
        head = np.mean(list(pool.map(member, phase["cuts"])), axis=0)
    model = ConcurrentPortableBaggedOffsetRanker(seed=seed).fit(phase["correction"],
        routing[phase["target"]], phase["oof"], entities=routing.__event_entity,
        timestamps=routing.__event_cutoff, origin=phase["origin"], horizon=phase["horizon"])
    covariates = pd.concat([correction_frame(phase["vstate"], head, phase["held_text"]), phase["vsemantic"]], axis=1)
    corrected = model.predict_values(covariates, head, strength=1.)
    np.testing.assert_array_equal(corrected, pickle.loads(pickle.dumps(model)).predict_values(covariates, head, strength=1.))
    if repeat:
        other_head, other, _ = fit_seed(phase, seed, repeat=False)
        np.testing.assert_allclose(head, other_head, rtol=0, atol=1e-9)
        np.testing.assert_allclose(corrected, other, rtol=0, atol=1e-9)
    gc.collect()
    return head, corrected, model.audit
