"""Deterministic historical-root sampling that accounts for label density."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalFitPlan:
    """A bounded final-fit regime chosen from cutoff availability, not labels."""

    training_frames: int
    estimator_mode: str
    recent_frame_rows: tuple[int, ...]
    reason: str


def choose_temporal_fit_plan(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    timestamp: str,
    root_frame: pd.DataFrame | None = None,
    label_key: str | tuple[str, ...] | None = None,
    root_key: str | tuple[str, ...] | None = None,
    root_timestamp: str | None = None,
    max_training_rows: int = 100_000,
    minimum_frame_rows: int = 100,
    balanced_frame_ratio: float = 0.5,
) -> TemporalFitPlan:
    """Choose a bounded three-cutoff fit for single-cutoff validation tasks.

    Only prediction timestamps, entity keys, root availability, and row counts
    participate. No target values or held-out test data influence the choice.
    The caller still needs to catch materialization resource errors and fall
    back to its existing one-cutoff fit.
    """

    if max_training_rows < 1 or minimum_frame_rows < 1:
        raise ValueError("Temporal fit row limits must be positive")
    if not 0 < balanced_frame_ratio <= 1:
        raise ValueError("balanced_frame_ratio must be in (0, 1]")
    if timestamp not in training or timestamp not in validation:
        raise KeyError(f"Missing temporal fit timestamp column {timestamp!r}")
    train_times = pd.to_datetime(training[timestamp])
    cutoff_dates = pd.Index(sorted(train_times.dropna().unique()))
    effective_times = train_times
    if root_frame is not None:
        if label_key is None or root_key is None:
            raise ValueError(
                "Root-aware temporal fit requires label_key and root_key"
            )
        label_keys = (label_key,) if isinstance(label_key, str) else tuple(label_key)
        root_keys = (root_key,) if isinstance(root_key, str) else tuple(root_key)
        if len(label_keys) != len(root_keys):
            raise ValueError("Root and label keys must have equal widths")
        root_columns = [*root_keys, *([root_timestamp] if root_timestamp else [])]
        root_dates = root_frame[root_columns].dropna(subset=root_keys).rename(
            columns={
                **dict(zip(root_keys, label_keys, strict=True)),
                **(
                    {root_timestamp: "__kurversc_root_available_at"}
                    if root_timestamp
                    else {}
                ),
            }
        )
        if root_timestamp is not None:
            root_dates["__kurversc_root_available_at"] = pd.to_datetime(
                root_dates["__kurversc_root_available_at"]
            )
        matched = training.dropna(subset=label_keys)[[*label_keys]].assign(
            __kurversc_prediction_time=train_times
        ).merge(
            root_dates,
            on=list(label_keys),
            how="inner",
            validate="many_to_one",
        )
        if root_timestamp is None:
            effective_times = matched["__kurversc_prediction_time"]
        else:
            effective_times = matched.loc[
                matched["__kurversc_root_available_at"].le(
                    matched["__kurversc_prediction_time"]
                ),
                "__kurversc_prediction_time",
            ]
    elif root_timestamp is not None:
        raise ValueError("root_timestamp requires root_frame")
    counts = effective_times.value_counts().reindex(cutoff_dates, fill_value=0)
    recent = tuple(int(count) for count in counts.iloc[-3:].tolist())
    if pd.to_datetime(validation[timestamp]).dropna().nunique() != 1:
        return TemporalFitPlan(1, "frame_ensemble", recent, "multiple_validation_cutoffs")
    if len(counts) < 3:
        return TemporalFitPlan(1, "frame_ensemble", recent, "insufficient_training_cutoffs")
    if min(recent) < minimum_frame_rows:
        return TemporalFitPlan(1, "frame_ensemble", recent, "sparse_training_cutoff")
    if sum(recent) > max_training_rows:
        return TemporalFitPlan(1, "frame_ensemble", recent, "training_row_budget")
    mode = (
        "frame_ensemble"
        if min(recent) / max(recent) >= balanced_frame_ratio
        else "pooled"
    )
    return TemporalFitPlan(3, mode, recent, "three_cutoff_candidate")


def sample_mature_history(
    training,
    *,
    timestamp,
    available_at,
    cutoff,
    preferred_cutoffs=8,
    per_cutoff=1200,
    minimum_rows=2000,
    random_state=0,
):
    """Keep recent cutoff diversity; backfill sparse histories to a row floor.

    A fixed number of dates can mean either millions of examples or only a few
    dozen. Always enforce strict maturity, never inspect target values, and
    never oversample/duplicate rows to manufacture the requested support.
    """
    if min(preferred_cutoffs, per_cutoff) < 1 or minimum_rows < 0:
        raise ValueError("Sampling budgets must be nonnegative with positive cutoffs")
    if training[[timestamp, available_at]].isna().any().any():
        raise ValueError("Prediction and label-availability times must be complete")
    available = pd.to_datetime(training[available_at])
    if (available < pd.to_datetime(training[timestamp])).any():
        raise ValueError("Labels cannot be available before prediction time")
    eligible = training.loc[available.lt(pd.Timestamp(cutoff))]
    chosen, count = [], 0
    groups = list(eligible.groupby(timestamp, sort=True))
    for _, block in reversed(groups):
        if len(chosen) >= preferred_cutoffs and count >= minimum_rows:
            break
        sampled = block.sample(n=min(per_cutoff, len(block)), random_state=random_state)
        chosen.append(sampled)
        count += len(sampled)
    if not chosen:
        raise ValueError("No strictly mature historical examples")
    return pd.concat(reversed(chosen), ignore_index=True)
