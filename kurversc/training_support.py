"""Opt-in, target-blind selection of bounded historical training support."""

from dataclasses import dataclass

import pandas as pd


def available_root_training(
    training: pd.DataFrame,
    root_frame: pd.DataFrame,
    *,
    label_key: str | tuple[str, ...],
    root_key: str | tuple[str, ...],
    timestamp: str,
    root_timestamp: str | None = None,
) -> pd.DataFrame:
    """Count only examples whose root exists at their prediction cutoff.

    Uses declared root metadata, not targets or future evaluation cohorts.
    Original training columns and row order are retained. This is planning
    support only: historical label/provider sources remain untouched.
    """
    label_keys = (label_key,) if isinstance(label_key, str) else tuple(label_key)
    root_keys = (root_key,) if isinstance(root_key, str) else tuple(root_key)
    if not label_keys or len(label_keys) != len(root_keys):
        raise ValueError("Root and label keys must have equal positive widths")
    roots = root_frame.dropna(subset=list(root_keys))
    if roots.duplicated(list(root_keys)).any():
        raise ValueError("Root keys must be unique")
    root_payload = roots[list(root_keys)].rename(
        columns=dict(zip(root_keys, label_keys))
    )
    if root_timestamp:
        root_payload = root_payload.assign(
            __support_available=pd.to_datetime(roots[root_timestamp]).to_numpy()
        )
    examples = training[list(label_keys)].assign(
        __support_position=range(len(training)),
        __support_prediction=pd.to_datetime(training[timestamp]).to_numpy(),
    )
    joined = examples.merge(
        root_payload, on=list(label_keys), how="inner", validate="many_to_one"
    )
    if root_timestamp:
        joined = joined.loc[joined.__support_available.le(joined.__support_prediction)]
    positions = sorted(joined.__support_position.tolist())
    return training.iloc[positions].copy()


@dataclass(frozen=True)
class TrainingSupport:
    cutoffs: tuple[pd.Timestamp, ...]
    rows: int
    minimum_rows: int
    complete: bool


def select_training_support(
    training: pd.DataFrame,
    *,
    timestamp: str,
    evaluation_cutoff,
    label_horizon,
    minimum_rows: int,
    minimum_frames: int = 1,
    max_frames: int = 64,
) -> TrainingSupport:
    """Backfill recent completed-label cutoffs to a bounded support floor.

    Completion at the evaluation boundary is legal for downstream fitting:
    a label window ending at that boundary is fully observed. This does NOT
    change peer-feature eligibility, which retains its strict-before rule.
    Counts describe the supplied cohort, so use materialized rows when root
    availability reduces it. No targets, entity identities or task names enter
    the decision; rows are never duplicated to manufacture support.
    """
    if minimum_rows < 1 or not 1 <= minimum_frames <= max_frames:
        raise ValueError("Positive row floor and valid bounded frame counts required")
    times = pd.to_datetime(training[timestamp])
    if times.isna().any():
        raise ValueError("Training prediction timestamps must be complete")
    horizon = pd.Timedelta(label_horizon)
    if horizon < pd.Timedelta(0):
        raise ValueError("Label horizon must be nonnegative")
    eligible = times.loc[(times + horizon).le(pd.Timestamp(evaluation_cutoff))]
    counts = eligible.value_counts().sort_index(ascending=False)
    selected, rows = [], 0
    for cutoff, count in counts.items():
        if len(selected) == max_frames:
            break
        selected.append(pd.Timestamp(cutoff))
        rows += int(count)
        if rows >= minimum_rows and len(selected) >= minimum_frames:
            break
    if not selected:
        raise ValueError("No completed historical labels before evaluation")
    return TrainingSupport(
        tuple(reversed(selected)), rows, minimum_rows, rows >= minimum_rows
    )


def recency_weights(timestamps: pd.Series, *, half_life_days: float) -> pd.Series:
    """Deterministic exponential weights, normalized to mean one."""
    if half_life_days <= 0:
        raise ValueError("Recency half-life must be positive")
    times = pd.to_datetime(timestamps)
    if times.empty or times.isna().any():
        raise ValueError("Recency timestamps must be nonempty and complete")
    ages = (times.max() - times).dt.total_seconds() / 86_400
    weights = 2.0 ** (-ages / half_life_days)
    return weights / weights.mean()
