"""Target-blind chronological support planning for the policy campaign."""

import pandas as pd

from kurversc.temporal_policy import (
    InsufficientPolicySupport,
    bounded_temporal_policies,
    policy_support,
)


def retrospective_folds(cutoffs, *, size, origins=3):
    if size < 1 or origins < 2:
        raise ValueError(
            "Positive block size and at least two requested origins required"
        )
    cuts = sorted(pd.to_datetime(list(cutoffs)).unique())
    count = min(origins, (len(cuts) - size - 1) // size)
    if count < 2:
        return []
    return [
        tuple(cuts[len(cuts) - number * size - size : len(cuts) - number * size])
        for number in reversed(range(count))
    ]


def required_policy_cutoffs(
    training, validation, *, timestamp, horizon, boundary_lag, task, origins=3
):
    """Union only the cohorts required by the common menu and holdouts.

    Use dates and cohort sizes, never target values or observed scores. Keep
    full source tables/features; this reduces redundant label-frame replay.
    """
    labels = pd.concat([training, validation], ignore_index=True)
    cuts = pd.to_datetime(labels[timestamp]).unique()
    size = pd.to_datetime(validation[timestamp]).nunique()
    folds = retrospective_folds(cuts, size=size, origins=origins)
    required = set()
    for heldout in folds:
        required.update(map(pd.Timestamp, heldout))
        origin = pd.Timestamp(heldout[0]) - pd.Timedelta(boundary_lag)
        for policy in bounded_temporal_policies(task):
            try:
                support = policy_support(
                    labels,
                    timestamp=timestamp,
                    origin=origin,
                    label_horizon=horizon,
                    refit_frames=size,
                    policy=policy,
                )
            except InsufficientPolicySupport:
                continue
            required.update(pd.Timestamp(cut) for cut in support["cutoffs"])
    training_cuts = set(map(pd.Timestamp, pd.to_datetime(training[timestamp]).unique()))
    return tuple(sorted(required & training_cuts)), folds
