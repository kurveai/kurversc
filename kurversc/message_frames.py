"""Bind a fitted downstream frame to its historical message encoder version."""

import copy

import pandas as pd


def freeze_message_family(family, training_cutoff):
    """Return an independent inference family with one eligible scorer per path.

    Relational input rows still use each query's own timestamp. Only supervised
    encoder parameters are frozen. Later labels/models cannot silently redefine
    feature coordinates for a downstream model fitted at ``training_cutoff``.
    The caller must save this family with the corresponding downstream model.
    """
    if not hasattr(family, "paths_"):
        raise ValueError("Fit the message family before freezing a frame")
    cutoff = pd.Timestamp(training_cutoff)
    if pd.isna(cutoff):
        raise ValueError("A nonmissing training cutoff is required")
    result = copy.deepcopy(family)
    for path, versions in result.models_.items():
        eligible = [v for v in versions if v[0] < cutoff]
        result.models_[path] = [max(eligible, key=lambda v: v[0])] if eligible else []
    result.fitted_frame_cutoff_ = cutoff
    return result
