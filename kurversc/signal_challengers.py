"""Opt-in, bounded post-promotion readouts; never change the graph defaults.

The caller supplies point-in-time features and mature training support. This
module has no benchmark names, test scorer, external model downloads, or target
driven schema discovery. Admission uses the existing paired temporal gate.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .temporal_policy import TemporalFitPolicy, policy_support


class ChallengerUnavailable(ValueError):
    """A bounded challenger is structurally inapplicable, not a failed baseline."""


@dataclass(frozen=True)
class SignalChallenger:
    name: str
    kind: str
    weight: float = 1.0
    training_frames: int | None = None
    regularized: bool = False

    def __post_init__(self):
        if self.kind not in {"text", "support"} or not 0 < self.weight <= 1:
            raise ValueError("A supported challenger and bounded weight are required")
        if self.training_frames is not None and self.training_frames not in {3, 5}:
            raise ValueError("Only the bounded 3/5-frame support challengers are supported")


def challenger_menu(*, text=True, exposure=False, support=True, exposure_blends=False):
    """Explicit opt-in menu, independent of the graph-configuration lattice."""
    if exposure or exposure_blends:
        raise ValueError("The exposure challenger has been removed from the submitted candidate menu")
    menu = []
    if text:
        menu += [SignalChallenger("text25", "text", .25),
                 SignalChallenger("text50", "text", .5)]
    if support:
        menu += [SignalChallenger("frame_ensemble3", "support", training_frames=3),
                 SignalChallenger("frame_ensemble5", "support", training_frames=5)]
    return tuple(menu)


def challenger_fit_key(challenger):
    """Identify raw fitted heads independently of their blend/recipe name."""
    return challenger.kind, challenger.training_frames, challenger.regularized


def apply_challenger_weight(challenger, incumbent, prediction):
    """Apply the declared weight once, identically in validation and inference.

    Exposure and support heads have the same weight contract as text heads.
    Full-weight legacy candidates retain their exact existing prediction array.
    """
    if challenger.weight == 1:
        a, b = np.asarray(incumbent), np.asarray(prediction)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Aligned finite challenger and incumbent predictions required")
        return prediction
    return combine_predictions(incumbent, prediction, challenger.weight)


def challenger_policy(base, challenger):
    if challenger.training_frames is None:
        return base
    # Extra support is a separate complete fitting procedure, not a silent
    # mutation of the incumbent. Faithful evaluation may reject it as unavailable.
    return TemporalFitPolicy(challenger.name, training_frames=challenger.training_frames)


def mature_support_mask(timestamps, *, origin, horizon, refit_frames, policy, available_at=None):
    times = pd.to_datetime(timestamps)
    frame = pd.DataFrame({"cutoff":times})
    if available_at is not None:
        frame["__kurversc_label_available_at"] = pd.to_datetime(available_at)
    support = policy_support(frame, timestamp="cutoff",
        origin=origin, label_horizon=horizon, refit_frames=refit_frames, policy=policy)
    mask = times.isin(pd.to_datetime(support["cutoffs"])).to_numpy()
    if available_at is not None:
        ready = pd.Series(pd.to_datetime(available_at),index=times.index)
        boundary = support["refit"][0] if support["refit"] else pd.Timestamp(origin)
        mask &= (ready.le(pd.Timestamp(origin)) & (~times.isin(support["initial"]) | ready.le(boundary))).to_numpy()
    if ((times[mask] + pd.Timedelta(horizon)) > pd.Timestamp(origin)).any():
        raise ValueError("Immature labels reached a signal readout")
    return mask, support


def select_text_columns(features, categorical, mask, *, root_prefix=None, maximum=8):
    candidates = []
    for name in sorted(set(categorical) & set(features.columns)):
        if root_prefix is not None and not name.startswith(root_prefix + "_"):
            continue
        values = features.loc[mask, name].astype("string").fillna("").replace("__missing__", "")
        lengths = values.str.len()
        positive = lengths[lengths > 0]
        if len(positive) and positive.mean() >= 32 and values.nunique() >= 20:
            candidates.append((-float(positive.mean()), name))
    return [name for _, name in sorted(candidates)[:maximum]]


def documents(features, columns):
    parts = [features[name].astype("string").fillna("").replace("__missing__", "").str.slice(0, 2048)
             for name in columns]
    if not parts:
        raise ChallengerUnavailable("No reusable parent text columns")
    return [" ".join(row) for row in zip(*(p.tolist() for p in parts), strict=True)]


class BoundedTextReadout:
    """Training-only sparse word readout, bounded to 8 fields / 16K words."""

    def __init__(self, *, task, random_state=0):
        if task not in {"classification", "regression"}:
            raise ValueError("Binary classification or numeric regression required")
        self.task, self.random_state = task, random_state

    def fit(self, features, labels, *, categorical, root_prefix=None, row_limit=100_000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, Ridge
        from .modeling import sample_training_rows

        if len(features) != len(labels) or row_limit < 1:
            raise ValueError("Aligned, bounded training examples required")
        self.columns = select_text_columns(features, categorical,
            np.ones(len(features), dtype=bool), root_prefix=root_prefix)
        if not self.columns:
            raise ChallengerUnavailable("No reusable parent text columns")
        x, y = sample_training_rows(features.loc[:, self.columns], labels,
            limit=row_limit, task=self.task, random_state=self.random_state)
        self.vectorizer = TfidfVectorizer(max_features=16384, ngram_range=(1, 2),
            min_df=2, max_df=.98, sublinear_tf=True, strip_accents="unicode", dtype=np.float32)
        try:
            matrix = self.vectorizer.fit_transform(documents(x, self.columns))
        except ValueError as exc:
            if "empty vocabulary" in str(exc) or "After pruning" in str(exc):
                raise ChallengerUnavailable(str(exc)) from exc
            raise
        if self.task == "classification":
            if set(pd.unique(y)) != {0, 1}:
                raise ChallengerUnavailable("Text head requires ordinary binary targets")
            self.model = LogisticRegression(C=1., solver="liblinear", max_iter=1000,
                random_state=self.random_state).fit(matrix, y.astype(int))
        else:
            self.model = Ridge(alpha=10., solver="lsqr").fit(matrix, y.astype(float))
        self.audit = dict(columns=self.columns, training_rows=len(x),
            vocabulary_size=len(self.vectorizer.vocabulary_), task=self.task,
            maximum_fields=8, characters_per_field=2048, row_limit=row_limit,
            scope="Vocabulary and readout fit on mature training examples only")
        return self

    def predict_values(self, features):
        x = self.vectorizer.transform(documents(features, self.columns))
        return (self.model.predict_proba(x)[:, 1] if self.task == "classification"
                else self.model.predict(x))


def combine_predictions(control, extra, weight):
    a, b = np.asarray(control), np.asarray(extra)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Aligned finite predictions required")
    if not 0 < weight <= 1:
        raise ValueError("Bounded blend weight required")
    return (1 - weight) * a + weight * b
