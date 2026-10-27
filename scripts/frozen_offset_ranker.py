"""Research-only additive ranking correction to a frozen binary predictor.

The baseline must be entity-disjoint OOF on mature outer-training rows. Train
the correction with that baseline as a fixed offset, rather than learning a
replacement predictor or choosing a blend from official test outcomes.
"""

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from scripts.bounded_pairwise_readout import bounded_positive_pairs


def probability_logits(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Finite one-dimensional probabilities required")
    return logit(np.clip(values, 1e-6, 1 - 1e-6))


def cohort_pairs(labels, timestamps, *, maximum=100_000, seed=0):
    """Bounded positive/negative pairs only WITHIN the same forecast cohort."""
    labels = np.asarray(labels)
    times = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if len(labels) != len(times) or times.hasnans or not 1 <= maximum <= 100_000:
        raise ValueError("Aligned complete cohorts and a bounded pair budget required")
    cutoffs = sorted(times.unique())
    eligible = [np.flatnonzero(times == cutoff) for cutoff in cutoffs]
    eligible = [rows for rows in eligible if set(np.unique(labels[rows])) == {0, 1}]
    if not eligible:
        raise ValueError("At least one binary training cohort required")
    if maximum < len(eligible):
        raise ValueError("Pair budget must cover all eligible cohorts")
    pairs = []
    for number, rows in enumerate(eligible):
        budget = maximum // len(eligible) + int(number < maximum % len(eligible))
        local = bounded_positive_pairs(labels[rows], maximum=budget, seed=seed + number)
        pairs.append(rows[local])
    return np.concatenate(pairs)


def replace_strong_half(control, legacy, corrected_head):
    """Retain the exact legacy component of the accepted 50/50 ensemble."""
    control, legacy, corrected_head = [np.asarray(v, dtype=float) for v in (control, legacy, corrected_head)]
    for value in (control, legacy, corrected_head):
        probability_logits(value)
    if control.shape != legacy.shape or control.shape != corrected_head.shape:
        raise ValueError("Aligned protected component predictions required")
    head = 2 * control - legacy
    if np.any((head < -1e-12) | (head > 1 + 1e-12)):
        raise ValueError("Control does not contain the declared protected strong half")
    # Using a delta keeps exactly unchanged rows bit-identical to the incumbent.
    return control + .5 * (corrected_head - head)


class FrozenOffsetRanker:
    def __init__(self, *, seed=0, maximum_columns=64, iterations=240, thread_count=4):
        if not 1 <= maximum_columns <= 64 or not 1 <= iterations <= 240 or not 1 <= thread_count <= 8:
            raise ValueError("Bounded correction feature/tree/thread budgets required")
        self.seed = seed
        self.maximum_columns = maximum_columns
        self.iterations = iterations
        self.thread_count = thread_count

    def fit(self, features, labels, baseline, *, entities, timestamps, origin, horizon):
        from catboost import CatBoostRanker, Pool

        x = pd.DataFrame(features).reset_index(drop=True)
        y = np.asarray(labels, dtype=float)
        entities = np.asarray(entities)
        times = pd.DatetimeIndex(pd.to_datetime(timestamps))
        offset = probability_logits(baseline)
        if not len(x) == len(y) == len(offset) == len(entities) == len(times):
            raise ValueError("Aligned mature OOF training rows required")
        if set(np.unique(y)) != {0, 1} or times.hasnans or pd.isna(entities).any():
            raise ValueError("Complete binary training observations required")
        if pd.Timedelta(horizon) <= pd.Timedelta(0) or (times + pd.Timedelta(horizon) > pd.Timestamp(origin)).any():
            raise ValueError("Training labels must have matured by the fit boundary")
        if x.columns.duplicated().any() or {"__event_entity", "__event_cutoff"} & set(x):
            raise ValueError("Unique payload columns only; structural routing keys forbidden")
        if len(x.columns) > 512:
            raise ValueError("Bounded existing-state inputs required, not a full graph expansion")
        numeric = x.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
        residual = pd.Series(y - np.asarray(baseline)).rank(pct=True).to_numpy()
        residual -= residual.mean()
        residual_norm = np.linalg.norm(residual)
        utility = []
        for number, column in enumerate(numeric):
            values = numeric[column]
            if values.notna().mean() < .05 or values.nunique(dropna=True) < 2:
                continue
            ranked = values.rank(pct=True).fillna(.5).to_numpy()
            ranked -= ranked.mean()
            denominator = np.linalg.norm(ranked) * residual_norm
            score = abs(float(ranked @ residual / denominator)) if denominator else 0.
            utility.append((score, number, column))
        self.columns = [item[2] for item in sorted(utility, key=lambda v: (-v[0], v[1]))[:self.maximum_columns]]
        if not self.columns:
            raise ValueError("Reusable numeric training payload required")
        order = np.argsort(times.asi8, kind="stable")
        ordered_times = times[order]
        pairs = cohort_pairs(y[order], ordered_times, maximum=100_000, seed=self.seed + 61010)
        group = pd.factorize(ordered_times, sort=False)[0]
        pool = Pool(numeric.loc[order, self.columns].to_numpy(dtype="float32"), y[order],
            group_id=group, pairs=pairs, baseline=offset[order])
        self.model = CatBoostRanker(loss_function="PairLogit", iterations=self.iterations,
            depth=3, learning_rate=.03, l2_leaf_reg=30., random_seed=self.seed,
            thread_count=self.thread_count, verbose=False, allow_writing_files=False).fit(pool)
        self.audit = dict(training_rows=len(x), training_entities=len(np.unique(entities)),
            selected_columns=self.columns, maximum_columns=self.maximum_columns,
            training_only_column_utility=True, maximum_pairs=100_000, pairs=len(pairs),
            pair_scope="same forecast cutoff only", fixed_training_offset="entity-disjoint OOF baseline logit",
            latest_training_label_complete_at=str((times + pd.Timedelta(horizon)).max()),
            origin=str(origin), iterations=self.iterations, depth=3, thread_count=self.thread_count,
            seed=self.seed, structural_keys_are_predictors=False)
        return self

    def correction(self, features):
        raw = pd.DataFrame(features)[self.columns].replace([np.inf, -np.inf], np.nan)
        value = np.asarray(self.model.predict(raw.to_numpy(dtype="float32")), dtype=float)
        if not np.isfinite(value).all():
            raise ValueError("Finite learned ranking corrections required")
        return np.clip(value, -1., 1.)

    def predict_values(self, features, baseline, *, strength):
        if strength not in (.25, 1.):
            raise ValueError("Only the two predeclared correction strengths are allowed")
        raw = probability_logits(baseline)
        if len(raw) != len(features):
            raise ValueError("Aligned forecast features and baseline required")
        return expit(raw + strength * self.correction(features))
