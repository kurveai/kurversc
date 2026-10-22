"""Research-only, bounded residual ranking of difficult existing-state cases.

Both feature selection and comparison mining use mature entity-disjoint OOF
training predictions, never validation labels. No graph/default changes.
"""

import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.bounded_pairwise_readout import bounded_positive_pairs
from scripts.frozen_offset_ranker import FrozenOffsetRanker, probability_logits


RECIPES = {
    "pair_gradient_uniform": {"utility": "pair_gradient", "comparisons": "uniform"},
    "residual_hard": {"utility": "point_residual", "comparisons": "mixed_hard"},
    "pair_gradient_hard": {"utility": "pair_gradient", "comparisons": "mixed_hard"},
}


def comparison_pairs(labels, timestamps, baseline, *, comparisons, maximum=100_000, seed=0):
    """Half uniform/half nearby-logit comparisons, with a strict global cap."""
    y = np.asarray(labels)
    times = pd.DatetimeIndex(pd.to_datetime(timestamps))
    logits = probability_logits(baseline)
    if (y.ndim != 1 or not len(y) == len(times) == len(logits) or times.hasnans
            or set(np.unique(y)) != {0, 1} or not 1 <= maximum <= 100_000
            or comparisons not in {"uniform", "mixed_hard"}):
        raise ValueError("Complete aligned binary cohorts and bounded comparisons required")
    cohorts = [np.flatnonzero(times == cutoff) for cutoff in sorted(times.unique())]
    cohorts = [rows for rows in cohorts if set(np.unique(y[rows])) == {0, 1}]
    if maximum < len(cohorts):
        raise ValueError("Pair budget must cover every eligible cohort")
    results = []
    for number, rows in enumerate(cohorts):
        budget = maximum // len(cohorts) + int(number < maximum % len(cohorts))
        local_y, local_z = y[rows], logits[rows]
        pos, neg = np.flatnonzero(local_y == 1), np.flatnonzero(local_y == 0)
        total = len(pos) * len(neg)
        if comparisons == "uniform" or total <= budget:
            local = bounded_positive_pairs(local_y, maximum=budget, seed=seed + number)
        else:
            rng = np.random.default_rng(seed + number)
            uniform = bounded_positive_pairs(local_y, maximum=budget, seed=seed + number)
            fixed = uniform[:budget // 2]
            neg = neg[np.argsort(local_z[neg], kind="stable")]
            # Select competitors within one fixed log-odds unit. In an empty
            # interval use the nearest score; do not peek at held-out errors.
            left = np.searchsorted(local_z[neg], local_z[pos] - 1., side="left")
            right = np.searchsorted(local_z[neg], local_z[pos] + 1., side="right")
            nearest = np.clip(np.searchsorted(local_z[neg], local_z[pos]), 0, len(neg) - 1)
            empty = left == right
            left[empty], right[empty] = nearest[empty], nearest[empty] + 1
            samples = []
            for _ in range(4):
                chosen = rng.integers(len(pos), size=budget)
                rival = left[chosen] + (rng.random(budget) * (right[chosen] - left[chosen])).astype(int)
                samples.append(np.column_stack((pos[chosen], neg[rival])))
            mined = np.concatenate(samples)
            # First-occurrence de-duplication preserves the declared 50% uniform
            # prefix; uniform fallback fills any shortage of unique hard pairs.
            proposed = np.concatenate((fixed, mined, uniform))
            _, unique = np.unique(proposed, axis=0, return_index=True)
            local = proposed[np.sort(unique)[:budget]]
        results.append(rows[local])
    pairs = np.concatenate(results)
    if len(pairs) > maximum or not np.array_equal(times[pairs[:, 0]], times[pairs[:, 1]]):
        raise ValueError("Comparison budget/time scope violated")
    return pairs


def comparison_gradient(baseline, pairs):
    """Negative gradient of PairLogit at the frozen training-only offset."""
    z = probability_logits(baseline)
    pairs = np.asarray(pairs, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or np.any((pairs < 0) | (pairs >= len(z))):
        raise ValueError("Valid bounded comparison indices required")
    error = expit(z[pairs[:, 1]] - z[pairs[:, 0]])
    return (np.bincount(pairs[:, 0], weights=error, minlength=len(z))
            - np.bincount(pairs[:, 1], weights=error, minlength=len(z)))


class HardComparisonOffsetRanker(FrozenOffsetRanker):
    def __init__(self, *, utility, comparisons, **kwargs):
        super().__init__(**kwargs)
        if utility not in {"point_residual", "pair_gradient"} or comparisons not in {"uniform", "mixed_hard"}:
            raise ValueError("Only predeclared training-only objectives allowed")
        self.utility, self.comparisons = utility, comparisons

    def fit(self, features, labels, baseline, *, entities, timestamps, origin, horizon):
        from catboost import CatBoostRanker, Pool

        x = pd.DataFrame(features).reset_index(drop=True)
        y, entities = np.asarray(labels, dtype=float), np.asarray(entities)
        times = pd.DatetimeIndex(pd.to_datetime(timestamps))
        offset = probability_logits(baseline)
        if not len(x) == len(y) == len(offset) == len(entities) == len(times):
            raise ValueError("Aligned mature entity-disjoint OOF training rows required")
        if times.hasnans or pd.isna(entities).any() or set(np.unique(y)) != {0, 1}:
            raise ValueError("Complete binary training observations required")
        if pd.Timedelta(horizon) <= pd.Timedelta(0) or (times + pd.Timedelta(horizon) > pd.Timestamp(origin)).any():
            raise ValueError("Training labels must have matured by fit origin")
        if x.columns.duplicated().any() or {"__event_entity", "__event_cutoff"} & set(x) or len(x.columns) > 512:
            raise ValueError("Bounded unique payload columns; structural keys forbidden")
        order = np.argsort(times.asi8, kind="stable")
        pairs = comparison_pairs(y[order], times[order], np.asarray(baseline)[order],
            comparisons=self.comparisons, maximum=100_000, seed=self.seed + 61010)
        if self.utility == "pair_gradient":
            residual = np.empty(len(y))
            residual[order] = comparison_gradient(np.asarray(baseline)[order], pairs)
        else:
            residual = pd.Series(y - np.asarray(baseline)).rank(pct=True).to_numpy()
        residual -= residual.mean()
        norm = np.linalg.norm(residual)
        numeric = x.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
        utilities = []
        for number, column in enumerate(numeric):
            values = numeric[column]
            if values.notna().mean() < .05 or values.nunique(dropna=True) < 2:
                continue
            ranked = values.rank(pct=True).fillna(.5).to_numpy()
            ranked -= ranked.mean()
            denominator = np.linalg.norm(ranked) * norm
            score = abs(float(ranked @ residual / denominator)) if denominator else 0.
            utilities.append((score, number, column))
        ranked = sorted(utilities, key=lambda item: (-item[0], item[1]))
        self.columns = [item[2] for item in ranked[:self.maximum_columns]]
        if not self.columns:
            raise ValueError("Reusable numeric payload required")
        pool = Pool(numeric.loc[order, self.columns].to_numpy(dtype="float32"), y[order],
            group_id=pd.factorize(times[order], sort=False)[0], pairs=pairs, baseline=offset[order])
        self.model = CatBoostRanker(loss_function="PairLogit", iterations=self.iterations,
            depth=3, learning_rate=.03, l2_leaf_reg=30., random_seed=self.seed,
            thread_count=self.thread_count, verbose=False, allow_writing_files=False).fit(pool)
        margins = offset[order][pairs[:, 0]] - offset[order][pairs[:, 1]]
        self.audit = dict(training_rows=len(x), training_entities=len(np.unique(entities)),
            selected_columns=self.columns, selected_utilities={c:float(u) for u, _, c in ranked[:self.maximum_columns]},
            maximum_columns=self.maximum_columns, utility=self.utility, comparisons=self.comparisons,
            training_only_selection=True, maximum_pairs=100_000, pairs=len(pairs),
            pair_scope="same forecast cutoff only", pair_incorrect_fraction=float((margins < 0).mean()),
            pair_nearby_fraction=float((abs(margins) <= 1).mean()), fixed_offset="entity-disjoint OOF logit",
            latest_training_label_complete_at=str((times + pd.Timedelta(horizon)).max()),
            origin=str(origin), iterations=self.iterations, depth=3, thread_count=self.thread_count,
            seed=self.seed, structural_keys_are_predictors=False)
        return self
