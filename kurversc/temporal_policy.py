"""Bounded post-promotion selection of complete temporal fitting procedures.

This module has no task/schema names or access to an official test scorer.
Every policy includes the observed-frame refit used at deployment. Forecast
origins, label maturity and frozen feature schemas are supplied by the caller.
The incumbent is retained unless paired temporal evidence supports a change.
"""

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from .modeling import (
    _binary_classes,
    FrameEnsemble,
    fit_final_catboost,
    sample_training_rows,
)
from .ordered_regression import OrderedTargetRegressor
from .training_support import select_training_support


@dataclass(frozen=True)
class TemporalFitPolicy:
    name: str
    mode: str = "frame_ensemble"
    readout: str = "direct"
    training_frames: int = 1
    minimum_training_rows: int = 1
    max_training_frames: int = 64
    blend_weight: float = 1.0

    def __post_init__(self):
        if self.mode not in {"frame_ensemble", "pooled"}:
            raise ValueError("Unknown temporal estimator mode")
        if self.readout not in {"direct", "ordered_median"}:
            raise ValueError("Unknown temporal readout")
        if not self.name or not 1 <= self.training_frames <= self.max_training_frames:
            raise ValueError("Named policy and bounded positive frame counts required")
        if self.minimum_training_rows < 1:
            raise ValueError("Positive support floor required")
        if not 0 < self.blend_weight <= 1:
            raise ValueError("Challenger blend weight must be in (0, 1]")


class InsufficientPolicySupport(ValueError):
    """A policy cannot be evaluated faithfully with the supplied history."""


class TemporalPredictionBlend:
    """Serializable, validation-fixed blend; inference never reads labels."""

    def __init__(self, incumbent, challenger, *, weight: float, task: str):
        if not 0 < weight < 1 or task not in {"classification", "regression"}:
            raise ValueError("Interior weight and supported task required")
        self.incumbent, self.challenger = incumbent, challenger
        self.weight, self.task = weight, task

    def predict_proba(self, x):
        if self.task != "classification":
            raise AttributeError("Regression blends do not predict probabilities")
        return (1 - self.weight) * self.incumbent.predict_proba(
            x
        ) + self.weight * self.challenger.predict_proba(x)

    def predict(self, x):
        if self.task == "classification":
            return (self.predict_proba(x)[:, 1] >= 0.5).astype("int8")
        return (1 - self.weight) * self.incumbent.predict(
            x
        ) + self.weight * self.challenger.predict(x)


def add_incumbent_blends(evidence: list[dict], *, weights: tuple[float, ...]):
    """Add a bounded weight grid using fitted paired predictions, no new fits."""
    if (
        len(weights) > 4
        or len(set(weights)) != len(weights)
        or any(not 0 < w < 1 for w in weights)
    ):
        raise ValueError("At most four distinct interior blend weights required")
    controls = {
        (r["origin"], r["seed"]): r for r in evidence if r["policy"] == "incumbent"
    }
    expanded, recipes = list(evidence), {}
    for row in evidence:
        if row["policy"] == "incumbent":
            continue
        control = controls.get((row["origin"], row["seed"]))
        if control is None:
            continue
        if not np.array_equal(row["truth"], control["truth"]) or not np.array_equal(
            row["groups"], control["groups"]
        ):
            raise ValueError("Blended policies must score identical ordered rows")
        for weight in weights:
            name = f"{row['policy']}__blend_{weight:g}"
            recipes[name] = dict(base_policy=row["policy"], blend_weight=weight)
            expanded.append(
                {
                    **row,
                    "policy": name,
                    "prediction": (1 - weight) * np.asarray(control["prediction"])
                    + weight * np.asarray(row["prediction"]),
                }
            )
    return expanded, recipes


def bounded_temporal_policies(task: str) -> tuple[TemporalFitPolicy, ...]:
    """Small generic post-promotion menu, not another graph search dimension."""
    if task not in {"classification", "regression"}:
        raise ValueError("Binary classification or numeric regression required")
    policies = [
        TemporalFitPolicy("incumbent"),
        TemporalFitPolicy("pooled12", mode="pooled", training_frames=12),
        TemporalFitPolicy("floor1000", mode="pooled", minimum_training_rows=1000),
    ]
    if task == "regression":
        policies.extend(
            [
                TemporalFitPolicy(
                    "ordered_same_support", mode="pooled", readout="ordered_median"
                ),
                TemporalFitPolicy(
                    "ordered12",
                    mode="pooled",
                    readout="ordered_median",
                    training_frames=12,
                ),
            ]
        )
    return tuple(policies)


def policy_support(
    frame: pd.DataFrame,
    *,
    timestamp: str,
    origin,
    label_horizon,
    refit_frames: int,
    policy: TemporalFitPolicy,
) -> dict[str, Any]:
    """Reproduce initial support + observed validation refit at any origin.

    The last ``refit_frames`` mature cutoffs represent the known validation
    cohorts. Initial support must already be mature at THEIR first cutoff,
    exactly as in the train/validation/production protocol. Heldout origins
    never enter training; no target values enter this support decision.
    """
    if refit_frames < 0:
        raise ValueError("Nonnegative observed refit frame count required")
    times = pd.to_datetime(frame[timestamp])
    ready = (pd.to_datetime(frame["__kurversc_label_available_at"])
        if "__kurversc_label_available_at" in frame else times+pd.Timedelta(label_horizon))
    if times.isna().any() or pd.Timedelta(label_horizon) < pd.Timedelta(0):
        raise ValueError("Complete timestamps and nonnegative horizon required")
    eligible = tuple(
        sorted(
            times.loc[
                (times < pd.Timestamp(origin))
                & ready.le(pd.Timestamp(origin))
            ].unique()
        )
    )
    if len(eligible) <= refit_frames:
        raise InsufficientPolicySupport(
            "No initial training frame before observed refit"
        )
    refit = eligible[-refit_frames:] if refit_frames else ()
    initial_boundary = refit[0] if refit else pd.Timestamp(origin)
    initial_frame = frame.loc[(times < pd.Timestamp(initial_boundary)) & ready.le(pd.Timestamp(initial_boundary))]
    try:
        initial = select_training_support(
            initial_frame,
            timestamp=timestamp,
            evaluation_cutoff=initial_boundary,
            label_horizon=label_horizon,
            minimum_rows=policy.minimum_training_rows,
            minimum_frames=policy.training_frames,
            max_frames=policy.max_training_frames,
        )
    except ValueError as exc:
        raise InsufficientPolicySupport(str(exc)) from exc
    if not initial.complete or len(initial.cutoffs) < policy.training_frames:
        raise InsufficientPolicySupport("Initial support budget is not available")
    return dict(
        initial=initial.cutoffs,
        refit=tuple(map(pd.Timestamp, refit)),
        cutoffs=(*initial.cutoffs, *map(pd.Timestamp, refit)),
    )


def fit_temporal_policy(
    features: pd.DataFrame,
    labels: pd.Series,
    timestamps: pd.Series,
    *,
    origin,
    label_horizon,
    refit_frames: int,
    policy: TemporalFitPolicy,
    task: str,
    categorical: list[int],
    random_state: int = 0,
    row_limit: int | None = 100_000,
    model_params: dict | None = None,
    frame_model_cache: dict | None = None,
    incumbent_policy: TemporalFitPolicy | None = None,
    available_at=None,
) -> tuple[Any, dict]:
    """Fit a complete, serializable policy with no test/holdout targets.

    Frozen features must be materialized at each row's original cutoff. The
    optional frame-model cache is local to one immutable data/schema/parameter
    context, never shared between tasks or a changed training population.
    """
    from .core import _sample_temporal_frame

    if not (len(features) == len(labels) == len(timestamps)):
        raise ValueError("Aligned feature, target and cutoff rows required")
    if task not in {"classification", "regression"}:
        raise ValueError("Binary classification or regression required")
    if policy.readout != "direct" and task != "regression":
        raise ValueError("Ordered median is a numeric regression readout")
    if policy.blend_weight < 1:
        incumbent = incumbent_policy or TemporalFitPolicy("incumbent")
        if incumbent.blend_weight != 1:
            raise ValueError("The incumbent cannot itself be a nested blend")
        kwargs = dict(
            origin=origin,
            label_horizon=label_horizon,
            refit_frames=refit_frames,
            task=task,
            categorical=categorical,
            random_state=random_state,
            row_limit=row_limit,
            model_params=model_params,
            frame_model_cache=frame_model_cache,
            available_at=available_at,
        )
        base, base_audit = fit_temporal_policy(
            features, labels, timestamps, policy=incumbent, **kwargs
        )
        challenger, audit = fit_temporal_policy(
            features,
            labels,
            timestamps,
            policy=replace(policy, blend_weight=1.0),
            **kwargs,
        )
        model = TemporalPredictionBlend(
            base, challenger, weight=policy.blend_weight, task=task
        )
        return model, dict(
            policy=asdict(policy), incumbent=base_audit, challenger=audit
        )
    x, y = features.reset_index(drop=True), labels.reset_index(drop=True)
    times = pd.to_datetime(timestamps).reset_index(drop=True)
    available_at = available_at if available_at is not None else features.attrs.get("kurversc_label_available_at")
    ready = pd.Series(pd.to_datetime(available_at)).reset_index(drop=True) if available_at is not None else None
    support_frame = pd.DataFrame({"cutoff":times})
    if ready is not None:
        if len(ready)!=len(times) or ready.isna().any() or (ready<times+pd.Timedelta(label_horizon)).any():
            raise ValueError("Complete aligned label availability after horizon completion is required")
        support_frame["__kurversc_label_available_at"] = ready
    support = policy_support(
        support_frame,
        timestamp="cutoff",
        origin=origin,
        label_horizon=label_horizon,
        refit_frames=refit_frames,
        policy=policy,
    )
    params = dict(model_params or {})

    def cohort_mask(cutoff):
        mask = times.eq(cutoff)
        if ready is not None:
            boundary = support["refit"][0] if cutoff in support["initial"] and support["refit"] else pd.Timestamp(origin)
            mask &= ready.le(pd.Timestamp(boundary))
        return mask

    def fit_one(tx, ty):
        if policy.readout == "ordered_median":
            allowed = {
                "iterations",
                "depth",
                "learning_rate",
                "l2_leaf_reg",
                "thread_count",
            }
            if set(params) - allowed:
                raise ValueError("Unsupported ordered-readout model parameters")
            model = OrderedTargetRegressor(
                l2_leaf_reg=params.get("l2_leaf_reg", 10.0),
                random_state=random_state,
                **{k: v for k, v in params.items() if k != "l2_leaf_reg"},
            )
            return model.fit(tx, ty, cat_features=categorical)
        return fit_final_catboost(
            tx,
            ty,
            task=task,
            categorical=categorical,
            random_state=random_state,
            model_params=params,
        )[0]

    models, skipped, fitted_rows = [], [], 0
    if policy.mode == "frame_ensemble":
        for number, cutoff in enumerate(support["cutoffs"], start=1):
            mask = cohort_mask(cutoff)
            tx, ty = sample_training_rows(
                x.loc[mask],
                y.loc[mask],
                limit=row_limit,
                task=task,
                random_state=random_state + number,
            )
            if task == "classification" and ty.nunique() < 2:
                skipped.append(str(cutoff))
                continue
            key = (str(cutoff), random_state, number, policy.readout, row_limit)
            if ready is not None:
                # Availability may change membership at a later phase even
                # for the same cohort. Never reuse a partial-cohort model.
                key = (*key,str(origin),tuple(np.flatnonzero(mask.to_numpy())))
            cached = (
                frame_model_cache.get(key) if frame_model_cache is not None else None
            )
            model = cached if cached is not None else fit_one(tx, ty)
            if frame_model_cache is not None:
                frame_model_cache[key] = model
            models.append(model)
            fitted_rows += len(tx)
        if not models:
            raise InsufficientPolicySupport("No trainable incumbent frame")
        model = FrameEnsemble(models, task=task)
    else:
        # Match the original production lifecycle: bounded initial support,
        # separately bounded refit cohorts, then the final global row ceiling.
        parts_x, parts_y = [], []
        for group_number, cuts in enumerate((support["initial"], support["refit"])):
            group_x, group_y = [], []
            for number, cutoff in enumerate(cuts):
                mask = cohort_mask(cutoff)
                tx, ty = _sample_temporal_frame(
                    x.loc[mask],
                    y.loc[mask],
                    total_limit=row_limit,
                    frame_count=len(cuts),
                    frame_index=number,
                    task=task,
                    random_state=random_state
                    + (len(support["initial"]) if group_number else 0),
                )
                group_x.append(tx.reset_index(drop=True))
                group_y.append(ty.reset_index(drop=True))
            if group_x:
                gx, gy = (
                    pd.concat(group_x, ignore_index=True),
                    pd.concat(group_y, ignore_index=True),
                )
                if not group_number:
                    gx, gy = sample_training_rows(
                        gx, gy, limit=row_limit, task=task, random_state=random_state
                    )
                parts_x.append(gx)
                parts_y.append(gy)
        tx, ty = sample_training_rows(
            pd.concat(parts_x, ignore_index=True),
            pd.concat(parts_y, ignore_index=True),
            limit=row_limit,
            task=task,
            random_state=random_state,
        )
        if task == "classification" and ty.nunique() < 2:
            raise InsufficientPolicySupport("Pooled support lacks both target classes")
        model = fit_one(tx, ty)
        fitted_rows = len(tx)
    audit = dict(
        policy=asdict(policy),
        origin=str(pd.Timestamp(origin)),
        initial_cutoffs=list(map(str, support["initial"])),
        observed_refit_cutoffs=list(map(str, support["refit"])),
        training_rows=fitted_rows,
        skipped_single_class_cutoffs=skipped,
    )
    return model, audit


def _score(y, prediction, task):
    if task == "classification":
        labels = pd.Series(np.asarray(y))
        classes = _binary_classes(labels)
        if len(classes) != 2:
            raise ValueError("Binary validation requires both observed classes")
        y = labels.map({label: index for index, label in enumerate(classes)}).to_numpy()
    return float(
        roc_auc_score(y, prediction)
        if task == "classification"
        else mean_absolute_error(y, prediction)
    )


def _temporal_gain_lower_bound(pairs, *, task, alpha, repeats):
    """Recent-weighted paired gain, clustered by entity across ALL origins."""
    codes, unique = pd.factorize(np.concatenate([p[3] for p in pairs]), sort=True)
    if len(unique) < 2:
        return None
    split_codes, position = [], 0
    for truth, _, _, _ in pairs:
        split_codes.append(codes[position : position + len(truth)])
        position += len(truth)
    ordered_pairs, ordered_codes = [], []
    for (truth, control, candidate, groups), ids in zip(pairs, split_codes):
        # Fix weighted reduction order as well as cluster enumeration. This
        # keeps even floating-point confidence bounds row-order invariant.
        order = np.lexsort((candidate, control, truth, ids))
        ordered_pairs.append(
            (truth[order], control[order], candidate[order], groups[order])
        )
        ordered_codes.append(ids[order])
    weights = np.full(len(pairs), 0.5 / max(1, len(pairs) - 1))
    weights[-1] = 0.5
    rng, gains = np.random.default_rng(43), []
    for _ in range(repeats):
        # One multiplier per entity shared across origins preserves repeated
        # entity dependence; no seed averaging is used in these predictions.
        multipliers = rng.poisson(1.0, len(unique))
        fold_gains = []
        for (truth, control, candidate, _), ids in zip(ordered_pairs, ordered_codes):
            sample_weight = multipliers[ids]
            if not sample_weight.sum() or (
                task == "classification"
                and np.unique(truth[sample_weight > 0]).size < 2
            ):
                break
            metric = roc_auc_score if task == "classification" else mean_absolute_error
            a = metric(truth, control, sample_weight=sample_weight)
            b = metric(truth, candidate, sample_weight=sample_weight)
            fold_gains.append(b - a if task == "classification" else a - b)
        if len(fold_gains) == len(pairs):
            gains.append(float(weights @ fold_gains))
    return float(np.quantile(gains, alpha)) if gains else None


def select_temporal_policy(
    evidence: list[dict],
    *,
    task: str,
    incumbent: str = "incumbent",
    alpha: float = 0.05,
    bootstrap_repeats: int = 600,
    minimum_origins: int = 2,
    minimum_seeds: int = 2,
    classification_loss_cap: float = 0.005,
    regression_relative_loss_cap: float = 0.01,
    deployment_seed: int = 0,
) -> dict:
    """Recent-origin paired improvement, older-origin veto, safe fallback.

    Evidence contains only retrospective predictions: origin, seed, policy,
    truth, prediction, and entity ``groups``. Recent intervals resample entity
    clusters; repeated entities aren't treated as independent observations.
    Confidence uses a recent-weighted paired gain over all origins, sharing
    entity multipliers across time. The deployed seed's recent gain must be
    positive and every other seed must agree; older origins provide a veto.
    Bonferroni controls the tested challenger menu, not graph search or future
    distribution shift. These are safeguards, not a no-regression guarantee.
    """
    if task not in {"classification", "regression"} or not 0 < alpha < 1:
        raise ValueError("Valid task and confidence level required")
    if min(bootstrap_repeats, minimum_seeds) < 1 or minimum_origins < 2:
        raise ValueError("Positive budgets and at least two temporal origins required")
    rows = {}
    for row in evidence:
        key = (pd.Timestamp(row["origin"]), int(row["seed"]), row["policy"])
        if key in rows:
            raise ValueError("Duplicate origin/seed/policy evidence")
        truth, prediction, groups = [
            np.asarray(row[name]) for name in ("truth", "prediction", "groups")
        ]
        if (
            truth.ndim != 1
            or truth.shape != prediction.shape
            or truth.shape != groups.shape
            or not len(truth)
        ):
            raise ValueError("Aligned nonempty one-dimensional evidence required")
        if task == "classification":
            if pd.isna(truth).any():
                raise ValueError("Complete classification targets required")
            classes = _binary_classes(pd.Series(truth))
            if len(classes) != 2:
                raise ValueError("Binary validation requires both observed classes")
            truth = (
                pd.Series(truth)
                .map({label: index for index, label in enumerate(classes)})
                .to_numpy()
            )
        if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
            raise ValueError("Finite evidence required")
        if pd.isna(groups).any():
            raise ValueError("Complete entity groups required for paired uncertainty")
        rows[key] = {**row, "truth": truth}
    origins = sorted({key[0] for key in rows if key[2] == incumbent})
    names = sorted({key[2] for key in rows} - {incumbent})
    decision = dict(
        selected=incumbent, reason="insufficient_temporal_evidence", candidates={}
    )
    if len(origins) < minimum_origins:
        return decision
    eligible = []
    for name in names:
        fold_results, paired_folds, reason = [], [], None
        for origin in origins:
            seeds = sorted(
                {key[1] for key in rows if key[0] == origin and key[2] == incumbent}
            )
            if (
                deployment_seed not in seeds
                or len(seeds) < minimum_seeds
                or any((origin, seed, name) not in rows for seed in seeds)
            ):
                reason = "missing_matched_origin_or_seed"
                break
            reference = rows[(origin, seeds[0], incumbent)]
            y, groups = np.asarray(reference["truth"]), np.asarray(reference["groups"])
            cp, pp, seed_gains = [], [], []
            for seed in seeds:
                a, b = rows[(origin, seed, incumbent)], rows[(origin, seed, name)]
                for item in (a, b):
                    if not np.array_equal(item["truth"], y) or not np.array_equal(
                        item["groups"], groups
                    ):
                        raise ValueError(
                            "Paired policies must score identical ordered rows"
                        )
                av, bv = (
                    _score(y, a["prediction"], task),
                    _score(y, b["prediction"], task),
                )
                seed_gains.append(bv - av if task == "classification" else av - bv)
                cp.append(a["prediction"])
                pp.append(b["prediction"])
            # Production is one fixed seed, NOT a seed ensemble. Other seeds
            # test stability; do not give validation an unshipped ensemble.
            primary = seeds.index(deployment_seed)
            control, candidate = np.asarray(cp[primary]), np.asarray(pp[primary])
            paired_folds.append((y, control, candidate, groups))
            control_score, candidate_score = (
                _score(y, control, task),
                _score(y, candidate, task),
            )
            gain = (
                candidate_score - control_score
                if task == "classification"
                else control_score - candidate_score
            )
            cap = (
                classification_loss_cap
                if task == "classification"
                else regression_relative_loss_cap * control_score
            )
            result = dict(
                origin=str(origin),
                control=control_score,
                candidate=candidate_score,
                gain=gain,
                practical_loss_cap=cap,
                seed_gains=seed_gains,
                deployment_seed=deployment_seed,
            )
            if origin == origins[-1]:
                codes, unique = pd.factorize(groups, sort=True)
                blocks = [
                    np.flatnonzero(codes == index) for index in range(len(unique))
                ]
                rng, gains = np.random.default_rng(42), []
                if len(blocks) >= 2:
                    for _ in range(bootstrap_repeats):
                        pos = np.concatenate(
                            [
                                blocks[i]
                                for i in rng.integers(0, len(blocks), len(blocks))
                            ]
                        )
                        if task == "classification" and np.unique(y[pos]).size < 2:
                            continue
                        av, bv = (
                            _score(y[pos], control[pos], task),
                            _score(y[pos], candidate[pos], task),
                        )
                        gains.append(bv - av if task == "classification" else av - bv)
                result["recent_lower_bound"] = (
                    float(np.quantile(gains, alpha / max(1, len(names))))
                    if gains
                    else None
                )
            fold_results.append(result)
        if reason is None:
            latest, older = fold_results[-1], fold_results[:-1]
            if latest["gain"] <= 0:
                reason = "no_recent_improvement"
            elif min(latest["seed_gains"]) <= 0:
                reason = "recent_gain_not_seed_stable"
            elif any(r["gain"] < -r["practical_loss_cap"] for r in older):
                reason = "older_origin_regression"
            else:
                lower = _temporal_gain_lower_bound(
                    paired_folds,
                    task=task,
                    alpha=alpha / max(1, len(names)),
                    repeats=bootstrap_repeats,
                )
                latest["temporal_gain_lower_bound"] = lower
                if lower is None or lower <= 0:
                    reason = "no_confident_temporal_improvement"
                else:
                    reason = "eligible"
                    eligible.append((latest["gain"], name))
        decision["candidates"][name] = dict(reason=reason, folds=fold_results)
    if eligible:
        _, decision["selected"] = sorted(
            eligible, key=lambda item: (-item[0], item[1])
        )[0]
        decision["reason"] = "paired_recent_gain_with_temporal_veto"
    else:
        decision["reason"] = "incumbent_retained"
    decision["scope"] = (
        "Retrospective validation only; no guarantee under future distribution shift"
    )
    return decision


@dataclass
class SelectedTemporalEstimator:
    estimator: Any
    policy: TemporalFitPolicy
    selection_audit: dict
    fit_audit: dict


@dataclass
class TemporalHoldoutSnapshot:
    """Provider-declared phase-frozen features, indexed by original row position."""

    features: pd.DataFrame
    database_cutoff: Any
    observed_labels_complete_at: Any


def select_and_fit_temporal_policy(
    features: pd.DataFrame,
    labels: pd.Series,
    timestamps: pd.Series,
    entities: pd.Series,
    *,
    forecast_origin,
    label_horizon,
    refit_frames: int,
    task: str,
    categorical: list[int],
    seeds: tuple[int, ...] = (0, 1, 2),
    random_state: int = 0,
    max_origins: int = 3,
    holdout_frames: int | None = None,
    row_limit: int | None = 100_000,
    model_params: dict | None = None,
    policies: tuple[TemporalFitPolicy, ...] | None = None,
    on_evaluation=None,
    heldout_feature_provider=None,
    blend_weights: tuple[float, ...] = (),
) -> SelectedTemporalEstimator:
    """Standalone schema-generic downstream selection and final refit.

    Supply frozen, point-in-time features and observed labels after graph
    promotion. The same policy fitter is used in every retrospective fold and
    in final production. No agent, benchmark-specific lookup or test score is
    needed. Default graph search and existing ``kurversc.fit`` are unchanged.

    This is an opt-in experimental stage, not a statistical promise that future
    performance cannot regress. The caller may supply its own incumbent recipe
    (named ``incumbent``); the default incumbent is the frame-ensemble control.
    Multi-cutoff holdouts require a provider that freezes BOTH the database and
    observed-label providers at the phase start, not at each later row date.
    The provider receives (origin, ordered original row positions) and returns
    TemporalHoldoutSnapshot. It may drop dangling entities, not add rows.
    """
    if not (len(features) == len(labels) == len(timestamps) == len(entities)):
        raise ValueError(
            "Aligned frozen features, labels, cutoffs and entities required"
        )
    if max_origins < 2 or len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("At least two origins and distinct seeds required")
    add_incumbent_blends([], weights=blend_weights)  # Validate before expensive fits.
    holdout_frames = max(1, refit_frames) if holdout_frames is None else holdout_frames
    if holdout_frames < 1 or refit_frames < 0:
        raise ValueError(
            "Positive holdout and nonnegative refit frame budgets required"
        )
    if holdout_frames > 1 and heldout_feature_provider is None:
        raise ValueError("Multi-cutoff holdouts require phase-frozen feature snapshots")
    times = pd.to_datetime(timestamps).reset_index(drop=True)
    if times.isna().any() or pd.Timedelta(label_horizon) < pd.Timedelta(0):
        raise ValueError("Complete cutoff timestamps and nonnegative horizon required")
    # Future/uncompleted labels are dropped before any fold or seed is scored.
    known = (times < pd.Timestamp(forecast_origin)) & (
        times + pd.Timedelta(label_horizon)
    ).le(pd.Timestamp(forecast_origin))
    original_positions = np.flatnonzero(known.to_numpy())
    x = features.reset_index(drop=True).loc[known].reset_index(drop=True)
    y = labels.reset_index(drop=True).loc[known].reset_index(drop=True)
    groups = entities.reset_index(drop=True).loc[known].reset_index(drop=True)
    times = times.loc[known].reset_index(drop=True)
    if times.empty or times.isna().any() or groups.isna().any():
        raise ValueError("Complete nonempty observed support required")
    policies = bounded_temporal_policies(task) if policies is None else tuple(policies)
    if len({p.name for p in policies}) != len(policies) or not any(
        p.name == "incumbent" for p in policies
    ):
        raise ValueError("Unique policies and an explicit incumbent required")
    cutoffs = sorted(times.unique())
    fold_count = min(max_origins, (len(cutoffs) - refit_frames - 1) // holdout_frames)
    folds = []
    for number in reversed(range(max(0, fold_count))):
        end = len(cutoffs) - number * holdout_frames
        folds.append(tuple(cutoffs[end - holdout_frames : end]))
    evidence, records = [], []
    if len(folds) >= 2:
        for seed in seeds:
            cache = {}
            for cuts in folds:
                origin = pd.Timestamp(cuts[0])
                holdout = times.isin(cuts)
                positions = np.flatnonzero(holdout.to_numpy())
                vx = x.iloc[positions]
                fit_origin = origin
                if heldout_feature_provider is not None:
                    snapshot = heldout_feature_provider(
                        origin, original_positions[positions]
                    )
                    if not isinstance(snapshot, TemporalHoldoutSnapshot):
                        raise ValueError(
                            "Phase provider must declare its snapshot boundaries"
                        )
                    if pd.isna(pd.Timestamp(snapshot.database_cutoff)) or pd.isna(
                        pd.Timestamp(snapshot.observed_labels_complete_at)
                    ):
                        raise ValueError("Complete phase snapshot boundaries required")
                    if pd.Timestamp(snapshot.database_cutoff) > origin or pd.Timestamp(
                        snapshot.observed_labels_complete_at
                    ) > pd.Timestamp(snapshot.database_cutoff):
                        raise ValueError(
                            "Future database events or labels reached a retrospective phase"
                        )
                    vx = snapshot.features
                    fit_origin = pd.Timestamp(snapshot.database_cutoff)
                    requested = original_positions[positions]
                    if (
                        not vx.index.is_unique
                        or not vx.index.isin(requested).all()
                        or list(vx.columns) != list(x.columns)
                    ):
                        raise ValueError(
                            "Snapshot rows/schema must match the requested frozen cohort"
                        )
                    lookup = pd.Series(positions, index=requested)
                    positions = lookup.loc[vx.index].to_numpy()
                truth, entity_groups = (
                    y.iloc[positions].to_numpy(),
                    groups.iloc[positions].to_numpy(),
                )
                if task == "classification" and np.unique(truth).size < 2:
                    continue
                for policy in policies:
                    try:
                        model, audit = fit_temporal_policy(
                            x,
                            y,
                            times,
                            origin=fit_origin,
                            label_horizon=label_horizon,
                            refit_frames=refit_frames,
                            policy=policy,
                            task=task,
                            categorical=categorical,
                            random_state=seed,
                            row_limit=row_limit,
                            model_params=model_params,
                            frame_model_cache=cache,
                        )
                        prediction = np.asarray(
                            model.predict_proba(vx)[:, 1]
                            if task == "classification"
                            else model.predict(vx)
                        ).reshape(-1)
                        evidence.append(
                            dict(
                                origin=str(origin),
                                seed=seed,
                                policy=policy.name,
                                truth=truth,
                                prediction=prediction,
                                groups=entity_groups,
                            )
                        )
                        record = dict(
                            origin=str(origin),
                            seed=seed,
                            policy=policy.name,
                            score=_score(truth, prediction, task),
                            fit_audit=audit,
                        )
                    except InsufficientPolicySupport as exc:
                        record = dict(
                            origin=str(origin),
                            seed=seed,
                            policy=policy.name,
                            unavailable=str(exc),
                        )
                    records.append(record)
                    if on_evaluation is not None:
                        on_evaluation(record)
            del cache
    expanded, recipes = add_incumbent_blends(evidence, weights=blend_weights)
    selection = select_temporal_policy(
        expanded, task=task, deployment_seed=random_state
    )
    if selection["selected"] in recipes:
        recipe = recipes[selection["selected"]]
        base = next(p for p in policies if p.name == recipe["base_policy"])
        selected = replace(
            base, name=selection["selected"], blend_weight=recipe["blend_weight"]
        )
    else:
        selected = next(p for p in policies if p.name == selection["selected"])
    model, fit_audit = fit_temporal_policy(
        x,
        y,
        times,
        origin=forecast_origin,
        label_horizon=label_horizon,
        refit_frames=refit_frames,
        policy=selected,
        task=task,
        categorical=categorical,
        random_state=random_state,
        row_limit=row_limit,
        model_params=model_params,
        incumbent_policy=next(p for p in policies if p.name == "incumbent"),
    )
    selection["evaluations"] = records
    selection["selected_policy"] = asdict(selected)
    return SelectedTemporalEstimator(model, selected, selection, fit_audit)
