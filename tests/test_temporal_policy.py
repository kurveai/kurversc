from dataclasses import replace
import pickle

import numpy as np
import pandas as pd
import pytest

from kurversc.temporal_policy import (
    InsufficientPolicySupport,
    TemporalFitPolicy,
    TemporalHoldoutSnapshot,
    add_incumbent_blends,
    bounded_temporal_policies,
    fit_temporal_policy,
    policy_support,
    select_and_fit_temporal_policy,
    select_temporal_policy,
)


def test_support_preserves_initial_plus_observed_refit_and_maturity():
    frame = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=9)})
    policy = TemporalFitPolicy("pool", mode="pooled", training_frames=3)
    plan = policy_support(
        frame,
        timestamp="t",
        origin="2024-01-09",
        label_horizon="2 days",
        refit_frames=2,
        policy=policy,
    )
    assert plan["refit"] == tuple(pd.to_datetime(["2024-01-06", "2024-01-07"]))
    assert plan["initial"] == tuple(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    )
    assert (
        pd.Timestamp("2024-01-05") not in plan["cutoffs"]
    )  # Label incomplete at refit start.


def test_support_ignores_labels_future_rows_and_row_order():
    frame = pd.DataFrame(
        {"t": pd.date_range("2024-01-01", periods=9), "target": range(9)}
    )
    kwargs = dict(
        timestamp="t",
        origin="2024-01-07",
        label_horizon="1 day",
        refit_frames=1,
        policy=TemporalFitPolicy("base"),
    )
    original = policy_support(frame, **kwargs)
    changed = frame.sample(frac=1, random_state=2)
    changed["target"] = -999
    assert original == policy_support(changed, **kwargs)


def test_incomplete_initial_support_is_not_silently_relaxed():
    with pytest.raises(InsufficientPolicySupport):
        policy_support(
            pd.DataFrame({"t": pd.date_range("2024-01-01", periods=4)}),
            timestamp="t",
            origin="2024-01-05",
            label_horizon="1 day",
            refit_frames=1,
            policy=TemporalFitPolicy("pool12", training_frames=12),
        )


def evidence(gains=(0.1, 0.1, 0.1), *, seeds=(0, 1)):
    records = []
    y = np.zeros(80)
    for index, gain in enumerate(gains):
        for seed in seeds:
            for name, prediction in (
                ("incumbent", np.ones(80)),
                ("challenger", np.ones(80) - gain),
            ):
                records.append(
                    dict(
                        origin=f"202{index}-01-01",
                        seed=seed,
                        policy=name,
                        truth=y,
                        prediction=prediction,
                        groups=np.arange(80),
                    )
                )
    return records


def test_selection_accepts_repeatable_gain_without_test_scores():
    result = select_temporal_policy(evidence(), task="regression", bootstrap_repeats=50)
    assert result["selected"] == "challenger"


def test_selection_rejects_mean_regression_and_older_fold_regression():
    assert (
        select_temporal_policy(evidence((-0.1, -0.1, -0.1)), task="regression")[
            "selected"
        ]
        == "incumbent"
    )
    result = select_temporal_policy(
        evidence((-0.1, 0.1, 0.1)), task="regression", bootstrap_repeats=30
    )
    assert result["candidates"]["challenger"]["reason"] == "older_origin_regression"


def test_selection_retains_incumbent_on_ties_or_insufficient_evidence():
    for records in (evidence((0, 0, 0)), evidence((0.1,)), evidence(seeds=(0,))):
        assert (
            select_temporal_policy(records, task="regression", bootstrap_repeats=30)[
                "selected"
            ]
            == "incumbent"
        )


def test_selection_missing_candidate_origin_is_not_ignored():
    records = [
        r
        for r in evidence()
        if not (r["policy"] == "challenger" and r["origin"].startswith("2020"))
    ]
    result = select_temporal_policy(records, task="regression")
    assert (
        result["candidates"]["challenger"]["reason"] == "missing_matched_origin_or_seed"
    )


def test_paired_row_mismatch_rejected():
    records = evidence()
    records[1]["groups"] = records[1]["groups"][::-1]
    with pytest.raises(ValueError, match="identical ordered rows"):
        select_temporal_policy(records, task="regression")


def test_seed_instability_is_not_averaged_away():
    records = evidence()
    for r in records:
        if r["policy"] == "challenger":
            r["prediction"] = np.ones(80) * (0.7 if r["seed"] == 0 else 1.02)
    result = select_temporal_policy(records, task="regression", bootstrap_repeats=30)
    assert result["candidates"]["challenger"]["reason"] == "recent_gain_not_seed_stable"


def test_selection_ranks_the_deployed_seed_not_an_unshipped_seed_ensemble():
    records = evidence()
    additional = []
    for r in records:
        if r["policy"] == "challenger":
            r["prediction"] = np.full(80, 0.98 if r["seed"] == 0 else 0.6)
            additional.append(
                {
                    **r,
                    "policy": "production_best",
                    "prediction": np.full(80, 0.95 if r["seed"] == 0 else 0.99),
                }
            )
    result = select_temporal_policy(
        [*records, *additional], task="regression", bootstrap_repeats=30
    )
    assert result["selected"] == "production_best"


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_fit_is_serializable_future_target_blind_and_bounded(task):
    x = pd.DataFrame({"x": np.tile(np.arange(12), 6)})
    times = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=6), 12))
    y = pd.Series((x.x % 2).astype(float) if task == "classification" else x.x * 0.1)
    kwargs = dict(
        origin="2024-01-05",
        label_horizon="1 day",
        refit_frames=1,
        policy=TemporalFitPolicy("pool", mode="pooled", training_frames=2),
        task=task,
        categorical=[],
        model_params={"iterations": 5, "thread_count": 1},
    )
    model, audit = fit_temporal_policy(x, y, times, **kwargs)
    changed = y.copy()
    changed.loc[times >= pd.Timestamp("2024-01-05")] = 999
    other, other_audit = fit_temporal_policy(x, changed, times, **kwargs)
    assert audit == other_audit and audit["training_rows"] == 36
    method = "predict_proba" if task == "classification" else "predict"
    np.testing.assert_array_equal(getattr(model, method)(x), getattr(other, method)(x))
    loaded = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(getattr(model, method)(x), getattr(loaded, method)(x))


def test_paired_uncertainty_is_row_order_invariant():
    records = evidence()
    expected = select_temporal_policy(records, task="regression", bootstrap_repeats=30)
    order = np.random.default_rng(3).permutation(80)
    reordered = [
        {**r, **{k: np.asarray(r[k])[order] for k in ("truth", "prediction", "groups")}}
        for r in records
    ]
    assert (
        select_temporal_policy(reordered, task="regression", bootstrap_repeats=30)
        == expected
    )


def test_missing_group_or_duplicate_evidence_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        select_temporal_policy([*evidence(), evidence()[0]], task="regression")
    records = evidence()
    records[0]["groups"] = np.full(80, np.nan)
    with pytest.raises(ValueError, match="Complete entity groups"):
        select_temporal_policy(records, task="regression")


def test_one_entity_repeated_across_origins_is_not_independent_evidence():
    records = evidence()
    for row in records:
        row["groups"] = np.zeros(80)
    result = select_temporal_policy(records, task="regression", bootstrap_repeats=30)
    assert result["selected"] == "incumbent"


@pytest.mark.parametrize("classes", [("negative", "positive"), (17, 2)])
def test_binary_selector_supports_native_label_order_not_only_zero_one(classes):
    records = []
    truth = np.array([classes[i % 2] for i in range(80)])
    good = np.array([i % 2 for i in range(80)], dtype=float)
    for origin in ["2020-01-01", "2021-01-01"]:
        for seed in [0, 1]:
            records.extend(
                [
                    dict(
                        origin=origin,
                        seed=seed,
                        policy="incumbent",
                        truth=truth,
                        prediction=np.full(80, 0.5),
                        groups=np.arange(80),
                    ),
                    dict(
                        origin=origin,
                        seed=seed,
                        policy="better",
                        truth=truth,
                        prediction=good,
                        groups=np.arange(80),
                    ),
                ]
            )
    result = select_temporal_policy(
        records, task="classification", bootstrap_repeats=30
    )
    assert result["selected"] == "better"


def test_ordered_policy_only_available_for_regression():
    assert len(bounded_temporal_policies("classification")) == 3
    assert len(bounded_temporal_policies("regression")) == 5
    with pytest.raises(ValueError):
        replace(TemporalFitPolicy("bad"), mode="unknown")


def test_zero_horizon_does_not_train_on_the_holdout_cutoff():
    plan = policy_support(
        pd.DataFrame({"t": pd.date_range("2024-01-01", periods=6)}),
        timestamp="t",
        origin="2024-01-06",
        label_horizon="0 days",
        refit_frames=1,
        policy=TemporalFitPolicy("base"),
    )
    assert plan["cutoffs"] == tuple(pd.to_datetime(["2024-01-04", "2024-01-05"]))


def test_standalone_selection_refit_and_audit_are_future_target_blind():
    x = pd.DataFrame({"x": np.tile(np.arange(8), 10)})
    times = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=10), 8))
    y = pd.Series(np.sin(x.x) + np.arange(80) * 0.005)
    kwargs = dict(
        forecast_origin="2024-01-09",
        label_horizon="1 day",
        refit_frames=1,
        task="regression",
        categorical=[],
        seeds=(0, 1),
        model_params={"iterations": 4, "thread_count": 1},
        policies=(TemporalFitPolicy("incumbent"), TemporalFitPolicy("identical")),
    )
    events = []
    fitted = select_and_fit_temporal_policy(
        x, y, times, pd.Series(x.x), on_evaluation=events.append, **kwargs
    )
    changed = y.copy()
    changed.loc[times >= pd.Timestamp("2024-01-09")] = 99999
    repeated = select_and_fit_temporal_policy(
        x, changed, times, pd.Series(x.x), **kwargs
    )
    assert fitted.policy.name == "incumbent" and events
    assert fitted.selection_audit == repeated.selection_audit
    assert fitted.fit_audit == repeated.fit_audit
    np.testing.assert_array_equal(
        fitted.estimator.predict(x), repeated.estimator.predict(x)
    )
    loaded = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(
        fitted.estimator.predict(x), loaded.estimator.predict(x)
    )


def test_standalone_selector_can_use_a_pooled_incumbent():
    x = pd.DataFrame({"x": np.tile(np.arange(8), 6)})
    times = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=6), 8))
    incumbent = TemporalFitPolicy("incumbent", mode="pooled", training_frames=2)
    fitted = select_and_fit_temporal_policy(
        x,
        pd.Series(x.x * 0.1),
        times,
        pd.Series(x.x),
        forecast_origin="2024-01-07",
        label_horizon="1 day",
        refit_frames=1,
        task="regression",
        categorical=[],
        seeds=(0, 1),
        policies=(incumbent,),
        model_params={"iterations": 4, "thread_count": 1},
    )
    assert fitted.policy == incumbent and fitted.fit_audit["training_rows"] == 24


def test_multicutoff_selector_requires_a_phase_snapshot_provider():
    with pytest.raises(ValueError, match="phase-frozen"):
        select_and_fit_temporal_policy(
            pd.DataFrame({"x": range(10)}),
            pd.Series(range(10)),
            pd.Series(pd.date_range("2024-01-01", periods=10)),
            pd.Series(range(10)),
            forecast_origin="2024-01-11",
            label_horizon="1 day",
            refit_frames=2,
            task="regression",
            categorical=[],
        )


def test_phase_snapshot_future_boundary_rejected_before_training():
    x = pd.DataFrame({"x": np.tile(np.arange(8), 10)})
    times = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=10), 8))

    def future_snapshot(origin, positions):
        return TemporalHoldoutSnapshot(
            x.iloc[positions], origin + pd.Timedelta(days=1), origin
        )

    with pytest.raises(ValueError, match="Future database"):
        select_and_fit_temporal_policy(
            x,
            pd.Series(x.x * 0.1),
            times,
            pd.Series(x.x),
            forecast_origin="2024-01-11",
            label_horizon="1 day",
            refit_frames=2,
            task="regression",
            categorical=[],
            seeds=(0, 1),
            policies=(TemporalFitPolicy("incumbent"),),
            heldout_feature_provider=future_snapshot,
        )


def test_phase_snapshot_validated_and_used_for_predictions():
    x = pd.DataFrame({"x": np.tile(np.arange(8), 10)})
    times = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=10), 8))
    calls = []

    def snapshot(origin, positions):
        calls.append((str(origin), positions.copy()))
        return TemporalHoldoutSnapshot(x.iloc[positions].assign(x=0), origin, origin)

    result = select_and_fit_temporal_policy(
        x,
        pd.Series(x.x * 0.1),
        times,
        pd.Series(x.x),
        forecast_origin="2024-01-11",
        label_horizon="1 day",
        refit_frames=2,
        task="regression",
        categorical=[],
        seeds=(0, 1),
        model_params={"iterations": 4, "thread_count": 1},
        policies=(TemporalFitPolicy("incumbent"),),
        heldout_feature_provider=snapshot,
    )
    assert len(calls) == 6 and result.policy.name == "incumbent"


def test_blends_reuse_aligned_predictions_and_keep_the_same_weight_all_folds():
    original = evidence()
    expanded, recipes = add_incumbent_blends(original, weights=(0.125, 0.5))
    assert len(expanded) == len(original) * 2 and len(recipes) == 2
    for row in expanded:
        if "__blend_" in row["policy"]:
            weight = recipes[row["policy"]]["blend_weight"]
            np.testing.assert_allclose(row["prediction"], 1 - 0.1 * weight)
    with pytest.raises(ValueError):
        add_incumbent_blends(original, weights=(0,))


def test_blended_policy_fits_and_serializes_the_exact_incumbent_challenger_mix():
    x = pd.DataFrame({"x": np.tile(np.arange(8), 6)})
    times = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=6), 8))
    y = pd.Series(x.x * 0.1)
    kwargs = dict(
        origin="2024-01-07",
        label_horizon="1 day",
        refit_frames=1,
        task="regression",
        categorical=[],
        model_params={"iterations": 4, "thread_count": 1},
    )
    policy = TemporalFitPolicy(
        "blend", mode="pooled", training_frames=3, blend_weight=0.125
    )
    model, audit = fit_temporal_policy(x, y, times, policy=policy, **kwargs)
    control, _ = fit_temporal_policy(
        x, y, times, policy=TemporalFitPolicy("incumbent"), **kwargs
    )
    challenger, _ = fit_temporal_policy(
        x, y, times, policy=replace(policy, blend_weight=1), **kwargs
    )
    np.testing.assert_array_equal(
        model.predict(x), 0.875 * control.predict(x) + 0.125 * challenger.predict(x)
    )
    assert audit["policy"]["blend_weight"] == 0.125
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(model)).predict(x), model.predict(x)
    )
