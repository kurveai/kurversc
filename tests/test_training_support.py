import pandas as pd
import pytest

from kurversc.training_support import (
    available_root_training,
    recency_weights,
    select_training_support,
)


def cohort():
    return pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2024-01-01"] * 3
                + ["2024-01-03"] * 2
                + ["2024-01-05"]
                + ["2024-01-09"] * 20
            ),
            "target": range(26),
        }
    )


def select(frame, **kwargs):
    return select_training_support(
        frame,
        timestamp="time",
        evaluation_cutoff="2024-01-07",
        label_horizon="2 days",
        minimum_rows=4,
        **kwargs,
    )


def test_recent_backfill_and_completed_boundary():
    plan = select(cohort())
    assert plan.cutoffs == tuple(
        pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"])
    )
    assert plan.rows == 6 and plan.complete


def test_no_target_or_row_order_dependence():
    frame = cohort().sample(frac=1, random_state=5)
    frame["target"] = -999
    assert select(frame) == select(cohort().drop(columns="target"))


def test_budget_does_not_duplicate_sparse_rows():
    plan = select(cohort(), max_frames=2)
    assert plan.rows == 3 and not plan.complete and len(plan.cutoffs) == 2


def test_minimum_frames():
    plan = select_training_support(
        cohort(),
        timestamp="time",
        evaluation_cutoff="2024-01-07",
        label_horizon="2 days",
        minimum_rows=1,
        minimum_frames=2,
    )
    assert len(plan.cutoffs) == 2 and plan.rows == 3


def test_support_rule_is_reapplied_at_the_new_forecast_origin():
    training = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2024-01-01"] * 500
                + ["2024-01-03"] * 500
                + ["2024-01-05"] * 500
                + ["2024-01-07"] * 500
            )
        }
    )
    earlier = select_training_support(
        training,
        timestamp="time",
        evaluation_cutoff="2024-01-07",
        label_horizon="2 days",
        minimum_rows=1000,
    )
    final = select_training_support(
        training,
        timestamp="time",
        evaluation_cutoff="2024-01-09",
        label_horizon="2 days",
        minimum_rows=1000,
    )
    assert earlier.cutoffs == tuple(pd.to_datetime(["2024-01-03", "2024-01-05"]))
    assert final.cutoffs == tuple(pd.to_datetime(["2024-01-05", "2024-01-07"]))
    assert final.rows == 1000  # Not earlier 1000 plus the newly observed 500.


def test_recency_normalization():
    weights = recency_weights(
        pd.Series(pd.to_datetime(["2024-01-01", "2024-01-03"])), half_life_days=2
    )
    assert weights.mean() == pytest.approx(1)
    assert weights.iloc[1] / weights.iloc[0] == pytest.approx(2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_rows": 0},
        {"minimum_frames": 2, "max_frames": 1},
        {"label_horizon": "-1 day"},
    ],
)
def test_invalid_budgets(kwargs):
    params = dict(
        timestamp="time",
        evaluation_cutoff="2024-01-07",
        label_horizon="2 days",
        minimum_rows=4,
    )
    params.update(kwargs)
    with pytest.raises(ValueError):
        select_training_support(cohort(), **params)


def test_no_mature_history():
    with pytest.raises(ValueError, match="No completed"):
        select(cohort().iloc[-20:])


def test_root_availability_preserves_values_order_and_ignores_target():
    training = pd.DataFrame(
        {
            "id": [2, 1, 3, 1],
            "time": pd.to_datetime(
                ["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-01"]
            ),
            "target": [9, 8, 7, 6],
        }
    )
    roots = pd.DataFrame(
        {"key": [1, 2], "created": pd.to_datetime(["2024-01-02", "2024-01-01"])}
    )
    kwargs = dict(
        label_key="id", root_key="key", timestamp="time", root_timestamp="created"
    )
    result = available_root_training(training, roots, **kwargs)
    pd.testing.assert_frame_equal(result, training.iloc[[0, 1]])
    assert (
        len(available_root_training(training.drop(columns="target"), roots, **kwargs))
        == 2
    )


def test_root_duplicate_key_rejected():
    with pytest.raises(ValueError, match="unique"):
        available_root_training(
            pd.DataFrame({"id": [1], "time": ["2024-01-01"]}),
            pd.DataFrame({"id": [1, 1]}),
            label_key="id",
            root_key="id",
            timestamp="time",
        )
