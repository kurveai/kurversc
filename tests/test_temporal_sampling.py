import pandas as pd
import pytest

from kurversc.temporal_sampling import (
    choose_temporal_fit_plan,
    sample_mature_history,
)


def test_temporal_fit_plan_uses_only_cutoff_support() -> None:
    def frame(rows: tuple[int, ...]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        f"2020-0{month}-01"
                        for month, count in enumerate(rows, start=1)
                        for _ in range(count)
                    ]
                ),
                "target": [-999] * sum(rows),
            }
        )

    validation = frame((10,))
    balanced = choose_temporal_fit_plan(
        frame((180, 160, 200)), validation, timestamp="time"
    )
    sparse = choose_temporal_fit_plan(
        frame((180, 700, 2000)), validation, timestamp="time"
    )
    assert (balanced.training_frames, balanced.estimator_mode) == (
        3,
        "frame_ensemble",
    )
    assert (sparse.training_frames, sparse.estimator_mode) == (3, "pooled")
    assert balanced.recent_frame_rows == (180, 160, 200)
    assert choose_temporal_fit_plan(
        frame((180, 160, 200)).assign(target=1),
        validation.assign(target=1),
        timestamp="time",
    ) == balanced
    assert choose_temporal_fit_plan(
        frame((180, 160, 200)), frame((10, 10)), timestamp="time"
    ).reason == "multiple_validation_cutoffs"
    assert choose_temporal_fit_plan(
        frame((180, 160)), validation, timestamp="time"
    ).reason == "insufficient_training_cutoffs"
    assert choose_temporal_fit_plan(
        frame((80, 160, 200)), validation, timestamp="time"
    ).reason == "sparse_training_cutoff"
    assert choose_temporal_fit_plan(
        frame((180, 160, 200)),
        validation,
        timestamp="time",
        max_training_rows=500,
    ).reason == "training_row_budget"


def test_temporal_fit_plan_counts_only_point_in_time_available_roots() -> None:
    training = pd.DataFrame(
        {
            "entity_id": list(range(300)) * 3,
            "time": pd.to_datetime(
                ["2020-01-01"] * 300
                + ["2020-03-01"] * 300
                + ["2020-04-01"] * 300
            ),
            "target": [-1] * 900,
        }
    )
    root = pd.DataFrame(
        {
            "root_id": range(300),
            "created_at": pd.to_datetime(
                ["2019-12-01"] * 150 + ["2020-02-01"] * 150
            ),
        }
    )
    validation = pd.DataFrame({"time": pd.to_datetime(["2020-05-01"])})
    raw = choose_temporal_fit_plan(training, validation, timestamp="time")
    eligible = choose_temporal_fit_plan(
        training,
        validation,
        timestamp="time",
        root_frame=root,
        label_key="entity_id",
        root_key="root_id",
        root_timestamp="created_at",
    )
    assert raw.recent_frame_rows == (300, 300, 300)
    assert raw.estimator_mode == "frame_ensemble"
    assert eligible.recent_frame_rows == (150, 300, 300)
    assert eligible.estimator_mode == "frame_ensemble"
    root.loc[100:149, "created_at"] = pd.Timestamp("2020-02-01")
    eligible = choose_temporal_fit_plan(
        training,
        validation,
        timestamp="time",
        root_frame=root,
        label_key="entity_id",
        root_key="root_id",
        root_timestamp="created_at",
    )
    assert eligible.recent_frame_rows == (100, 300, 300)
    assert eligible.estimator_mode == "pooled"


def test_sparse_history_backfills_by_support_without_future_labels():
    frame = pd.DataFrame(
        {"time": pd.date_range("2000-01-01", periods=12, freq="D"), "y": range(12)}
    )
    frame["ready"] = frame.time + pd.Timedelta(days=1)
    kwargs = dict(
        timestamp="time",
        available_at="ready",
        cutoff="2000-01-12",
        preferred_cutoffs=3,
        per_cutoff=2,
    )
    legacy = sample_mature_history(frame, **kwargs, minimum_rows=0)
    extended = sample_mature_history(frame, **kwargs, minimum_rows=8)
    assert len(legacy) == 3
    assert len(extended) == 8
    assert extended.ready.lt(pd.Timestamp("2000-01-12")).all()
    poisoned = frame.copy()
    poisoned.y = -100
    pd.testing.assert_frame_equal(
        extended.drop(columns="y"),
        sample_mature_history(poisoned, **kwargs, minimum_rows=8).drop(columns="y"),
    )


def test_dense_history_keeps_recent_diversity_and_no_resampling_duplication():
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2000-01-01"] * 5 + ["2000-02-01"] * 5),
            "id": range(10),
        }
    )
    frame["ready"] = frame.time + pd.Timedelta(days=1)
    args = dict(
        timestamp="time",
        available_at="ready",
        cutoff="2001-01-01",
        preferred_cutoffs=1,
        per_cutoff=3,
        minimum_rows=4,
    )
    first = sample_mature_history(frame, **args)
    pd.testing.assert_frame_equal(first, sample_mature_history(frame, **args))
    assert len(first) == 6
    assert first.id.is_unique
    with pytest.raises(ValueError, match="No strictly"):
        sample_mature_history(frame, **dict(args, cutoff="1900-01-01"))
