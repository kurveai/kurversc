import numpy as np
import pandas as pd
import pytest

from scripts.sequence_basis import SequenceBasis


def example():
    frame = pd.DataFrame(dict(a_observed_history_seconds=[86400., np.nan],
        a_events_per_observed_day=[5., np.nan], a_num_events_1d=[2., np.nan],
        a_num_events_3d=[6., np.nan], a_date_min=pd.to_datetime(["2020-01-01", None]),
        a_date_max=pd.to_datetime(["2020-01-02", None])))
    schema = dict(columns=list(frame), categorical_columns=[], datetime_columns=["a_date_min", "a_date_max"])
    return frame, schema


def test_reconstructs_existing_sequence_without_labels_or_guessed_lifetime_window():
    frame, schema = example()
    basis = SequenceBasis().fit(schema)
    value = basis.transform(frame)
    assert value.a_activity_rate_1d.iloc[0] == 2
    assert value.a_activity_share_1d.iloc[0] == .2  # Lifetime=10, not largest window=6.
    assert value.a_activity_burst_1v3.iloc[0] == 1/3
    assert value.a_active_span_seconds.iloc[0] == 86400
    assert value.a_activities_per_active_day.iloc[0] == 5
    assert value.iloc[1].isna().all()
    assert basis.audit["alternative_graph_used"] is False
    assert basis.audit["labels_used"] is False
    pd.testing.assert_frame_equal(frame, example()[0])


def test_existing_outputs_are_not_duplicated_and_full_block_is_budgeted():
    frame, schema = example()
    schema["columns"].append("a_activity_rate_1d")
    assert "a_activity_rate_1d" not in SequenceBasis().fit(schema).columns
    with pytest.raises(ValueError, match="budget"):
        SequenceBasis().fit(schema, maximum_columns=2)


def test_inconsistent_source_aggregates_fail_closed():
    frame, schema = example()
    basis = SequenceBasis().fit(schema)
    with pytest.raises(ValueError, match="monotone"):
        basis.transform(frame.assign(a_num_events_1d=[7., np.nan]))
    with pytest.raises(ValueError, match="integer event"):
        basis.transform(frame.assign(a_events_per_observed_day=[5.25, np.nan]))
    with pytest.raises(ValueError, match="window"):
        basis.transform(frame.assign(a_num_events_3d=[11., np.nan]))


def test_span_matches_duckdb_second_boundary_semantics():
    frame, schema = example()
    frame.loc[0, "a_date_min"] = pd.Timestamp("2020-01-01 00:00:01.9")
    frame.loc[0, "a_date_max"] = pd.Timestamp("2020-01-01 00:00:02.1")
    assert SequenceBasis().fit(schema).transform(frame).a_active_span_seconds.iloc[0] == 1


def test_float_roundoff_in_a_single_event_denominator_is_not_a_zero_event():
    frame, schema = example()
    frame.loc[0, "a_events_per_observed_day"] = .49999999999999994
    frame.loc[0, "a_num_events_1d"] = 1.
    frame.loc[0, "a_num_events_3d"] = 1.
    value = SequenceBasis().fit(schema).transform(frame)
    assert value.a_activity_share_1d.iloc[0] == 1.
