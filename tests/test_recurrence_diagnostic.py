import numpy as np
import pandas as pd
import pytest

from scripts.activity_timeline_challenger import EventChannel
from scripts.diagnose_event_recurrence import RecurrenceDiagnostic


def test_lifetime_gap_and_channel_conditioning_are_strict_and_ordered(tmp_path):
    first = tmp_path / "arbitrary_a.parquet"
    second = tmp_path / "arbitrary_b.parquet"
    pd.DataFrame({"owner":["x","x","x","x"],"clock":pd.to_datetime(
        ["2017-01-01","2018-01-01","2020-01-01","2020-01-02"])}).to_parquet(first,index=False)
    pd.DataFrame({"who":["x"],"when":pd.to_datetime(["2019-12-30"])}).to_parquet(second,index=False)
    provider = RecurrenceDiagnostic([EventChannel(first,"owner","clock"),EventChannel(second,"who","when")],
        horizon="91D",temporary=tmp_path / "spill",threads=1)
    requests = pd.DataFrame({"__event_entity":["absent","x","x"],"__event_cutoff":["2020-01-01"]*3})
    try:
        result = provider.transform(requests,phase_boundary="2020-01-01")
        assert result.iloc[0].rc_all_events == 0
        assert np.isnan(result.iloc[0].rc_all_recency)
        assert result.iloc[1].rc_all_events == 3
        assert result.iloc[1].rc_channel0_events == 2  # boundary/future events excluded
        assert result.iloc[1].rc_channel0_returns_365 == 1
        assert result.iloc[1].rc_channel0_pre730_days == 2  # lifetime includes older history
        assert result.iloc[1].rc_channel1_recency == 2
        np.testing.assert_array_equal(result.iloc[1],result.iloc[2])
        assert len(result.columns) == 75
        with pytest.raises(ValueError,match="authorized"):
            provider.transform(requests,phase_boundary="2019-12-31")
    finally:
        provider.close()


def test_within_day_cutoff_and_multiple_query_times_do_not_open_future_rows(tmp_path):
    path = tmp_path / "events.parquet"
    pd.DataFrame({"owner":[7,7,7],"clock":pd.to_datetime(
        ["2020-01-01 10:00:00","2020-01-01 12:00:00","2020-01-01 13:00:00"])}).to_parquet(path,index=False)
    provider = RecurrenceDiagnostic([EventChannel(path,"owner","clock")],
        horizon="4D",temporary=tmp_path / "spill",threads=1)
    requests = pd.DataFrame({"__event_entity":[7,7],"__event_cutoff":pd.to_datetime(
        ["2020-01-01 13:00:00","2020-01-01 12:00:00"])})
    try:
        frame = provider.transform(requests,phase_boundary="2020-01-01 13:00:00")
        assert frame.iloc[0].rc_all_events == 2
        assert frame.iloc[1].rc_all_events == 1
        assert frame.iloc[0].rc_all_days == 1
        np.testing.assert_allclose(frame.rc_all_recency,[1/24,2/24],rtol=1e-6)
    finally:
        provider.close()


def test_diagnostic_rejects_missing_entities_and_excessive_channels(tmp_path):
    path = tmp_path / "events.parquet"
    pd.DataFrame({"owner":[7],"clock":pd.to_datetime(["2020-01-01"])}).to_parquet(path,index=False)
    channel = EventChannel(path,"owner","clock")
    with pytest.raises(ValueError,match="six"):
        RecurrenceDiagnostic([channel]*7,horizon="4D",temporary=tmp_path / "spill",threads=1)
    provider = RecurrenceDiagnostic([channel],horizon="4D",temporary=tmp_path / "spill",threads=1)
    try:
        requests = pd.DataFrame({"__event_entity":[None],"__event_cutoff":["2020-01-02"]})
        with pytest.raises(ValueError,match="Missing root"):
            provider.transform(requests,phase_boundary="2020-01-02")
    finally:
        provider.close()
