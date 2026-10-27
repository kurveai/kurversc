import numpy as np
import pandas as pd
import pytest

from scripts.activity_timeline_challenger import EventChannel
from scripts.bounded_channel_transitions import BoundedChannelTransitions
from scripts.engagement_hypothesis_features import ChannelTransitions


def test_bitset_matches_original_ties_windows_distant_predecessors_and_duplicates(tmp_path):
    channels = []
    for i, dates in enumerate((
        ["2012-01-01", "2020-04-01", "2020-04-01", "2020-05-01", "2020-05-02"],
        ["2012-01-01", "2020-04-01", "2020-04-02", "2020-04-25"],
        ["2020-04-01", "2020-04-02", "2020-04-25"],
    )):
        path = tmp_path / f"channel{i}.parquet"
        pd.DataFrame({"actor": ["arbitrary-key"]*len(dates), "at": pd.to_datetime(dates)}).to_parquet(path, index=False)
        channels.append(EventChannel(path, "actor", "at"))
    requests = pd.DataFrame({"__event_entity": ["arbitrary-key", "none", "arbitrary-key", "arbitrary-key"],
        "__event_cutoff": ["2020-05-01", "2020-05-01", "2020-05-01", "2020-04-15"]})
    results = []
    for cls in (ChannelTransitions, BoundedChannelTransitions):
        provider = cls(channels, temporary=tmp_path/cls.__name__, threads=1)
        try:
            results.append(provider.transform(requests, phase_boundary="2020-05-01", horizon="20D"))
        finally:
            provider.close()
    pd.testing.assert_frame_equal(*results)
    assert results[1].iloc[3].ct_w0_p0_c0_gap > 2000  # Window retains distant predecessor.
    np.testing.assert_array_equal(results[1].iloc[0], results[1].iloc[2])
    assert results[1].iloc[1].filter(like="_count").eq(0).all()
    assert len(results[1].columns) == 36


def test_recent_window_is_strict_and_future_changes_do_not_affect_features(tmp_path):
    path = tmp_path/"events.parquet"
    past = pd.DataFrame({"owner": ["s"]*3, "at": pd.to_datetime(["2010-01-01", "2020-04-01", "2020-04-20"])})
    future = pd.DataFrame({"owner": ["s"]*2, "at": pd.to_datetime(["2020-05-01", "2020-05-02"])})
    query = pd.DataFrame({"__event_entity": ["s"], "__event_cutoff": ["2020-05-01"]})
    results = []
    for i, data in enumerate((past, pd.concat([past, future]))):
        data.to_parquet(path, index=False)
        provider = BoundedChannelTransitions([EventChannel(path, "owner", "at")], temporary=tmp_path/f"spill{i}", threads=1)
        try:
            results.append(provider.transform(query, phase_boundary="2020-05-01", horizon="10D"))
            with pytest.raises(ValueError, match="authorized"):
                provider.transform(query, phase_boundary="2020-04-30", horizon="10D")
            with pytest.raises(ValueError, match="finite"):
                provider.transform(query, phase_boundary="2020-05-01", horizon="0D")
        finally:
            provider.close()
    pd.testing.assert_frame_equal(*results)
    assert results[0].iloc[0].ct_w0_p0_c0_count == 0
    assert results[0].iloc[0].ct_w1_p0_c0_count == 2
