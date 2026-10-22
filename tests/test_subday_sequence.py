import numpy as np
import pandas as pd
import pytest

from scripts.activity_timeline_challenger import EventChannel
from scripts.subday_sequence import SubdaySequence


def provider(tmp_path, times, actors=None, name="events"):
    path = tmp_path / f"{name}.parquet"
    pd.DataFrame({"actor": actors or [1] * len(times), "time": pd.to_datetime(times, format="mixed")}).to_parquet(path, index=False)
    return SubdaySequence([EventChannel(path, "actor", "time")], horizon="4d", temporary=tmp_path / f"spill-{name}")


def requests(actors, times):
    return pd.DataFrame({"__event_entity": actors, "__event_cutoff": pd.to_datetime(times), "ignored_target": 999})


def test_subday_gaps_preserved_duplicates_not_false_zero_gaps(tmp_path):
    p = provider(tmp_path, ["2020-01-01 01:00", "2020-01-01 01:00", "2020-01-01 03:00", "2020-01-01 07:00"])
    try:
        frame = p.transform(requests([1], ["2020-01-01 08:00"]), phase_boundary="2020-01-01 08:00")
        row = frame.iloc[0]
        assert row.sq_channel0_events == 4 and row.sq_channel0_instants == 3
        assert row.sq_channel0_gap_support == 2
        assert row.sq_channel0_gap_mean_h == pytest.approx(3 / 96)
        assert row.sq_channel0_age_h == pytest.approx(1 / 96)
        assert row.sq_channel0_session_count_h64 == 3
        assert frame.shape == (1, 54)
    finally:
        p.close()


def test_strict_cutoff_and_frozen_phase_for_future_requests(tmp_path):
    p = provider(tmp_path, ["2020-01-01 01:00", "2020-01-01 02:00", "2020-01-01 03:00"])
    try:
        frame = p.transform(requests([1, 1], ["2020-01-01 02:00", "2020-01-01 04:00"]),
            phase_boundary="2020-01-01 02:00")
        assert frame.sq_all_events.tolist() == [1, 1]
        assert frame.sq_all_age_h.iloc[1] == pytest.approx(3 / 96)
        assert p.last_audit["phases"][1]["source_ceiling"] == "2020-01-01 02:00:00"
    finally:
        p.close()


def test_row_keys_duplicates_missing_support_and_target_ignored(tmp_path):
    p = provider(tmp_path, ["2020-01-01 01:00", "2020-01-01 03:00"], [1, 2])
    try:
        r = requests([2, 1, 2, 99], ["2020-01-01 04:00"] * 4)
        a = p.transform(r, phase_boundary="2020-01-01 04:00")
        b = p.transform(r.assign(ignored_target=-999).iloc[[3, 1, 0, 2]], phase_boundary="2020-01-01 04:00")
        pd.testing.assert_frame_equal(a.iloc[[3, 1, 0, 2]].reset_index(drop=True), b)
        assert a.sq_all_events.tolist() == [1, 1, 1, 0]
        assert p.support.tolist() == [False, True, True, True]
        assert a.sq_all_age_h.iloc[0] != a.sq_all_age_h.iloc[1]
        assert np.isnan(a.sq_all_age_h.iloc[-1])
    finally:
        p.close()


def test_future_source_append_cannot_change_earlier_features(tmp_path):
    first = provider(tmp_path, ["2020-01-01 01:00"], name="first")
    second = provider(tmp_path, ["2020-01-01 01:00", "2020-01-02"], name="second")
    try:
        r = requests([1], ["2020-01-01 04:00"])
        pd.testing.assert_frame_equal(first.transform(r, phase_boundary="2020-01-01 04:00"),
            second.transform(r, phase_boundary="2020-01-01 04:00"))
    finally:
        first.close()
        second.close()


def test_invalid_horizon_and_missing_keys_rejected(tmp_path):
    with pytest.raises(ValueError):
        SubdaySequence([], horizon="4d", temporary=tmp_path)
    p = provider(tmp_path, ["2020-01-01"])
    try:
        with pytest.raises(ValueError):
            p.transform(requests([None], ["2020-01-02"]), phase_boundary="2020-01-02")
    finally:
        p.close()
