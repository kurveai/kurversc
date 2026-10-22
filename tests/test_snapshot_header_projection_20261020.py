import pandas as pd

from scripts import materialize_policy_folds as snapshots


def test_snapshot_alignment_does_not_load_historical_payloads(tmp_path, monkeypatch):
    parts = []
    for n, dates in (("history", ["2001-01-01", "2001-01-02"]), ("validation", ["2001-01-03"])):
        frame = pd.DataFrame(dict(__event_entity=[f"root-{d}" for d in dates],
            __event_cutoff=pd.to_datetime(dates), target=range(len(dates)),
            expensive_payload=["unused large text"*1000]*len(dates), numeric_payload=range(len(dates))))
        frame.to_parquet(tmp_path / f"{n}.parquet", index=False)
        parts.append(frame)
    old = pd.concat(parts, ignore_index=True)
    calls = []
    original_reader = pd.read_parquet

    def read(path, **kwargs):
        calls.append(kwargs["columns"])
        assert set(kwargs["columns"]) == {"__event_entity", "__event_cutoff", "target"}
        return original_reader(path, **kwargs)

    monkeypatch.setattr(snapshots.pd, "read_parquet", read)
    headers = snapshots.alignment_headers(tmp_path, dict(target="target"))
    pd.testing.assert_frame_equal(headers, old[list(headers)])
    assert len(calls) == 2
