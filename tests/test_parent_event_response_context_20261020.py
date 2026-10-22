import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.parent_event_response_context_20261020 import ParentEventResponseContext


def write(path, data, *, key, clock=None, foreign=None):
    table = pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False)
    metadata = {k.encode(): json.dumps(v).encode() for k, v in dict(pkey_col=key, time_col=clock,
        fkey_col_to_pkey_table=foreign or {}, available_at=None).items()}
    pq.write_table(table.replace_schema_metadata(metadata), path)


def bundle(path, extras=None):
    write(path / "roots.parquet", dict(id=list(range(1, 21)), segment=[0.]*10+[1.]*10), key="id")
    events = pd.DataFrame(dict(id=range(1, 7), root=[1, 1, 2, 2, 3, 3], item=[1]*6,
        timestamp=pd.to_datetime(["2001-01-02"]*6), response=[1, 1, 0, 0, 1, 0]))
    if extras is not None:
        events = pd.concat([events, extras], ignore_index=True)
    write(path / "events.parquet", events, key="id", clock="timestamp", foreign=dict(root="roots", item="items"))
    write(path / "items.parquet", dict(id=[1]), key="id")


def extract(path, requests):
    provider = ParentEventResponseContext(path, "roots", discovery_cutoff="2001-01-05", horizon="4 days", temporary=path / "tmp")
    try:
        assert provider.audit["task_targets_used"] is False
        assert provider.fields == ["segment"]
        return provider.transform(requests, phase_boundary="2001-01-10")
    finally:
        provider.close()


def test_cold_root_can_borrow_observed_event_response_and_own_root_is_excluded(tmp_path):
    bundle(tmp_path)
    requests = pd.DataFrame(dict(__event_entity=[1, 4], __event_cutoff=["2001-01-10"]*2))
    before = extract(tmp_path, requests)
    assert before.loc[0, "parent_response_0_w0_peer_support"] == 2
    assert before.loc[1, "parent_response_0_w0_peer_support"] == 3
    bundle(tmp_path, pd.DataFrame(dict(id=[7], root=[1], item=[1], timestamp=pd.to_datetime(["2001-01-06"]), response=[0])))
    after = extract(tmp_path, requests)
    np.testing.assert_array_equal(before.iloc[0], after.iloc[0])


def test_strict_clock_and_batch_order(tmp_path):
    bundle(tmp_path)
    requests = pd.DataFrame(dict(__event_entity=[1, 4], __event_cutoff=["2001-01-10"]*2))
    before = extract(tmp_path, requests)
    bundle(tmp_path, pd.DataFrame(dict(id=[7, 8], root=[2, 3], item=[1, 1],
        timestamp=pd.to_datetime(["2001-01-10", "2001-01-11"]), response=[1, 1])))
    after = extract(tmp_path, requests)
    np.testing.assert_array_equal(before, after)
    reverse = extract(tmp_path, requests.iloc[::-1])
    np.testing.assert_array_equal(before, reverse.iloc[::-1])
    assert len(before.columns) <= 48
    assert not any("root" == c or "response" == c for c in before)


def test_training_vocabulary_does_not_expose_later_training_events(tmp_path):
    bundle(tmp_path, pd.DataFrame(dict(id=[7], root=[2], item=[1],
        timestamp=pd.to_datetime(["2001-01-04"]), response=[1])))
    provider = ParentEventResponseContext(tmp_path, "roots", discovery_cutoff="2001-01-05",
        earliest_query="2001-01-03", horizon="4 days", temporary=tmp_path / "tmp")
    try:
        actual = provider.transform(pd.DataFrame(dict(__event_entity=[1], __event_cutoff=["2001-01-03"])), phase_boundary="2001-01-10")
        assert actual.loc[0, "parent_response_0_w0_peer_support"] == 2
        assert actual.loc[0, "parent_response_0_w0_event_rate"] == np.float32(.25)
    finally:
        provider.close()
