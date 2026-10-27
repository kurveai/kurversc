import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.enum_event_history_20261020 import EnumEventHistory


def write(path, frame, **meta):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update({k.encode(): json.dumps(v).encode() for k, v in meta.items()})
    pq.write_table(table.replace_schema_metadata(metadata), path)


def provider(tmp_path, *, delayed=False):
    db = tmp_path / "db"
    db.mkdir()
    write(db / "root.parquet", pd.DataFrame(dict(Id=[1, 2])), pkey_col="Id", time_col=None, fkey_col_to_pkey_table={})
    frame = pd.DataFrame(dict(Id=np.arange(60), RootId=np.tile([1, 2], 30),
        TypeId=np.tile([2, 3, 2], 20), when=pd.to_datetime(["2025-01-01"]*60)))
    if delayed:
        frame["known"] = pd.Timestamp("2025-01-05")
    extra = pd.DataFrame(dict(Id=[60, 61, 62], RootId=[1, 1, 1], TypeId=[2, 2, 2],
        when=pd.to_datetime(["2025-01-06", "2025-01-08", "2025-01-07"])))
    if delayed:
        extra["known"] = pd.to_datetime(["2025-01-06", "2025-01-08", "2025-01-09"])
    frame = pd.concat([frame, extra], ignore_index=True)
    write(db / "events.parquet", frame, pkey_col="Id", time_col="when", fkey_col_to_pkey_table={"RootId": "root"},
        available_at="known" if delayed else None)
    return EnumEventHistory(db, "root", discovery_cutoff="2025-01-06", horizon="4 days", temporary=tmp_path / "spill")


def test_low_cardinality_nonstructural_id_suffix_is_preserved_as_an_event_type(tmp_path):
    p = provider(tmp_path)
    try:
        assert p.fields[0][0].response == "TypeId"
        assert p.fields[0][1] == [2., 3.]
        q = pd.DataFrame(dict(__event_entity=[1, 2], __event_cutoff=[pd.Timestamp("2025-01-06")]*2))
        values = p.transform(q, phase_boundary="2025-01-06")
        assert values.enum_event_0_0_count.tolist() == [20, 20]
        assert values.enum_event_0_1_count.tolist() == [10, 10]
        assert p.audit["structural_keys_excluded_by_metadata"]
        np.testing.assert_array_equal(values.iloc[:1], p.transform(q.iloc[:1], phase_boundary="2025-01-06"))
        np.testing.assert_array_equal(values.iloc[::-1], p.transform(q.iloc[::-1], phase_boundary="2025-01-06"))
        with pytest.raises(ValueError):
            p.transform(q.assign(hidden=1), phase_boundary="2025-01-06")
    finally:
        p.close()


def test_strict_phase_cutoff_excludes_equality_and_future_source_rows(tmp_path):
    p = provider(tmp_path, delayed=True)
    try:
        q = pd.DataFrame(dict(__event_entity=[1], __event_cutoff=[pd.Timestamp("2025-01-08")]))
        included = p.transform(q, phase_boundary="2025-01-06")
        assert included.enum_event_0_0_count.iloc[0] == 20
        # Jan6 is visible at Jan8, but the event exactly at the query and
        # the Jan7 event not announced until Jan9 both remain unavailable.
        later = p.transform(q, phase_boundary="2025-01-08")
        assert later.enum_event_0_0_count.iloc[0] == 21
        # Never allow requests before vocabulary discovery: that could expose
        # a future-defined column. The phase clock must also follow discovery.
        with pytest.raises(ValueError):
            p.transform(q, phase_boundary="2025-01-05")
    finally:
        p.close()
