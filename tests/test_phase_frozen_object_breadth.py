import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.object_breadth_challenger import ObjectBreadth
from scripts.phase_frozen_object_breadth import PhaseFrozenObjectBreadth


def save(path, frame, *, primary, foreign=None, time=None):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    info = {"pkey_col": primary, "fkey_col_to_pkey_table": foreign or {}, "time_col": time}
    metadata = dict(table.schema.metadata or {})
    for key, value in info.items():
        metadata[key.encode()] = json.dumps(value).encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)


def test_independent_knowledge_clock_and_single_phase_equivalence(tmp_path):
    database = tmp_path / "db"
    database.mkdir()
    save(database / "people.parquet", pd.DataFrame({"id": ["u"]}), primary="id")
    save(database / "items.parquet", pd.DataFrame({"id": range(64)}), primary="id")
    events = pd.DataFrame({"id": range(66), "person": ["u"] * 66,
        "item": [*range(64), 0, 1],
        "at": [pd.Timestamp("2020-01-01")] * 64 + [pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-12")]})
    source = database / "events.parquet"
    save(source, events, primary="id", foreign={"person": "people", "item": "items"}, time="at")
    kwargs = dict(training_cutoff="2020-01-02", horizon="30D", temporary=tmp_path / "spill")
    old = ObjectBreadth(database, "people", **kwargs)
    new = PhaseFrozenObjectBreadth(database, "people", **kwargs)
    try:
        same = pd.DataFrame({"__event_entity": ["u"], "__event_cutoff": [pd.Timestamp("2020-01-10")]})
        pd.testing.assert_frame_equal(old.transform(same, phase_boundary="2020-01-10"),
                                      new.transform(same, phase_boundary="2020-01-10"))
        future = pd.DataFrame({"__event_entity": ["u", "u"],
                               "__event_cutoff": pd.to_datetime(["2020-01-20", "2020-01-11"])})
        before = new.transform(future, phase_boundary="2020-01-11")
        assert before.loc[1, "ob_path0_w0_events"] == 64
        assert before.loc[0, "ob_path0_w0_events"] == 0  # Window advances; source visibility does not.
        events.loc[64:, "item"] = 63
        events.loc[64:, "at"] = pd.Timestamp("2020-01-15")
        save(source, events, primary="id", foreign={"person": "people", "item": "items"}, time="at")
        pd.testing.assert_frame_equal(before, new.transform(future, phase_boundary="2020-01-11"))
        assert np.isfinite(before.filter(regex="events|objects")).all().all()
    finally:
        old.close()
        new.close()
