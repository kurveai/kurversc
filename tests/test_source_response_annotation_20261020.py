import numpy as np
import pandas as pd
import pytest
import pyarrow.parquet as pq

from scripts.source_response_annotation_20261020 import SourceResponseAnnotation
from kurversc.signal_challengers import ChallengerUnavailable
from test_parent_event_response_context_20261020 import write


def bundle(path, *, reverse=False, future_mark=0):
    write(path / "actors.parquet", dict(id=range(1, 21), segment=[0]*10+[1]*10), key="id")
    write(path / "objects.parquet", dict(id=[1, 2], quality=[0., 1.]), key="id")
    frame = pd.DataFrame(dict(id=range(200), actor=np.arange(200)%20+1,
        object=np.arange(200)%2+1, at=pd.Timestamp("2001-01-01"),
        position=np.arange(200)%4, response=np.arange(200)%3==0))
    frame.response = frame.response.astype(int)
    future = pd.DataFrame(dict(id=[200], actor=[1], object=[1], at=pd.Timestamp("2001-01-10"),
        position=[0], response=[future_mark]))
    frame = pd.concat([frame, future], ignore_index=True)
    if reverse:
        frame = frame.iloc[::-1]
    write(path / "events.parquet", frame, key="id", clock="at", foreign=dict(actor="actors", object="objects"))


def extract(path, requests):
    provider = SourceResponseAnnotation(path, "actors", training_cutoff="2001-01-05", horizon="4 days",
        temporary=path / "temp", maximum_training_rows=100, model_iterations=3)
    try:
        assert provider.audit["task_labels_used"] is False
        assert provider.audit["training_rows"] == 100
        assert all(f["name"] not in {"id", "actor", "object", "response", "at"} for f in provider.fields)
        result = provider.transform(requests, phase_boundary="2001-01-10")
        return result, provider.support.copy()
    finally:
        provider.close()


def test_model_annotation_is_frozen_time_safe_order_invariant_and_key_free(tmp_path):
    bundle(tmp_path)
    requests = pd.DataFrame(dict(__event_entity=[1, 2, 999], __event_cutoff=["2001-01-10"]*3))
    before, support = extract(tmp_path, requests)
    assert support.tolist() == [True, True, False]
    assert len(before.columns) == 30
    assert before.iloc[-1].isna().all()
    bundle(tmp_path, reverse=True, future_mark=1)
    after, _ = extract(tmp_path, requests)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    reversed_rows, _ = extract(tmp_path, requests.iloc[::-1])
    pd.testing.assert_frame_equal(before, reversed_rows.iloc[::-1].reset_index(drop=True), check_exact=True)


def test_annotation_refuses_prediscovery_queries(tmp_path):
    bundle(tmp_path)
    provider = SourceResponseAnnotation(tmp_path, "actors", training_cutoff="2001-01-05", horizon="4 days",
        temporary=tmp_path / "temp", model_iterations=2)
    try:
        with pytest.raises(ValueError, match="precede"):
            provider.transform(pd.DataFrame(dict(__event_entity=[1], __event_cutoff=["2001-01-04"])),
                phase_boundary="2001-01-10")
    finally:
        provider.close()


def test_new_response_domain_is_not_silently_filtered_from_features(tmp_path):
    bundle(tmp_path, future_mark=2)
    provider = SourceResponseAnnotation(tmp_path, "actors", training_cutoff="2001-01-05", horizon="4 days",
        temporary=tmp_path / "temp", model_iterations=2)
    try:
        # The later nonbinary mark cannot affect an earlier source boundary.
        provider.transform(pd.DataFrame(dict(__event_entity=[1], __event_cutoff=["2001-01-10"])),
            phase_boundary="2001-01-10")
        with pytest.raises(ChallengerUnavailable, match="not binary"):
            provider.transform(pd.DataFrame(dict(__event_entity=[1], __event_cutoff=["2001-01-11"])),
                phase_boundary="2001-01-11")
    finally:
        provider.close()


def test_explicit_same_clock_annotation_is_exact_but_late_arrival_stays_unsupported(tmp_path):
    bundle(tmp_path)
    requests = pd.DataFrame(dict(__event_entity=[1, 2], __event_cutoff=["2001-01-10"]*2))
    before, supported = extract(tmp_path, requests)
    file = tmp_path / "events.parquet"
    data = pq.read_table(file)
    metadata = dict(data.schema.metadata)
    metadata[b"available_at"] = b'"at"'
    pq.write_table(data.replace_schema_metadata(metadata), file)
    after, supported_again = extract(tmp_path, requests)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    np.testing.assert_array_equal(supported, supported_again)
    metadata[b"available_at"] = b'"arrived_later"'
    pq.write_table(data.replace_schema_metadata(metadata), file)
    with pytest.raises(ValueError, match="delayed source arrival"):
        SourceResponseAnnotation(tmp_path, "actors", training_cutoff="2001-01-05", horizon="4 days",
            temporary=tmp_path / "late-temp", model_iterations=2)
