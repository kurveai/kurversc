from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.benchmark_production_sequence import applicability, channel_availability, delayed_channel


def context():
    return (dict(metadata=dict(metric="roc_auc", atomic_date_replay=True), folds=[1, 2], seeds=[0, 1, 2]),
        dict(name="incumbent", blend_weight=1., readout="direct", mode="frame_ensemble"),
        [dict(selected="incumbent"), dict(selected="incumbent")])


def test_no_task_or_schema_name_gate():
    assert applicability(*context()) is None


def test_stronger_readouts_are_exact_fallbacks():
    raw, policy, selections = context()
    selections[1]["selected"] = "joint_semantic32"
    assert applicability(raw, policy, selections) == "preserve_already_admitted_signal_or_readout"


def test_composite_native_is_not_silently_replaced():
    raw, policy, selections = context()
    policy["blend_weight"] = .125
    assert applicability(raw, policy, selections) == "preserve_composite_native_procedure"


def test_all_available_evidence_is_required():
    raw, policy, selections = context()
    raw["seeds"] = [0]
    assert applicability(raw, policy, selections) == "insufficient_available_temporal_evidence"
    raw, policy, selections = context()
    raw["metadata"]["metric"] = "mae"
    assert applicability(raw, policy, selections) == "not_binary_auc"


def test_availability_is_checked_with_legacy_channel_metadata(tmp_path):
    path = tmp_path / "events.parquet"
    table = pa.table(dict(key=[1], event=["2001-01-01"], visible=["2001-01-02"]))
    pq.write_table(table.replace_schema_metadata({b"available_at": b'"visible"'}), path)
    assert channel_availability(SimpleNamespace(path=path)) == "visible"
    assert delayed_channel(SimpleNamespace(path=path, timestamp="event"))
    assert not delayed_channel(SimpleNamespace(path=path, timestamp="visible"))
