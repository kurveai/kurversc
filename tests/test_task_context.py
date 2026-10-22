import json

import numpy as np
import pandas as pd
import pytest

from kurversc.task_context import PredictionTask, SourceTable, TaskContext
from kurversc.task_source import ContextTaskSource, load_task_source


def context(*, composite=False, late=False, renamed=False):
    root = "accounts with spaces" if renamed else "actors"
    key = ("tenant","uuid") if composite else "uuid"
    ids = ["account-a","account-b","account-c","account-d"]
    parents = pd.DataFrame(dict(uuid=ids,tenant=["east"]*4,value=[2.,4.,6.,8.]))
    events = pd.DataFrame(dict(uuid=ids*2,tenant=["east"]*8,amount=np.arange(8),
        happened=pd.to_datetime(["2020-01-01"]*4+["2020-05-01"]*4),
        learned=pd.to_datetime(["2020-02-15"]*4+["2020-05-01"]*4)))
    train = pd.DataFrame(dict(uuid=ids,tenant=["east"]*4,when=pd.Timestamp("2020-01-01"),outcome=[0,1,0,1]))
    val = train.assign(when=pd.Timestamp("2020-03-01"))
    query = train.drop(columns="outcome").assign(when=pd.Timestamp("2020-06-01"))
    if late:
        train["known"] = pd.Timestamp("2020-02-20")
        val["known"] = pd.Timestamp("2020-04-20")
    tables = {root:SourceTable(parents,primary_key=key,timeless=True),
        "activity":SourceTable(events,foreign_keys={key:root},timestamp="happened",available_at="learned")}
    task = PredictionTask(root,key,"when","outcome","30d",label_available_at="known" if late else None,
        name="previously unseen problem")
    return TaskContext.from_tables(tables=tables,task=task,training=train,validation=val,requests=query,
        inner_cutoff="2020-02-28",outer_cutoff="2020-05-30",dataset_name="unregistered-database")


@pytest.mark.parametrize("composite",[False,True])
def test_new_schema_string_and_composite_keys_share_routing_without_payload_leakage(composite):
    c = context(composite=composite)
    inner,outer = c.split("inner"),c.split("outer")
    assert set(inner.db_state.table_dict)=={"table0","table1"}
    root,child = inner.db_state.table_dict.values()
    assert child.df[next(iter(child.fkey_col_to_pkey_table))].isin(root.df[root.pkey_col]).all()
    assert c.requests[c.task.entity_key].is_unique
    assert pd.api.types.is_integer_dtype(c.requests[c.task.entity_key].dtype)
    assert len(child.df)==4 and len(outer.db_state.table_dict["table1"].df)==8
    assert "outcome" not in outer.eval_table.df
    assert c.request_keys.uuid.tolist()==["account-a","account-b","account-c","account-d"]


def test_routing_ids_do_not_depend_on_future_entities_or_physical_row_order():
    from kurversc.task_context import _canonical_tables
    base = pd.DataFrame(dict(uid=["uuid-b","uuid-a"]))
    tables = {"root":SourceTable(base,primary_key="uid",timeless=True)}
    _,mapping,_ = _canonical_tables(tables)
    expanded = pd.DataFrame(dict(uid=["uuid-0-future","uuid-a","uuid-b"]))
    _,other,_ = _canonical_tables({"root":SourceTable(expanded,primary_key="uid",timeless=True)})
    assert all(mapping["root"][k]==other["root"][k] for k in mapping["root"])


def test_future_uint64_outlier_does_not_remap_existing_native_integer_ids():
    from kurversc.task_context import _canonical_tables
    base = pd.DataFrame(dict(uid=pd.Series([1,2],dtype="uint64")))
    expanded = pd.DataFrame(dict(uid=pd.Series([1,2,2**63+17],dtype="uint64")))
    _,before,_ = _canonical_tables({"root":SourceTable(base,primary_key="uid",timeless=True)})
    _,after,_ = _canonical_tables({"root":SourceTable(expanded,primary_key="uid",timeless=True)})
    assert before["root"][(1,)]==after["root"][(1,)]==1
    assert before["root"][(2,)]==after["root"][(2,)]==2
    assert 0<=after["root"][(2**63+17,)]<=np.iinfo("int64").max


def test_table_renaming_does_not_change_canonical_schema_or_payload():
    a,b = context(),context(renamed=True)
    assert a.task==b.task
    for name,table in a.database.table_dict.items():
        pd.testing.assert_frame_equal(table.df,b.database.table_dict[name].df)


def test_phase_and_query_clocks_do_not_make_late_arrivals_visible_early():
    c = context(late=True)
    assert len(c.database.upto("2020-02-01").table_dict["table1"].df)==0
    assert len(c.database.upto("2020-02-20").table_dict["table1"].df)==4
    assert c.observed_before(c.training,"2020-02-01").empty
    assert len(c.observed_before(c.training,"2020-02-21"))==4
    assert not hasattr(c.split("outer"),"eval_target")


def test_checksums_bind_every_stage_to_explicit_source_not_a_registry(tmp_path,monkeypatch):
    c = context(composite=True)
    file = c.export(tmp_path/"bundle")
    monkeypatch.setenv("KURVERSC_TASK_CONTEXT",str(file))
    source = load_task_source("unregistered-database","previously unseen problem")
    assert isinstance(source,ContextTaskSource)
    assert source.task.entity_col=="__kurversc_entity_key"
    assert "outcome" not in source.outer_split().eval_table.df
    assert np.isnan(source.task.evaluate(np.ones(4))["roc_auc"])
    target = source.inner_split().eval_target
    assert source.task.evaluate(target.df.outcome.to_numpy(dtype=float),target)["roc_auc"]==1.
    with pytest.raises(ValueError,match="does not match"):
        load_task_source("rel-f1","driver-dnf")
    with pytest.raises(FileExistsError):
        c.export(tmp_path/"bundle")
    data = tmp_path/"bundle/inner/train_table.parquet"
    data.write_bytes(data.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="changed"):
        load_task_source("unregistered-database","previously unseen problem")


def test_forecast_targets_and_unknown_foreign_keys_fail_closed():
    c = context()
    tables = {"actors":SourceTable(pd.DataFrame(dict(id=[1,2])),primary_key="id",timeless=True),
        "events":SourceTable(pd.DataFrame(dict(uid=[3],date=[pd.Timestamp("2020-01-01")])),
            foreign_keys={"uid":"actors"},timestamp="date")}
    frame = pd.DataFrame(dict(uid=[1,2],date=pd.Timestamp("2020-01-01"),target=[0,1]))
    common = dict(tables=tables,task=PredictionTask("actors","uid","date","target","30d"),
        training=frame,validation=frame.assign(date=pd.Timestamp("2020-03-01")),
        inner_cutoff="2020-02-01",outer_cutoff="2020-05-01")
    with pytest.raises(ValueError,match="held-out"):
        TaskContext.from_tables(**common,requests=frame.assign(date=pd.Timestamp("2020-06-01")))
    with pytest.raises(ValueError,match="unknown"):
        TaskContext.from_tables(**common,requests=frame.drop(columns="target").assign(date=pd.Timestamp("2020-06-01")))


def test_bundle_rejects_files_outside_its_declared_root(tmp_path):
    file = context().export(tmp_path/"bundle")
    manifest = json.loads(file.read_text())
    manifest["files"]["../secret"] = "x"
    file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="escaped"):
        ContextTaskSource(file)


def test_declared_event_boundary_survives_phase_copies_and_export(tmp_path):
    c = context()
    c.database.table_dict["table1"].cutoff_boundary="inclusive"
    source = ContextTaskSource(c.export(tmp_path/"inclusive"))
    assert source.inner_split().db_state.table_dict["table1"].cutoff_boundary=="inclusive"
    assert source.outer_split().db_state.table_dict["table1"].cutoff_boundary=="inclusive"
