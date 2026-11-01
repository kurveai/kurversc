"""Boundary and dependency checks for the revised generic submission."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import kurversc
from kurversc.core import _materialize_at_cutoff, _table_workspace
from kurversc.task_context import ContextDatabase, ContextTable
from scripts.benchmark_generic_readouts import CONTRACT, RECIPES, resolve_task_metadata


@pytest.mark.parametrize("compatibility",[False,True])
def test_strict_sources_exclude_equality_without_separate_availability_clock(compatibility):
    cutoff=pd.Timestamp("2020-02-01 12:34:56")
    root=kurversc.Table(pd.DataFrame({"uid":[0,1]}),name="people",key="uid",timeless=True,prefix="p")
    events=kurversc.Table(pd.DataFrame({"uid":[0,0,0,1,1,1],
        "time":[cutoff-pd.Timedelta(seconds=1),cutoff,cutoff+pd.Timedelta(seconds=1)]*2,
        "amount":[1.,1000.,1e6,2.,2000.,2e6]}),
        name="events",date="time",prefix="e",cutoff_boundary="strict")
    labels=pd.DataFrame({"uid":[0,1],"time":cutoff,"target":[0,1]})
    spec=kurversc.Labels(labels,key="uid",timestamp="time",target="target",target_history=False)
    tables={"people":root,"events":events}
    config=kurversc.GraphConfig(feature_families=("base",),auto_annotate_features=False,
        include_target_history=False,base_distribution_features=False,
        published_cutoff_compatibility=compatibility)
    with _table_workspace(None,tables,sample_rows=100,random_state=0,search=False,
            search_full_data=True,duckdb_memory_limit="1GB",duckdb_max_temp_directory_size="1GB") as workspace:
        frame=_materialize_at_cutoff(workspace,tables,
            (kurversc.Relationship("people","events","uid","uid"),),"people",config,
            labels,spec,cut_date=cutoff,compute_period_days=365,verbose=False)
    means=[c for c in frame if c.startswith("e_amount") and ("mean" in c or "avg" in c)]
    assert means
    np.testing.assert_allclose(frame[means[0]],[1.,2.])


def test_phase_censoring_honors_event_and_availability_boundaries_independently():
    cutoff=pd.Timestamp("2020-02-01")
    table=ContextTable(pd.DataFrame({"id":[0,1,2],
        "event":[cutoff-pd.Timedelta(seconds=1),cutoff,cutoff-pd.Timedelta(seconds=1)],
        "known":[cutoff,cutoff,cutoff+pd.Timedelta(seconds=1)]}),
        pkey_col="id",time_col="event",available_col="known",cutoff_boundary="strict")
    db=ContextDatabase({"arbitrary":table}).upto(cutoff)
    assert db.table_dict["arbitrary"].df.id.tolist()==[0]
    assert len(table.df)==3


def test_strict_root_boundary_keeps_all_forecast_requests_with_missing_payload():
    cutoff=pd.Timestamp("2020-02-01")
    root=kurversc.Table(pd.DataFrame({"uid":[0,1],
        "created":[cutoff-pd.Timedelta(days=1),cutoff],"payload":[7,999]}),
        name="people",key="uid",date="created",prefix="p",cutoff_boundary="strict")
    query=pd.DataFrame({"uid":[1,0],"when":cutoff,"__kurversc_prediction_order__":[0,1]})
    spec=kurversc.Labels(query,key="uid",timestamp="when",target_history=False)
    config=kurversc.GraphConfig(feature_families=("base",),auto_annotate_features=False,
        include_target_history=False,base_distribution_features=False)
    with _table_workspace(None,{"people":root},sample_rows=100,random_state=0,
            search=False,search_full_data=True,duckdb_memory_limit="1GB",
            duckdb_max_temp_directory_size="1GB") as workspace:
        frame=_materialize_at_cutoff(workspace,{"people":root},(),"people",config,
            query,spec,cut_date=cutoff,compute_period_days=365,verbose=False,train=False)
    assert frame.uid.tolist()==[1,0]
    payload=next(c for c in frame if c.endswith("payload"))
    assert pd.isna(frame.loc[0,payload])
    assert frame.loc[1,payload]==7


def test_candidate_menu_has_only_accepted_components_and_one_global_gate():
    assert set(RECIPES)=={"text25","text50","frame_ensemble3","frame_ensemble5",
        "nonnegative","integer_counts","object_full","object_half",
        "zero_aware_0.125","zero_aware_0.5","zero_aware_1"}
    assert CONTRACT["selector"]=="select_temporal_policy"
    assert CONTRACT["eligibility_depends_on_previous_admissions"] is False
    assert CONTRACT["benchmark_score_dependent_choices"] is False


@pytest.mark.parametrize("late",[False,True])
def test_readout_root_comes_from_task_contract_without_requiring_a_label_clock(tmp_path,monkeypatch,late):
    from test_task_context import context
    from kurversc.task_source import CONTEXT_ENV,ContextTaskSource
    path=context(late=late).export(tmp_path/"context")
    monkeypatch.setenv(CONTEXT_ENV,str(path))
    source=ContextTaskSource(path)
    metadata=dict(dataset=source.manifest["dataset"],task=source.manifest["task"])
    resolved=resolve_task_metadata(metadata)
    assert resolved["entity_table"]==source.task.entity_table
    assert "entity_table" not in metadata
    with pytest.raises(ValueError,match="declared task root"):
        resolve_task_metadata({**metadata,"entity_table":"wrong_root"})
    with pytest.raises(ValueError,match="different task context"):
        resolve_task_metadata({**metadata,"task":"wrong_task"})
    monkeypatch.delenv(CONTEXT_ENV)
    with pytest.raises(ValueError,match="explicit task context"):
        resolve_task_metadata(metadata)


def test_procedure_resume_contract_matches_its_persisted_representation(tmp_path):
    from scripts.run_portable_staged import procedure_contract
    live=procedure_contract()
    path=tmp_path/"procedure.json"
    path.write_text(json.dumps(live))
    assert json.loads(path.read_text())==procedure_contract()
    assert isinstance(live["generic_readouts"]["candidates"],list)
