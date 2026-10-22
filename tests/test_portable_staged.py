import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import kurversc
from kurversc.core import _target_history_frame, _materialize_at_cutoff, _table_workspace
from kurversc.specs import TARGET_HISTORY_TIME
from kurversc.task_source import ContextTaskSource, phase_source_root
from scripts.run_kurversc_selected_relarena import _recipe_for
from scripts.run_portable_staged import prediction_artifact, require_supported_availability, remaining_context_budget
from test_task_context import context


def test_generic_search_recipe_never_reads_historical_benchmark_controls(monkeypatch):
    from scripts import run_kurversc_selected_relarena as module
    monkeypatch.setattr(module,"PUBLISHED_REFERENCES",{})
    monkeypatch.setattr(module,"RECIPES",{})
    recipe = _recipe_for(("not-in-relbench","unseen-task"),generic=True,task_type="binary_classification")
    assert recipe.published_score is None
    assert recipe.higher_is_better
    assert recipe.config==kurversc.GraphConfig()


def test_history_uses_observation_time_not_target_window_start():
    labels = pd.DataFrame(dict(uid=[0,1],when=pd.to_datetime(["2020-01-01"]*2),
        known=pd.to_datetime(["2020-02-01","2020-04-01"]),outcome=[0,1]))
    spec = kurversc.Labels(labels,key="uid",timestamp="when",target="outcome",available_at="known")
    history = _target_history_frame(labels,spec)
    np.testing.assert_array_equal(history[TARGET_HISTORY_TIME],labels.known)
    assert "known" not in history and "when" not in history


def test_event_windows_and_arrival_visibility_are_independent_before_annotation():
    root = kurversc.Table(pd.DataFrame(dict(uid=[0,1])),name="people",key="uid",timeless=True,prefix="person")
    child = kurversc.Table(pd.DataFrame(dict(uid=[0,0,1,1],
        event_time=pd.to_datetime(["2020-01-01"]*4),
        arrived=pd.to_datetime(["2020-01-02","2020-03-01"]*2),amount=[10.,1000.,20.,2000.])),
        name="activity",date="event_time",available_at="arrived",prefix="activity")
    labels = pd.DataFrame(dict(uid=[0,1],when=pd.Timestamp("2020-02-01"),outcome=[0,1]))
    spec = kurversc.Labels(labels,key="uid",timestamp="when",target="outcome",target_history=False)
    tables = {"people":root,"activity":child}
    relations = (kurversc.Relationship("people","activity","uid","uid"),)
    config = kurversc.GraphConfig(feature_families=("base",),auto_annotate_features=False,
        include_target_history=False,base_distribution_features=False)
    with _table_workspace(None,tables,sample_rows=100,random_state=0,search=False,
            search_full_data=True,duckdb_memory_limit="1GB",duckdb_max_temp_directory_size="1GB") as workspace:
        result = _materialize_at_cutoff(workspace,tables,relations,"people",config,labels,spec,
            cut_date=pd.Timestamp("2020-02-01"),compute_period_days=365,verbose=False)
    amounts = [c for c in result if c.startswith("activity_amount")]
    assert amounts
    assert result[amounts].max().max()<=20
    assert not any("arrived" in c for c in result)


def test_explicit_event_boundary_cannot_be_shifted_by_legacy_offset_semantics():
    root = kurversc.Table(pd.DataFrame(dict(uid=[0,1])),name="people",key="uid",timeless=True,prefix="person")
    cutoff = pd.Timestamp("2020-02-01 12:00")
    child = kurversc.Table(pd.DataFrame(dict(uid=[0,0,1,1],
        event_time=[cutoff-pd.Timedelta("1h"),cutoff]*2,
        amount=[10.,1000.,20.,2000.])),name="activity",date="event_time",available_at="event_time",
        prefix="activity",cutoff_boundary="strict")
    labels = pd.DataFrame(dict(uid=[0,1],when=cutoff,outcome=[0,1]))
    spec = kurversc.Labels(labels,key="uid",timestamp="when",target="outcome",target_history=False)
    tables = {"people":root,"activity":child}
    config = kurversc.GraphConfig(feature_families=("base",),auto_annotate_features=False,
        published_cutoff_compatibility=True,include_target_history=False,base_distribution_features=False)
    with _table_workspace(None,tables,sample_rows=100,random_state=0,search=False,
            search_full_data=True,duckdb_memory_limit="1GB",duckdb_max_temp_directory_size="1GB") as workspace:
        result = _materialize_at_cutoff(workspace,tables,
            (kurversc.Relationship("people","activity","uid","uid"),),"people",config,labels,spec,
            cut_date=cutoff,compute_period_days=365,verbose=False)
    amounts = [c for c in result if c.startswith("activity_amount")]
    assert amounts and result[amounts].max().max()<=20


def test_explicit_clock_also_anchors_lookback_windows_without_compatibility_offset():
    root = kurversc.Table(pd.DataFrame(dict(uid=[0,1])),name="people",key="uid",timeless=True,prefix="person")
    cutoff = pd.Timestamp("2020-02-01 12:00")
    child = kurversc.Table(pd.DataFrame(dict(uid=[0,1],
        event_time=[cutoff-pd.Timedelta("1d")+pd.Timedelta(microseconds=1)]*2,amount=[10.,20.])),
        name="activity",date="event_time",available_at="event_time",prefix="activity",cutoff_boundary="strict")
    labels = pd.DataFrame(dict(uid=[0,1],when=cutoff,outcome=[0,1]))
    spec = kurversc.Labels(labels,key="uid",timestamp="when",target="outcome",target_history=False)
    tables = {"people":root,"activity":child}
    config = kurversc.GraphConfig(feature_families=("base",),auto_annotate_features=False,
        published_cutoff_compatibility=True,include_target_history=False,base_distribution_features=False)
    with _table_workspace(None,tables,sample_rows=100,random_state=0,search=False,
            search_full_data=True,duckdb_memory_limit="1GB",duckdb_max_temp_directory_size="1GB") as workspace:
        result = _materialize_at_cutoff(workspace,tables,
            (kurversc.Relationship("people","activity","uid","uid"),),"people",config,labels,spec,
            cut_date=cutoff,compute_period_days=1,verbose=False)
    amounts = [c for c in result if c.startswith("activity_amount")]
    assert amounts and result[amounts].notna().all().all()


def test_optional_file_providers_receive_separate_authorized_phases(tmp_path,monkeypatch):
    file = context(composite=True).export(tmp_path/"context")
    monkeypatch.setenv("KURVERSC_TASK_CONTEXT",str(file))
    inner = phase_source_root("unregistered-database",phase="inner")
    outer = phase_source_root("unregistered-database",phase="outer")
    assert inner!=outer
    assert len(pd.read_parquet(inner/"db/table1.parquet"))==4
    assert len(pd.read_parquet(outer/"db/table1.parquet"))==8
    assert not (inner/"tasks/previously unseen problem/test.parquet").exists()
    source = ContextTaskSource(file)
    require_supported_availability(source)
    assert phase_source_root("unregistered-database",phase="inner")==inner
    file = inner/"db/table1.parquet"
    file.write_bytes(file.read_bytes()+b"modified")
    with pytest.raises(ValueError,match="snapshot changed"):
        phase_source_root("unregistered-database",phase="inner")


def test_partial_provider_export_is_not_mistaken_for_a_finished_snapshot(tmp_path,monkeypatch):
    file = context().export(tmp_path/"context")
    monkeypatch.setenv("KURVERSC_TASK_CONTEXT",str(file))
    output = tmp_path/"partial"
    output.mkdir()
    (output/"unfinished.parquet").write_bytes(b"partial")
    with pytest.raises(ValueError,match="Incomplete"):
        phase_source_root("unregistered-database",directory=output)


def test_resume_never_resets_the_original_deadline_or_excludes_preparation(tmp_path):
    assert remaining_context_budget(tmp_path,hours=1,preprocessing_seconds=60,now=1000)==3540
    assert remaining_context_budget(tmp_path,hours=1,preprocessing_seconds=0,now=1100)==3440
    with pytest.raises(ValueError,match="reset"):
        remaining_context_budget(tmp_path,hours=2,preprocessing_seconds=0,now=1100)
    with pytest.raises(TimeoutError,match="exhausted"):
        remaining_context_budget(tmp_path,hours=1,preprocessing_seconds=0,now=4541)


def test_portable_stage_preserves_resolved_runtime_not_a_sibling_checkout(monkeypatch):
    from scripts.run_temporal_policy_campaign import stage_environment
    monkeypatch.setenv("KURVERSC_TASK_CONTEXT","/explicit/context.json")
    monkeypatch.setenv("PYTHONPATH","/pinned/runtime:/pinned/graphreduce")
    assert stage_environment()["PYTHONPATH"]=="/pinned/runtime:/pinned/graphreduce"


def test_reliability_readout_uses_context_root_without_a_benchmark_registry():
    from scripts.benchmark_reliability_blending import declared_root
    assert declared_root(dict(entity_table="novel_root"))=="novel_root"


def test_variable_label_delay_is_explicitly_carried_to_readouts(tmp_path):
    source = ContextTaskSource(context(late=True).export(tmp_path/"late"))
    require_supported_availability(source)
    assert source.task.label_available_at=="__kurversc_label_available_at"
    assert source.mature_labels(source.inner_split().train_table.df,"2020-02-01").empty


def test_prediction_export_follows_the_admitted_procedure_not_file_discovery(tmp_path):
    task = tmp_path/"workers/arbitrary_task/temporal/tasks/arbitrary_task"
    count = task/"conditional-count"
    count.mkdir(parents=True)
    (count/"admission.json").write_text(json.dumps(dict(admitted=False)))
    assert prediction_artifact(tmp_path,"arbitrary_task")==task/"recovered-readouts/predictions.npz"
    (count/"admission.json").write_text(json.dumps(dict(admitted=True)))
    assert prediction_artifact(tmp_path,"arbitrary_task")==count/"official/predictions.npz"
    sequence = task/"production-sequence"
    sequence.mkdir()
    (sequence/"admission.json").write_text(json.dumps(dict(admitted=False)))
    assert prediction_artifact(tmp_path,"arbitrary_task")==count/"official/predictions.npz"
    (sequence/"admission.json").write_text(json.dumps(dict(admitted=True)))
    assert prediction_artifact(tmp_path,"arbitrary_task")==sequence/"official/predictions.npz"
    bounded = task/"bounded-views"
    bounded.mkdir()
    (bounded/"admission.json").write_text(json.dumps(dict(admitted=False)))
    assert prediction_artifact(tmp_path,"arbitrary_task")==sequence/"official/predictions.npz"
    (bounded/"admission.json").write_text(json.dumps(dict(admitted=True)))
    assert prediction_artifact(tmp_path,"arbitrary_task")==bounded/"official/predictions.npz"
