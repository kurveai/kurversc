import copy
from types import SimpleNamespace

import pandas as pd
import pytest

from kurversc import GraphConfig
from kurversc.relbench import _edge_instances
from kurversc.schema_blueprint import PLAN_KEY,occurrence_bound,source_blueprint,replay_limits,verify_replay


def graph(bound):
    entities=pd.DataFrame({'key':[1,2],'measure':[10.,20.]})
    events=pd.DataFrame({'key':[10,11],'owner':[1,2],'neighbor':[2,1],'amount':[1.,2.],
        'clock':pd.Timestamp('2020-01-01')})
    tables={'entities':SimpleNamespace(df=entities,pkey_col='key',time_col=None,fkey_col_to_pkey_table={}),
        'events':SimpleNamespace(df=events,pkey_col='key',time_col='clock',
            fkey_col_to_pkey_table={'owner':'entities','neighbor':'entities'})}
    parent,children,edges=_edge_instances(SimpleNamespace(table_dict=tables),root_table='entities',
        root_frame=entities,search_root_frame=entities,schema_depth=2,sample_rows=None,random_state=0,
        max_base_table_occurrences=bound)
    return {parent.name:parent,**{table.name:table for table in children}},edges


def test_adapter_superset_must_cover_the_actual_candidate_occurrence_limits():
    assert occurrence_bound(configs=[GraphConfig(max_base_table_occurrences=1)],options=(1,2))==2
    assert occurrence_bound(configs=[GraphConfig(max_base_table_occurrences=3)],options=(1,2))==3
    assert occurrence_bound()==1
    legacy,_=graph(1);expanded,_=graph(2)
    assert all(len(set(table.path_base_tables))==len(table.path_base_tables) for table in legacy.values())
    assert any(table.path_base_tables==('entities','events','entities') for table in expanded.values())
    assert len(expanded)>len(legacy)


def test_frozen_superset_limits_survive_selection_of_a_smaller_graph():
    tables,edges=graph(2);blueprint=source_blueprint(tables,edges,'entities')
    assert replay_limits({PLAN_KEY:blueprint},depth=4,occurrences=1)==(2,2)
    assert replay_limits(None,depth=4,occurrences=1)==(4,1)
    verify_replay({PLAN_KEY:blueprint},source_blueprint(tables,edges,'entities'))
    legacy,edges=graph(1)
    with pytest.raises(ValueError,match='source blueprint'):
        verify_replay({PLAN_KEY:blueprint},source_blueprint(legacy,edges,'entities'))
    changed=copy.deepcopy(blueprint);changed['instances'][1]['prefix']='different_alias'
    with pytest.raises(ValueError,match='source blueprint'):verify_replay({PLAN_KEY:blueprint},changed)


def test_staged_retention_passes_its_declared_search_bound_to_the_adapter(monkeypatch):
    import kurversc
    from scripts.run_kurversc_selected_relarena import AuditWinnerSystem
    class StopBeforeFitting(Exception):pass
    captured={}
    def build(*args,**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(fit_kwargs=lambda:{})
    def fit(**kwargs):raise StopBeforeFitting()
    monkeypatch.setattr(kurversc,'relbench_problem_from_objects',build)
    monkeypatch.setattr(kurversc,'fit',fit)
    task=SimpleNamespace(task_type='binary_classification',time_col='time',target_col='target')
    frame=pd.DataFrame({'time':pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03']*100),
        'target':[0,1]*150})
    system=AuditWinnerSystem(run_identity=SimpleNamespace(dataset='unseen_database',task='unseen_task'))
    system.profile='staged-retention';system.peer_context_mode=None
    with pytest.raises(StopBeforeFitting):
        system._fit_arm(task,None,SimpleNamespace(df=frame),SimpleNamespace(df=frame),seed=0,phase='inner')
    assert captured['max_base_table_occurrences']==2
    assert captured['schema_depth']==4


def test_execution_fingerprint_binds_source_aliases_and_legacy_plans_remain_valid():
    from kurversc.core import _execution_plan_fingerprint,_freeze_execution_plan
    tables,edges=graph(2);blueprint=source_blueprint(tables,edges,'entities')
    graph_stub=SimpleNamespace(freeze_execution_plan=lambda:{'records':[]},nodes=lambda:[],
        _kurversc_source_blueprint=blueprint)
    frozen=_freeze_execution_plan(graph_stub,pd.Timestamp('2020-01-01'))
    assert frozen[PLAN_KEY]==blueprint
    changed=copy.deepcopy(frozen);changed[PLAN_KEY]['instances'][1]['prefix']='other'
    assert _execution_plan_fingerprint(frozen)!=_execution_plan_fingerprint(changed)
    verify_replay({'records':[]},blueprint)


def test_acyclic_selected_model_replays_its_expanded_blueprint_without_alias_drift():
    import numpy as np
    import kurversc
    tables,edges=graph(2)
    labels=pd.DataFrame({'key':[1,2,1,2],'target':[0,1,0,1],
        'when':pd.to_datetime(['2020-01-02']*2+['2020-01-03']*2),
        'split':['train']*2+['validation']*2})
    result=kurversc.fit(parent_node=tables['entities'],
        label_node=kurversc.Labels(labels,key='key',target='target',timestamp='when',split='split'),
        tables=[table for name,table in tables.items() if name!='entities'],relationships=edges,
        graph_configs=[GraphConfig(depth=1,max_base_table_occurrences=1,feature_families=('base',))],
        full_training_frames=1,rerank_top_k=0,search_full_data=True,infer_ts_periods=False,
        model_params={'iterations':2,'depth':2,'thread_count':1})
    assert result.best_config.max_base_table_occurrences==1
    assert result.execution_plan[PLAN_KEY]['max_base_table_occurrences']==2
    prediction=kurversc.Labels(pd.DataFrame({'key':[1,2],'when':pd.Timestamp('2020-01-04')}),key='key',timestamp='when')
    def predict(source,relations):
        return kurversc.predict(result,parent_node=source['entities'],prediction_node=prediction,
            tables=[table for name,table in source.items() if name!='entities'],relationships=relations,
            use_validation_model=True).prediction.to_numpy()
    expected=predict(tables,edges)
    wrong,wrong_edges=graph(1)
    with pytest.raises(ValueError,match='source blueprint'):predict(wrong,wrong_edges)
    _,bound=replay_limits(result.execution_plan,depth=2,occurrences=1)
    restored,restored_edges=graph(bound)
    np.testing.assert_array_equal(predict(restored,restored_edges),expected)
