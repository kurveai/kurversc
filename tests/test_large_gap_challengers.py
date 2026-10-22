import json
import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.benchmark_large_gap_challengers import (
    complete_positive_recipes, isolated_zero_fitter, protected_blend,
)
from scripts.marked_response_history import MarkedResponseHistory, ResponsePathUnavailable
from scripts.zero_aware_quantile_readout import LEVELS, ZeroAwareQuantileReadout, mixture_median


def write_source(path,frame,foreign,clock,key=None):
    table=pa.Table.from_pandas(frame,preserve_index=False)
    table=table.replace_schema_metadata({name.encode():json.dumps(value).encode() for name,value in
        dict(fkey_col_to_pkey_table=foreign,time_col=clock,pkey_col=key).items()})
    pq.write_table(table,path)


def fixture_database(tmp_path):
    db=tmp_path/"db"
    db.mkdir()
    sessions=pd.DataFrame(dict(session=np.arange(12),actor=[1]*10+[2,1],
        created=pd.to_datetime(["2020-01-01"]*11+["2020-01-12"])))
    events=pd.DataFrame(dict(session=np.arange(12),object=[1,2]*6,
        happened=pd.to_datetime([f"2020-01-{day:02d} 12:00" for day in range(1,11)]+["2020-01-11","2020-01-08"],format="mixed"),
        mark=[0,1,0,1,0,1,0,1,0,1,1,1],slot=[1,2,3]*4))
    write_source(db/"sessions.parquet",sessions,dict(actor="actors"),"created","session")
    write_source(db/"events.parquet",events,dict(session="sessions",object="objects"),"happened")
    return db,sessions,events


def test_marked_features_are_bounded_exact_order_and_hide_future_ancestors(tmp_path):
    db,_,_=fixture_database(tmp_path)
    provider=MarkedResponseHistory(db,"actors",training_cutoff="2020-01-10",horizon="2d",temporary=tmp_path/"spill")
    try:
        requests=pd.DataFrame(dict(__event_entity=[1,2,99,1],__event_cutoff=pd.to_datetime(["2020-01-10"]*4)))
        features=provider.transform(requests,phase_boundary="2020-01-10")
        assert features.shape==(4,51)
        assert features.mr_exposures.iloc[0]==9
        assert features.mr_positive_events.iloc[0]==4
        assert features.mr_positive_arrivals.iloc[0]==4
        np.testing.assert_array_equal(provider.support["timing"],[True,False,False,True])
        pd.testing.assert_series_equal(features.iloc[0],features.iloc[3],check_names=False)
        assert features.mr_exposures.iloc[2]==0
        assert all(not name.endswith(("actor","object","session","mark","slot")) for name in features)
        # A later query clock increases age, never source availability.
        later=provider.transform(requests.assign(__event_cutoff=pd.Timestamp("2020-01-11")),phase_boundary="2020-01-10")
        assert later.mr_exposures.iloc[0]==9
        assert later.mr_positive_recency.iloc[0]>features.mr_positive_recency.iloc[0]
    finally:
        provider.close()


def test_direct_root_route_and_subday_cutoff_exclude_equal_time(tmp_path):
    db,_,_=fixture_database(tmp_path)
    provider=MarkedResponseHistory(db,"objects",training_cutoff="2020-01-10",horizon="2d",temporary=tmp_path/"spill")
    try:
        q=pd.DataFrame(dict(__event_entity=[1],__event_cutoff=pd.to_datetime(["2020-01-09 12:00"])))
        # Provider profiling is fixed earlier than every request.
        with pytest.raises(ValueError,match="Authorized"):
            provider.transform(q,phase_boundary="2020-01-10")
    finally:
        provider.close()
    provider=MarkedResponseHistory(db,"objects",training_cutoff="2020-01-05",horizon="2d",temporary=tmp_path/"spill2")
    try:
        out=provider.transform(q,phase_boundary="2020-01-10")
        assert out.mr_exposures.iloc[0]==4  # Jan9 equals the query clock and is excluded.
        assert provider.path.ancestor is None
    finally:
        provider.close()


def test_future_marks_cannot_change_role_discovery_or_visible_features(tmp_path):
    db,sessions,events=fixture_database(tmp_path)
    q=pd.DataFrame(dict(__event_entity=[1],__event_cutoff=pd.to_datetime(["2020-01-10"])))
    def run():
        p=MarkedResponseHistory(db,"actors",training_cutoff="2020-01-10",horizon="2d",temporary=tmp_path/"spill")
        try:
            return p.path,p.transform(q,phase_boundary="2020-01-10")
        finally:
            p.close()
    path,before=run()
    sessions=pd.concat([sessions,pd.DataFrame(dict(session=[100],actor=[1],created=pd.to_datetime(["2020-01-13"])))],ignore_index=True)
    events=pd.concat([events,pd.DataFrame(dict(session=[100],object=[1],happened=pd.to_datetime(["2020-01-13"]),mark=[100],slot=[100]))],ignore_index=True)
    write_source(db/"sessions.parquet",sessions,dict(actor="actors"),"created","session")
    write_source(db/"events.parquet",events,dict(session="sessions",object="objects"),"happened")
    new_path,after=run()
    assert path==new_path
    pd.testing.assert_frame_equal(before,after)


def test_duplicate_ancestor_key_is_rejected_before_feature_join(tmp_path):
    db,sessions,_=fixture_database(tmp_path)
    write_source(db/"sessions.parquet",pd.concat([sessions,sessions.iloc[:1]]),dict(actor="actors"),"created","session")
    with pytest.raises(ValueError,match="fanout"):
        MarkedResponseHistory(db,"actors",training_cutoff="2020-01-10",horizon="2d",temporary=tmp_path/"spill")


def test_schema_without_response_is_inapplicable_not_task_specific(tmp_path):
    db=tmp_path/"db"
    db.mkdir()
    write_source(db/"plain.parquet",pd.DataFrame(dict(actor=[1,2],time=pd.date_range("2020-01-01",periods=2))),dict(actor="actors"),"time")
    with pytest.raises(ResponsePathUnavailable):
        MarkedResponseHistory(db,"actors",training_cutoff="2020-01-10",horizon="2d",temporary=tmp_path/"spill")


def test_mixture_median_zero_mass_and_conditional_quantile_are_correct():
    q=np.tile(np.log1p([1,2,4,8,16]),(3,1))
    out=mixture_median([.25,.5,1.],q)
    np.testing.assert_allclose(out,[0,0,4],rtol=0,atol=1e-12)
    p=.5/(1-.25)
    assert abs(mixture_median([p],q[:1])[0]-2)<1e-12
    with pytest.raises(ValueError):
        mixture_median([1.1],q[:1])


def test_small_zero_readout_is_reproducible_serializable_and_training_only():
    x=pd.DataFrame(dict(value=np.arange(100)%7,category=["a","b"]*50))
    y=np.tile([0,0,1,2,9],20)
    model=ZeroAwareQuantileReadout(seed=0,model_params=dict(iterations=3,depth=2,thread_count=1)).fit(x,y,categorical=[1])
    pred=model.predict(x.iloc[:8])
    assert (pred>=0).all()
    np.testing.assert_array_equal(pred,pickle.loads(pickle.dumps(model)).predict(x.iloc[:8]))
    assert model.audit["quantiles"]==LEVELS.tolist()
    with pytest.raises(ValueError):
        ZeroAwareQuantileReadout().fit(x,-y,categorical=[1])


def test_failed_factory_restores_incumbent_estimator_and_unsupported_rows_are_exact():
    from kurversc import temporal_policy
    original=temporal_policy.fit_final_catboost
    with pytest.raises(RuntimeError):
        with isolated_zero_fitter():
            assert temporal_policy.fit_final_catboost is not original
            raise RuntimeError("simulated worker failure")
    assert temporal_policy.fit_final_catboost is original
    control=np.array([.1,.4,.6])
    candidate=np.array([.9,.2,.8])
    pred=protected_blend(control,candidate,[True,False,True],.5)
    assert pred[1]==control[1]
    np.testing.assert_allclose(pred[[0,2]],[.5,.7])


def test_incomplete_or_regressing_validation_can_never_promote():
    rows=[dict(fold=f,seed=s,gains=dict(good=.001,bad=-.00001)) for f in range(2) for s in range(3)]
    assert complete_positive_recipes(rows,["good","bad"],[0,1])==["good"]
    assert complete_positive_recipes(rows[:-1],["good"],[0,1])==[]
    with pytest.raises(ValueError,match="Duplicate"):
        complete_positive_recipes(rows+rows[:1],["good"],[0,1])
