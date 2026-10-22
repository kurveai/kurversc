import numpy as np
import pandas as pd
import pytest

from kurversc.modeling import prepare_prediction_features
from kurversc.temporal_policy import TemporalFitPolicy,fit_temporal_policy,policy_support
from scripts.activity_timeline_challenger import mature_label_state


def test_temporal_fits_do_not_use_results_that_arrive_after_the_initial_or_final_boundary(monkeypatch):
    from kurversc import temporal_policy as module
    times = pd.Series(pd.to_datetime(["2020-01-01","2020-02-01","2020-03-01","2020-04-01"]).repeat(4)).reset_index(drop=True)
    ready = times+pd.Timedelta("10d")
    ready.iloc[8] = pd.Timestamp("2020-04-20")  # Initial-model label unknown at Apr1 refit.
    ready.iloc[12] = pd.Timestamp("2020-06-01") # Refit label unknown at May1 forecast.
    frame = pd.DataFrame(dict(payload=np.arange(16),__kurversc_label_available_at=ready))
    x = prepare_prediction_features(frame,columns=["payload"],categorical_columns=set(),datetime_columns=set())
    y = pd.Series([0,1]*8)
    seen = []
    class Model:
        def __init__(self,value):self.value=value
        def predict_proba(self,x):return np.tile([1-self.value,self.value],(len(x),1))
    def fit(features,labels,**kwargs):
        seen.extend(features.payload.tolist())
        return Model(float(labels.mean())),0.,(0,1)
    monkeypatch.setattr(module,"fit_final_catboost",fit)
    kwargs = dict(origin="2020-05-01",label_horizon="10d",refit_frames=1,
        policy=TemporalFitPolicy("incumbent"),task="classification",categorical=[],random_state=0)
    before,_ = fit_temporal_policy(x,y,times,**kwargs)
    assert 8 not in seen and 12 not in seen
    assert set(seen)=={9,10,11,13,14,15}
    changed = y.copy()
    changed.iloc[[8,12]] = 99  # These unavailable answers must never reach the learner.
    after,_ = fit_temporal_policy(x,changed,times,**kwargs)
    np.testing.assert_array_equal(before.predict_proba(x),after.predict_proba(x))
    assert "__kurversc_label_available_at" not in x


def test_support_floor_counts_only_initial_labels_actually_known_at_refit():
    t = pd.Series(pd.to_datetime(["2020-01-01","2020-02-01","2020-03-01","2020-04-01"]).repeat(4))
    ready = t+pd.Timedelta("10d")
    ready.iloc[8:11] = pd.Timestamp("2020-04-15")
    data = pd.DataFrame(dict(t=t,__kurversc_label_available_at=ready))
    plan = policy_support(data,timestamp="t",origin="2020-05-01",label_horizon="10d",refit_frames=1,
        policy=TemporalFitPolicy("floor",mode="pooled",minimum_training_rows=4))
    assert plan["initial"]==tuple(pd.to_datetime(["2020-02-01","2020-03-01"]))


def test_out_of_order_result_arrival_cannot_leak_an_unobserved_earlier_outcome():
    history = pd.DataFrame(dict(person=[1,1,1],when=pd.to_datetime(["2020-01-01","2020-01-05","2020-01-10"]),
        observed=pd.to_datetime(["2020-03-01","2020-02-01","2020-02-05"]),outcome=[1,0,1]))
    query = pd.DataFrame(dict(__event_entity=[1],__event_cutoff=[pd.Timestamp("2020-02-10")]))
    kwargs = dict(entity="person",timestamp="when",target="outcome",horizon="10d",available_at="observed",
        phase_boundary="2020-04-01")
    before = mature_label_state(history,query,**kwargs)
    history.loc[0,"outcome"] = 0
    after = mature_label_state(history,query,**kwargs)
    pd.testing.assert_frame_equal(before,after)
    assert before.ls_count.iloc[0]==2


def test_declared_label_arrival_is_inclusive_but_implicit_frozen_contract_is_unchanged():
    history = pd.DataFrame(dict(person=[1],when=[pd.Timestamp("2020-01-01")],
        observed=[pd.Timestamp("2020-01-05")],outcome=[1]))
    query = pd.DataFrame(dict(__event_entity=[1],__event_cutoff=[pd.Timestamp("2020-01-05")]))
    kwargs = dict(entity="person",timestamp="when",target="outcome",horizon="4d",phase_boundary="2020-01-05")
    explicit = mature_label_state(history,query,available_at="observed",**kwargs)
    old = mature_label_state(history,query,**kwargs)
    assert explicit.ls_count.iloc[0]==1 and explicit.ls_last.iloc[0]==1
    assert old.ls_count.iloc[0]==0
