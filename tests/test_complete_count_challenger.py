from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from kurversc.temporal_policy import TemporalFitPolicy
from scripts import benchmark_complete_count_challenger as runner
from scripts.run_gap_followup_queue import digest


def test_complete_rule_keeps_paired_confidence_and_all_seed_older_cap(monkeypatch):
    def gate(records, **kwargs):
        assert kwargs["minimum_origins"] == 3 and kwargs["minimum_seeds"] == 3
        assert kwargs["alpha"] == .05*3/1024
        candidate = lambda gain, old_loss: dict(reason="eligible", folds=[
            dict(seed_gains=[-old_loss]*3, practical_loss_cap=.01),
            dict(gain=gain, recent_lower_bound=.001, temporal_gain_lower_bound=.001)])
        return dict(candidates={"count_likelihood_0.125":candidate(.02,.005),
            "count_likelihood_0.5":candidate(.03,.005),
            "count_likelihood_1":candidate(.04,.02)})
    monkeypatch.setattr(runner,"frozen_auc_select_temporal_policy",gate)
    selected,_=runner.choose(dict(fresh=[],archive=[]),origins=3)
    assert selected == "count_likelihood_0.5"  # The larger older loss is vetoed.


def test_reused_cases_reconstruct_all_declared_recipes_without_retraining(tmp_path):
    task,output,reuse=tmp_path/"task",tmp_path/"output",tmp_path/"reuse"
    for folder in [task/"cache",task/"snapshots/fold0",task/"raw-audit",task/"selection",output,reuse/"fold0"]:
        folder.mkdir(parents=True,exist_ok=True)
    schema=dict(columns=["past"],categorical_columns=[],datetime_columns=[])
    meta=dict(feature_schema=schema,target="label",label_horizon="1d",validation_cutoffs=1)
    history=pd.DataFrame(dict(past=np.tile([1,2,3],2),label=np.tile([0,1,3],2),
        __event_entity=np.tile([11,12,13],2),__event_cutoff=np.repeat(["2020-01-01","2020-01-02"],3)))
    held=history.iloc[:3].copy().assign(__event_cutoff="2020-01-04")
    history.to_parquet(task/"cache/history.parquet",index=False)
    held.to_parquet(task/"snapshots/fold0/validation.parquet",index=False)
    (task/"snapshots/fold0/manifest.json").write_text(json.dumps(dict(origin="2020-01-04",
        snapshot_cutoff="2020-01-04",latest_observed_label_complete_at="2020-01-03")))
    raw=dict(metadata=meta,folds=[{}],observed_refit_frames=1,row_limit=100_000,model_params=dict(thread_count=1))
    (task/"raw-audit/manifest.json").write_text(json.dumps(raw))
    (task/"selection/manifest.json").write_text(json.dumps(dict(metadata=meta)))
    native=np.array([.2,.7,2.7]); truth=np.array([0.,1.,3.]);groups=np.array([11,12,13])
    np.savez_compressed(task/"raw-audit/fold0-incumbent-seed0.npz",truth=truth,groups=groups,prediction=native)
    previous=reuse/"fold0/seed0.npz"
    np.savez_compressed(previous,truth=truth,groups=groups,native=native,raw_count_median=truth)
    previous.with_suffix(".json").write_text(json.dumps(dict(fits=[])))
    identity=dict(policy=asdict(TemporalFitPolicy("incumbent")),folds=[0,1,2],
        reused_case_files={str(p):digest(p) for p in (previous,previous.with_suffix(".json"))})
    (output/"manifest.json").write_text(json.dumps(identity))
    record=runner.case(task,output,reuse,0,0)
    assert record["reused"] and record["native_archive_max_error"] == 0
    with np.load(output/"fold0/seed0.npz") as arrays:
        for name in runner.RECIPES:
            assert "fresh_"+name in arrays and "archive_"+name in arrays
        np.testing.assert_array_equal(arrays["truth"],truth)
        np.testing.assert_array_equal(arrays["groups"],groups)
