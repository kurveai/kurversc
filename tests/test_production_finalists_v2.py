import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_production_finalists_v2 import (
    compare_cases, compatible_audits, planned_capacity, materialized_capacity,
    CONTRACT, MAXIMUM_FEATURE_CELLS, MAXIMUM_AUDIT_MATRIX_CELLS,
)
from kurversc.search import GraphConfig, Trial


def pair():
    def make(width, families):
        return Trial(config=GraphConfig(depth=1,feature_families=families),metric="mae",
            validation_score=.1,objective_score=-.1,feature_count=width,train_rows=100,
            validation_rows=50,feature_seconds=0,model_seconds=0,stage="rerank",
            cutoff_labels=("a","b","c"),cutoff_scores=(.12,.11,.1))
    return make(1200,("base","temporal")),make(450,("base",))


def cases(scores):
    return {(fold,seed):dict(truth=np.zeros(50),prediction=np.full(50,score),
        groups=np.arange(50),origin=f"2020-0{fold+1}-01")
        for fold,score in enumerate(scores) for seed in range(3)}


def clocked(count=3, blocks=1, scores=(.12,.11,.1)):
    data={k:v for k,v in cases(scores).items() if k[0]<count}
    for (fold,_),v in data.items():
        v["forecast_cutoffs"]=np.asarray([f"2020-0{fold+1}-{i%blocks+1:02d}" for i in range(50)])
    return data


@pytest.mark.parametrize("origins,blocks", [(2,1),(3,1),(3,23)])
def test_same_selector_supports_available_origins_and_multicutoff_forecasts(origins,blocks):
    result=compare_cases(*pair(),clocked(origins,blocks),clocked(origins,blocks,(.1198,.1101,.1001)))
    assert result["selected"]=="challenger" and result["origins"]==origins
    assert not result["official_targets_used_for_selection"]


def test_forecast_clocks_must_align_even_when_entities_and_targets_match():
    alt=clocked()
    alt[0,0]["forecast_cutoffs"][0]="1990-01-01"
    with pytest.raises(AssertionError):compare_cases(*pair(),clocked(),alt)


def test_single_origin_or_missing_seed_does_not_qualify():
    with pytest.raises(ValueError,match="Complete"):
        compare_cases(*pair(),clocked(1),clocked(1))
    alt=clocked(2)
    del alt[1,1]
    with pytest.raises(ValueError,match="Complete"):
        compare_cases(*pair(),clocked(2),alt)


def test_safety_is_not_weakened_when_there_are_only_two_origins():
    assert compare_cases(*pair(),clocked(2),clocked(2,scores=(.123,.10,.09)))["selected"]=="incumbent"


def test_capacity_is_target_blind_and_preserves_original_frame_budget():
    a,b=pair()
    small=planned_capacity(((Path("a"),a),(Path("b"),b)))
    assert small["eligible"]
    a.train_rows=MAXIMUM_FEATURE_CELLS
    assert not planned_capacity(((Path("a"),a),(Path("b"),b)))["eligible"]
    assert CONTRACT["initial_training_frames"]==1
    assert CONTRACT["minimum_origins"]==2
    assert MAXIMUM_AUDIT_MATRIX_CELLS==3_000_000_000


def test_audits_require_matched_estimator_budget_not_just_similar_scores(tmp_path):
    raw=dict(folds=[dict(origin="a"),dict(origin="b")],seeds=[0,1,2],observed_refit_frames=23,
        row_limit=100000,model_params=dict(thread_count=4),metadata=dict(atomic_date_replay=True,fold="current",
            dataset="generic",task="unknown",target="y",metric="mae",label_horizon="1d",validation_cutoffs=23))
    roots=[tmp_path/"a",tmp_path/"b"]
    for root in roots:
        (root/"raw-audit").mkdir(parents=True)
        (root/"raw-audit/manifest.json").write_text(json.dumps(raw))
    compatible_audits(*roots)
    raw["row_limit"]=10000
    (roots[1]/"raw-audit/manifest.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError,match="row_limit"):compatible_audits(*roots)


def test_actual_capacity_reads_only_clocks_and_counts_not_target_payload(tmp_path):
    (tmp_path/"cache").mkdir()
    metadata=dict(feature_schema=dict(columns=["x"]),label_horizon="1d",validation_cutoffs=1)
    (tmp_path/"cache/materialization.json").write_text(json.dumps(metadata))
    pd.DataFrame({"__event_cutoff":pd.date_range("2020-01-01",periods=10)}).to_parquet(tmp_path/"cache/history.parquet")
    pd.DataFrame({"not_a_target":[0,0]}).to_parquet(tmp_path/"cache/validation.parquet")
    for fold,origin in enumerate(["2020-01-08","2020-01-09"]):
        root=tmp_path/f"snapshots/fold{fold}"
        root.mkdir(parents=True)
        (root/"manifest.json").write_text(json.dumps(dict(snapshot_cutoff=origin)))
        pd.DataFrame({"not_a_target":[1,1]}).to_parquet(root/"validation.parquet")
    result=materialized_capacity(tmp_path)
    assert result["eligible"] and not result["target_values_accessed"]
    assert len(result["phase_cells"])==2
