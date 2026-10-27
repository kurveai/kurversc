import numpy as np
import pandas as pd
import pytest

from kurversc.modeling import sample_training_rows
from scripts.matched_learner_challengers import changed_parameters, member_positions, positioned_frame


def test_projection_preserves_dtypes_values_and_order(tmp_path):
    frame = pd.DataFrame({"number": [1.,np.nan,3.,4.], "text": ["a",None,"c","d"],
        "date": pd.date_range("2020-01-01",periods=4), "key": [4,3,2,1]})
    path = tmp_path/"data.parquet"
    frame.to_parquet(path,index=False)
    actual = positioned_frame(path,list(frame),np.array([3,0,2]),width=2)
    pd.testing.assert_frame_equal(actual,frame.iloc[[3,0,2]].reset_index(drop=True))


def test_projection_rejects_duplicates_and_missing_rows(tmp_path):
    path = tmp_path/"data.parquet"
    pd.DataFrame({"x": [1,2,3]}).to_parquet(path,index=False)
    with pytest.raises(ValueError):
        positioned_frame(path,["x"],[0,0])
    with pytest.raises(ValueError):
        positioned_frame(path,["x"],[3])


@pytest.mark.parametrize("task",["regression","classification"])
def test_projection_samples_exact_incumbent_positions(task):
    frame = pd.DataFrame({"__event_cutoff": ["2020-01-01"]*50+["2020-02-01"]*50,
        "target": np.arange(100)%2})
    actual = member_positions(frame,cutoff=pd.Timestamp("2020-02-01"),target="target",
        row_limit=23,task=task,seed=2,number=4)
    subset = frame.iloc[50:]
    x = pd.DataFrame({"position":np.arange(50,100)})
    selected,_ = sample_training_rows(x,subset.target.reset_index(drop=True),
        limit=23,task=task,random_state=6)
    np.testing.assert_array_equal(actual,selected.position)


def test_only_one_variable_changes():
    params = dict(iterations=600,depth=5,l2_leaf_reg=30,thread_count=2)
    ordered,limit = changed_parameters("ordered_boosting",params,100000)
    assert ordered == dict(params,boosting_type="Ordered") and limit==100000
    full,limit = changed_parameters("full_rows",params,100000)
    assert full==params and limit is None
    assert "boosting_type" not in params
    with pytest.raises(ValueError):
        changed_parameters("unknown",params,100000)
