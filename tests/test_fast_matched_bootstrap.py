import numpy as np
import pytest

from kurversc.temporal_policy import select_temporal_policy
from scripts.fast_matched_bootstrap import EntityBlocks, sample_positions, accelerated_select_temporal_policy


@pytest.mark.parametrize("codes",[[2,0,1,2,1,0],[3,0,1,2],[0,0,0,1,2,2,3]])
def test_blocks_and_resampling_are_bit_identical(codes):
    codes=np.asarray(codes)
    blocks=EntityBlocks(codes,len(np.unique(codes)))
    naive=[np.flatnonzero(codes==i) for i in range(len(blocks))]
    for i in range(len(blocks)):
        np.testing.assert_array_equal(blocks[i],naive[i])
    rng=np.random.default_rng(42)
    for _ in range(50):
        indices=rng.integers(0,len(blocks),len(blocks))
        np.testing.assert_array_equal(sample_positions(blocks,indices),np.concatenate([naive[i] for i in indices]))


@pytest.mark.parametrize("task",["classification","regression"])
def test_complete_selector_is_exactly_equal(task):
    rng=np.random.default_rng(10)
    groups=np.repeat(np.arange(30),2)
    evidence=[]
    for origin in ["2020-01-01","2020-04-01","2020-07-01"]:
        y=rng.integers(0,2,len(groups)) if task=="classification" else rng.normal(size=len(groups))
        prediction=y+rng.normal(scale=.6,size=len(y))
        for seed in (0,1,2):
            for policy,values in [("incumbent",prediction),("candidate",.8*prediction+.2*y)]:
                evidence.append(dict(origin=origin,seed=seed,policy=policy,truth=y,
                    groups=groups,prediction=values))
    kwargs=dict(task=task,bootstrap_repeats=30,minimum_origins=3,minimum_seeds=3)
    assert accelerated_select_temporal_policy(evidence,**kwargs)==select_temporal_policy(evidence,**kwargs)
