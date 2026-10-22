from pathlib import Path
from types import SimpleNamespace

import pytest

from kurversc.search import GraphConfig
from scripts.benchmark_small_finalist_readout import small_finalist


def trial(config, width, score, *, stage="rerank", status="completed", metric="mae"):
    return SimpleNamespace(config=config, feature_count=width, cutoff_scores=(.12, .08, score),
        stage=stage, status=status, metric=metric, execution_plan={"frozen":True})


def test_selects_small_current_finalist_without_official_or_historical_scores():
    large = GraphConfig(feature_families=("base", "temporal"), depth=1)
    small = GraphConfig(feature_families=("base",), depth=1)
    history = GraphConfig(feature_families=("base",), depth=1, include_target_history=True)
    rows = [(Path("large"), trial(large, 1200, .058)),
        (Path("small"), trial(small, 500, .0581)),
        (Path("smaller"), trial(history, 450, .0582))]
    assert small_finalist(rows, large)[0] == Path("smaller")
    assert small_finalist(list(reversed(rows)), large)[0] == Path("smaller")


def test_does_not_promote_a_bad_or_incomplete_small_graph():
    large = GraphConfig(feature_families=("base", "temporal"), depth=1)
    small = GraphConfig(feature_families=("base",), depth=1)
    rows = [(Path("large"), trial(large, 1200, .058)),
        (Path("bad"), trial(small, 450, .06)),
        (Path("screen"), trial(small, 400, .057, stage="screening")),
        (Path("failed"), trial(small, 300, .056, status="failed"))]
    assert small_finalist(rows, large) is None
    with pytest.raises(ValueError, match="bounded"):
        small_finalist(rows, large, relative_recent_loss_cap=.1)


def test_requires_incumbent_full_fidelity_and_never_expands_to_larger_graphs():
    large = GraphConfig(feature_families=("base", "temporal"), depth=1)
    small = GraphConfig(feature_families=("base",), depth=1)
    with pytest.raises(ValueError, match="deployed"):
        small_finalist([(Path("small"),trial(small, 450, .058))], large)
    assert small_finalist([(Path("small"),trial(small, 450, .058)),
        (Path("large"),trial(large, 1200, .0579))], small) is None
