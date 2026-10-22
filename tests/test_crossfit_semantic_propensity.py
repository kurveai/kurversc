import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_reliability_blending import group_masks
from scripts.crossfit_semantic_propensity import CrossfitSemanticPropensity, bounded_joint_annotation


def inputs():
    rng = np.random.default_rng(41)
    x = pd.DataFrame(rng.normal(size=(120, 8)), columns=[f"semantic_{i}" for i in range(8)])
    y = (x.semantic_0.to_numpy() + rng.normal(size=len(x)) > 0).astype(int)
    args = dict(entities=np.tile(np.arange(60), 2), timestamps=np.repeat(pd.Timestamp("2020-01-01"), len(x)),
                origin="2020-03-01", horizon="30d")
    return x, y, args


def test_own_entity_label_changes_cannot_change_its_oof_annotation():
    x, y, args = inputs()
    original = CrossfitSemanticPropensity().fit(x, y, **args)
    _, query = group_masks(args["entities"])[0]
    changed = y.copy()
    changed[query] = 1 - changed[query]
    replay = CrossfitSemanticPropensity().fit(x, changed, **args)
    np.testing.assert_array_equal(original.oof[query], replay.oof[query])
    assert all(a["own_entity_labels_excluded"] for a in original.audit["groups"])


def test_joint_annotation_keeps_width_and_forecast_cannot_fit():
    x, y, args = inputs()
    model = CrossfitSemanticPropensity().fit(x, y, **args)
    annotated = bounded_joint_annotation(x, model.oof)
    assert annotated.shape == x.shape
    assert list(annotated)[:-1] == list(x)[:-1]
    assert "semantic_propensity_crossfit_logit" in annotated
    before = model.model.coef_.copy()
    prediction = model.transform(x.iloc[:4])
    assert prediction.shape == (4,)
    np.testing.assert_array_equal(model.model.coef_, before)
    with pytest.raises(ValueError, match="mature"):
        CrossfitSemanticPropensity().fit(x, y, **{**args, "origin":"2020-01-15"})
