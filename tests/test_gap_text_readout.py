import numpy as np
import pandas as pd

from scripts.gap_text_readout import FieldTextReadout


def test_field_words_are_not_collapsed_across_roles():
    frame=pd.DataFrame({"first":["responding treatment"]*8+["unsuccessful trial"]*8,
                        "second":["unsuccessful trial"]*8+["responding treatment"]*8})
    labels=pd.Series([1]*8+[0]*8)
    head=FieldTextReadout(maximum_per_field=10).fit(frame,labels,list(frame))
    predictions=head.predict_proba(frame)[:,1]
    assert predictions[:8].mean()>predictions[8:].mean()
    assert sum(head.audit["vocabulary_sizes"].values())<=20


def test_forecast_words_never_expand_training_vocabulary():
    frame=pd.DataFrame({"description":["common clinical benefit"]*8+["common clinical failure"]*8})
    head=FieldTextReadout().fit(frame,pd.Series([1]*8+[0]*8),list(frame))
    before=dict(head.vectorizers["description"].vocabulary_)
    result=head.predict_proba(pd.DataFrame({"description":["unseenword futureword"]}))
    assert result.shape==(1,2)
    assert head.vectorizers["description"].vocabulary_==before
    assert "futureword" not in before
    assert np.isfinite(result).all()
