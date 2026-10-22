import numpy as np
import pandas as pd
import pytest

from scripts.causal_parent_text_annotation import CausalParentTextAnnotation


def fixture():
    rows = [dict(__event_entity=f"p{i}", __event_cutoff=date, text=("research scientist" if i % 2 else "student programmer"), target=i % 2)
        for date in ("2018-01-01", "2018-04-01", "2018-07-01", "2018-10-01", "2019-04-01")
        for i in range(60)]
    return pd.DataFrame(rows)


def fit(history):
    return CausalParentTextAnnotation(maximum_per_field=16).fit(history, target="target", columns=["text"],
        first_query="2019-04-01", horizon="91D")


def test_causal_crossfit_annotation_excludes_all_own_and_future_labels():
    history = fixture()
    query = history.loc[history["__event_cutoff"].eq("2019-04-01") & history["__event_entity"].eq("p0")]
    a = fit(history)
    changed = history.copy()
    mask = changed["__event_entity"].eq("p0") | changed["__event_cutoff"].eq("2019-04-01")
    changed.loc[mask, "target"] = 1 - changed.loc[mask, "target"]
    b = fit(changed)
    np.testing.assert_array_equal(a.transform(query), b.transform(query))
    shuffled = history.sample(frac=1, random_state=9)
    np.testing.assert_array_equal(a.transform(query), fit(shuffled).transform(query))
    assert pd.Timestamp(a.audit["latest_label_complete_at"]) < pd.Timestamp("2019-04-01")
    assert a.audit["output_features"] == 1
    with pytest.raises(ValueError, match="follow"):
        a.transform(history.iloc[:2])
