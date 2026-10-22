import pickle

import numpy as np
import pandas as pd
import pytest

from kurversc.parent_features import BoundedParentEncoder, _field_tokens, parent_payload
from kurversc.specs import Table


def example():
    return pd.DataFrame(
        {
            "number": [1.0, 2.0, 4.0, np.nan, 5.0, 3.0],
            "country": ["US", "UK", "US", "UK", None, "DE"],
            "narrative": [
                "blue drug study",
                "red drug study",
                "blue cohort",
                "new study",
                "red cohort",
                "old study",
            ],
            "date": pd.to_datetime(["1990-01-01"] * 6),
        },
        index=[3, 7, 9, 11, 13, 17],
    )


def times(frame):
    return pd.Series(pd.Timestamp("2001-01-01"), index=frame.index)


def test_structured_types_dates_and_no_one_hot_expansion():
    frame = example()
    encoder = BoundedParentEncoder(text_components=3).fit(frame)
    result = encoder.transform(frame, timestamps=times(frame))
    country = next(
        name
        for name, plan in zip(encoder.structured_columns_, encoder.plan_)
        if plan == ("country", "category")
    )
    assert result.loc[3, country] == "value:US"
    assert result.loc[13, country] == "missing:"
    age = next(
        name
        for name, plan in zip(encoder.structured_columns_, encoder.plan_)
        if plan == ("date", "age_days")
    )
    assert (
        result.loc[3, age]
        == (pd.Timestamp("2001-01-01") - pd.Timestamp("1990-01-01")).days
    )
    assert len([p for p in encoder.plan_ if p[0] == "country"]) == 1
    assert result.shape[1] <= 67


def test_replay_row_column_order_and_batch_independence():
    frame = example()
    encoder = BoundedParentEncoder(text_components=3).fit(frame)
    result = encoder.transform(frame, timestamps=times(frame))
    restored = pickle.loads(pickle.dumps(encoder))
    pd.testing.assert_frame_equal(
        result, restored.transform(frame, timestamps=times(frame))
    )
    pd.testing.assert_frame_equal(
        result.loc[[7]],
        restored.transform(frame.loc[[7]], timestamps=times(frame.loc[[7]])),
        atol=1e-6,
        rtol=1e-6,
    )
    other = BoundedParentEncoder(text_components=3).fit(frame.iloc[::-1, ::-1])
    assert other.audit_ == encoder.audit_
    pd.testing.assert_frame_equal(
        result, other.transform(frame, timestamps=times(frame))
    )


def test_hard_feature_cardinality_fit_row_and_document_budgets():
    frame = pd.DataFrame(
        {f"c{i}": [f"{j} category" for j in range(30)] for i in range(80)}
    )
    encoder = BoundedParentEncoder(
        structured_features=5,
        text_components=3,
        max_input_columns=10,
        max_fit_rows=12,
        max_categories=4,
        max_text_columns=2,
        max_document_characters=16,
    ).fit(frame)
    result = encoder.transform(frame, timestamps=times(frame))
    assert result.shape[1] <= 8
    assert encoder.audit_["fit_rows"] == 12
    assert len(encoder.columns_) == 10
    assert all(len(v) <= 4 for v in encoder.category_values_.values())
    assert all(sum(map(len, row)) <= 16 for row in encoder._documents(frame))


def test_query_only_values_cannot_change_frozen_decisions():
    frame = example()
    encoder = BoundedParentEncoder(text_components=3).fit(frame)
    original = pickle.dumps(encoder)
    query = frame.copy()
    query["country"] = "never observed country"
    query["narrative"] = "future secret word"
    result = encoder.transform(query, timestamps=times(query))
    assert pickle.dumps(encoder) == original
    country = next(
        name
        for name, plan in zip(encoder.structured_columns_, encoder.plan_)
        if plan == ("country", "category")
    )
    assert result[country].eq("other:").all()
    with pytest.raises(ValueError, match="aligned"):
        encoder.transform(query, timestamps=times(query).iloc[::-1])
    with pytest.raises(ValueError, match="schema"):
        encoder.transform(query.drop(columns="country"), timestamps=times(query))


def test_text_field_identity_is_not_lost():
    first = set(_field_tokens(("blue drug", "red disease")))
    swapped = set(_field_tokens(("red disease", "blue drug")))
    assert first != swapped
    assert "f0:blue" in first and "f1:blue" not in first


def test_parent_point_in_time_key_exclusion_and_label_poisoning():
    table = Table(
        pd.DataFrame(
            {
                "id": [1, 2],
                "fk": [8, 9],
                "time": pd.to_datetime(["2001-01-01", "2003-01-01"]),
                "value": [20.0, 30.0],
            }
        ),
        key="id",
        date="time",
    )
    rows = pd.DataFrame(
        {
            "entity": [1, 2, 1, 9],
            "date": pd.to_datetime(
                ["2002-01-01", "2002-01-01", "2001-01-01", "2002-01-01"]
            ),
            "target": [0, 1, 1, 0],
        }
    )
    result = parent_payload(
        rows, table, entity_key="entity", timestamp="date", excluded=("fk",)
    )
    assert list(result) == ["time", "value"]
    assert result.loc[0, "value"] == 20
    assert result.iloc[1:].isna().all().all()
    rows["target"] = 999
    pd.testing.assert_frame_equal(
        result,
        parent_payload(
            rows, table, entity_key="entity", timestamp="date", excluded=("fk",)
        ),
    )


def test_empty_query_no_text_and_nested_cells():
    frame = pd.DataFrame({"x": [1, 2, 3], "nested": [["a", "b"], ["b"], {"k": 4}]})
    encoder = BoundedParentEncoder(text_components=0).fit(frame)
    result = encoder.transform(frame.iloc[:0], timestamps=times(frame.iloc[:0]))
    assert result.empty
    assert len(result.columns) == len(encoder.feature_names_)
    with pytest.raises(ValueError, match="nonempty"):
        BoundedParentEncoder().fit(frame.iloc[:0])


def test_catboost_accepts_generated_categories_and_floats():
    from catboost import CatBoostClassifier

    frame = example()
    encoder = BoundedParentEncoder(text_components=2).fit(frame)
    result = encoder.transform(frame, timestamps=times(frame))
    model = CatBoostClassifier(
        iterations=2, thread_count=1, verbose=False, allow_writing_files=False
    )
    model.fit(
        result, [0, 1, 0, 1, 1, 0], cat_features=list(encoder.categorical_columns_)
    )
    assert np.isfinite(model.predict_proba(result)).all()
