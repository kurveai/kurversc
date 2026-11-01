from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kurversc.signal_challengers import (BoundedTextReadout, ChallengerUnavailable,
    SignalChallenger, challenger_menu, challenger_policy, combine_predictions,
    mature_support_mask, select_text_columns, apply_challenger_weight, challenger_fit_key)
from kurversc.temporal_policy import TemporalFitPolicy, InsufficientPolicySupport
from scripts.schema_exposure_challenger import discover_exposure_path


def test_menu_is_bounded_and_disabled_categories_add_nothing():
    assert len(challenger_menu()) == 4
    assert challenger_menu(text=False, exposure=False, support=False) == ()
    assert len({p.name for p in challenger_menu()}) == 4


def test_removed_exposure_cannot_be_enabled_through_the_public_menu():
    for kwargs in ({"exposure":True},{"exposure_blends":True}):
        with pytest.raises(ValueError,match="removed"):
            challenger_menu(**kwargs)
    with pytest.raises(ValueError,match="supported challenger"):
        SignalChallenger("old_exposure","exposure")


@pytest.mark.parametrize("kind", ["text", "support"])
def test_every_readout_honors_its_declared_weight_exactly_once(kind):
    spec = SignalChallenger("partial", kind, weight=.5)
    native, head = np.array([.2,.8]), np.array([.4,.6])
    np.testing.assert_allclose(apply_challenger_weight(spec,native,head),[.3,.7])
    np.testing.assert_array_equal(native,[.2,.8])
    np.testing.assert_array_equal(head,[.4,.6])
    assert apply_challenger_weight(SignalChallenger("full",kind),native,head) is head


def test_support_challengers_do_not_mutate_incumbent():
    base = TemporalFitPolicy("pooled12", mode="pooled", training_frames=12)
    spec = SignalChallenger("more_support", "support", training_frames=3)
    selected = challenger_policy(base, spec)
    assert selected.training_frames == 3 and selected.mode == "frame_ensemble"
    assert base.training_frames == 12 and base.mode == "pooled"


def test_insufficient_three_frame_support_is_not_silently_capped():
    times = pd.Series(pd.date_range("2020-01-01", periods=3, freq="D"))
    with pytest.raises(InsufficientPolicySupport):
        mature_support_mask(times, origin="2020-01-04", horizon="1D", refit_frames=1,
            policy=TemporalFitPolicy("three", training_frames=3))


def test_text_selection_ignores_heldout_and_nonparent_columns():
    frame = pd.DataFrame({"parent_summary":[f"shared clinical intervention description number {i}" for i in range(30)],
        "child_text":[f"very long unrelated child words example {i}" for i in range(30)]})
    assert select_text_columns(frame, list(frame), np.ones(30, dtype=bool), root_prefix="parent") == ["parent_summary"]
    assert select_text_columns(frame, list(frame), np.zeros(30, dtype=bool), root_prefix="parent") == []


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_text_vocabulary_is_train_only_and_predictions_are_finite(task):
    x = pd.DataFrame({"parent_summary":[f"clinical intervention description outcome subgroup{i % 4} record {i}" for i in range(40)]})
    y = pd.Series(np.arange(40) % 2 if task == "classification" else np.arange(40) / 10)
    head = BoundedTextReadout(task=task).fit(x, y, categorical=list(x), root_prefix="parent")
    test = x.iloc[:3].copy()
    test["parent_summary"] = "neverobservedheldouttoken"
    assert np.isfinite(head.predict_values(test)).all()
    assert "neverobservedheldouttoken" not in head.vectorizer.vocabulary_
    assert head.audit["vocabulary_size"] <= 16384


def test_no_text_is_an_explicit_unavailable_challenger():
    with pytest.raises(ChallengerUnavailable):
        BoundedTextReadout(task="classification").fit(pd.DataFrame({"id":[1, 2]}),
            pd.Series([0, 1]), categorical=[])


def table(frame, pk=None, time=None, fk=None):
    return SimpleNamespace(df=frame, pkey_col=pk, time_col=time, fkey_col_to_pkey_table=fk or {})


def generic_schema():
    return {"people":table(pd.DataFrame({"key":[0,1]}), "key"),
        "sessions":table(pd.DataFrame({"sid":[0,1],"owner":[0,1],"clock":pd.date_range("2020-01-01",periods=2)}),
            "sid", "clock", {"owner":"people"}),
        "observations":table(pd.DataFrame({"session":[0,1],"object":[0,1],"stamp":pd.date_range("2020-01-01",periods=2),
            "response":[0,1],"context":[2,3]}), None, "stamp", {"session":"sessions","object":"objects"}),
        "objects":table(pd.DataFrame({"oid":[0,1]}), "oid")}


def test_exposure_discovery_uses_fk_metadata_not_avito_names():
    path = discover_exposure_path(generic_schema(), "people")
    assert path.parent == "sessions" and path.event == "observations"
    assert path.response == "response" and path.context == "context"
    assert path.object_table == "objects"


def test_absent_fk_or_binary_event_is_inapplicable():
    tables = generic_schema()
    tables["observations"].df["response"] = [2,3]
    assert discover_exposure_path(tables, "people") is None
    tables = generic_schema()
    tables["sessions"].fkey_col_to_pkey_table = {}
    assert discover_exposure_path(tables, "people") is None


def test_prediction_blend_checks_ordered_shape_and_finiteness():
    np.testing.assert_allclose(combine_predictions([.2,.8],[.4,.6],.5),[.3,.7])
    with pytest.raises(ValueError):
        combine_predictions([.2,.8],[.4],.5)
    with pytest.raises(ValueError):
        combine_predictions([.2,.8],[np.nan,.5],.5)
