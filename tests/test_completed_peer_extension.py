import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from kurversc import Relationship, Table
from scripts.completed_peer_extension import CompletedPeerExtension


def fit(history=None, *, late=False):
    ids = [1,2,3,4] + ([5] if late else [])
    clocks = pd.to_datetime(["2000-07-01"]*4 + (["2001-07-01"] if late else []))
    tables = {
        "roots":Table(pd.DataFrame(dict(id=ids)),key="id",timeless=True),
        "sessions":Table(pd.DataFrame(dict(sid=ids,root=ids,time=clocks)),key="sid",date="time"),
        "events":Table(pd.DataFrame(dict(eid=ids,session=ids,obj=[10]*len(ids),time=clocks)),key="eid",date="time"),
        "objects":Table(pd.DataFrame(dict(oid=[10])),key="oid",timeless=True),
    }
    edges = [Relationship("roots","sessions","id","root"),Relationship("sessions","events","sid","session"),
             Relationship("events","objects","obj","oid",reduce=False)]
    if history is None:
        history = pd.DataFrame(dict(__event_entity=[1,2,3],__event_cutoff=pd.to_datetime(["2000-01-01"]*3),
            __available_at=pd.to_datetime(["2001-01-01"]*3),__outcome=[0.,1.,.2]))
    provider = CompletedPeerExtension().fit(history,tables=tables,relationships=edges,root_name="roots",
                                           phase_boundary="2002-01-01")
    return provider,history


def query(root=4, date="2001-01-01"):
    return pd.DataFrame(dict(__event_entity=[root],__event_cutoff=pd.to_datetime([date])))


def test_completed_window_interactions_are_available_at_exact_maturity_not_before():
    provider,_ = fit()
    at = provider.transform(query())
    assert at.loc[0,"completed_peer__peer_p0__mean__mean"]==pytest.approx(.4)
    assert at.loc[0,"completed_peer__peer_p0__support__mean"]==3
    before = provider.transform(query(date="2000-12-31"))
    assert before.loc[0,"completed_peer__peer_p0__supported_entities"]==0
    assert provider.audit["membership_clock"]=="true target availability"


def test_own_current_or_historical_labels_cannot_change_own_context():
    provider,history = fit()
    changed,_ = fit(history.assign(__outcome=[999.,1.,.2]))
    assert_frame_equal(provider.transform(query(root=1)),changed.transform(query(root=1)))


def test_sources_after_completion_cannot_enter_historical_memberships():
    base,history = fit()
    # Root 5 has an old mature outcome but acquired the object AFTER it completed.
    augmented = pd.concat([history,history.iloc[:1].assign(__event_entity=5,__outcome=999.)],ignore_index=True)
    provider,_ = fit(augmented,late=True)
    result = provider.transform(query(date="2002-01-01"))
    assert result.loc[0,"completed_peer__peer_p0__support__mean"]==3
    assert result.loc[0,"completed_peer__peer_p0__mean__mean"]==pytest.approx(.4)


def test_instantaneous_targets_are_not_shifted_into_the_past():
    _,history = fit()
    with pytest.raises(ValueError,match="positive"):
        fit(history.assign(__available_at=history.__event_cutoff))


def test_future_query_clock_cannot_advance_the_frozen_source_phase():
    provider,_ = fit(late=True)
    assert_frame_equal(provider.transform(query(date="2002-01-01")),
                       provider.transform(query(date="2025-01-01")))
