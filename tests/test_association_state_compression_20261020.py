import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from kurversc import Relationship, Table
from scripts.association_state_compression_20261020 import AssociationState, IncidenceBasis


def test_incidence_encoder_is_training_only_order_invariant_and_does_not_emit_keys():
    rows = pd.DataFrame(dict(__example_id=[0, 0, 1, 1, 2, 3], __neighbor_key_0=[10, 11, 11, 12, 12, 13]))
    a = IncidenceBasis().fit(rows, 4)
    b = IncidenceBasis().fit(rows.sample(frac=1, random_state=7), 4)
    x, support = a.transform(rows, 4)
    y, _ = b.transform(rows, 4)
    np.testing.assert_array_equal(x, y)
    cold = pd.DataFrame(dict(__example_id=[0], __neighbor_key_0=[999]))
    values, valid = a.transform(cold, 1)
    assert not valid.any() and not values.any()
    assert support.all() and x.shape[1] <= 16


def test_actual_memberships_reject_future_sources_without_using_targets():
    roots = pd.DataFrame(dict(id=[0, 1, 2, 3]))
    events = pd.DataFrame(dict(eid=[0, 1, 2, 3, 4, 5, 6], uid=[0, 0, 1, 1, 2, 3, 3],
        oid=[10, 11, 11, 12, 12, 13, 10], at=pd.to_datetime(["2000-01-01"]*6+["2001-01-01"])))
    tables = dict(actors=Table(roots, key="id", timeless=True), activity=Table(events, key="eid", date="at"),
        objects=Table(pd.DataFrame(dict(id=[10, 11, 12, 13])), key="id", timeless=True))
    edges = [Relationship("actors", "activity", "id", "uid"), Relationship("activity", "objects", "oid", "id", reduce=False)]
    train = pd.DataFrame(dict(__event_entity=[0, 1, 2, 3], __event_cutoff=pd.Timestamp("2000-02-01"), target=[0, 1, 0, 1]))
    query = train.assign(__event_cutoff=pd.Timestamp("2000-03-01"))
    a, _ = AssociationState().fit_transform(train, query, tables=tables, relationships=edges, root_name="actors", phase_boundary="2000-03-01")
    changed = dict(tables, activity=Table(events.assign(oid=[10, 11, 11, 12, 12, 13, 999]), key="eid", date="at"))
    b, _ = AssociationState().fit_transform(train.assign(target=[999]*4), query,
        tables=changed, relationships=edges, root_name="actors", phase_boundary="2000-03-01")
    for first, second in zip(a, b):
        assert_frame_equal(first, second, check_exact=True)
        assert not any("key" in c or c=="id" for c in first) and len(first.columns)<=32
