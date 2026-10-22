"""Bounded three-hop shared-object history, without changing existing peers.

The standard peer provider stops at two hops. This research extension exposes
only schema-derived three-hop paths ending at a referenced entity. It reuses
the existing point-in-time membership extraction and leave-own-root-out peer
readout. No task-specific names or raw identity predictors are introduced.
"""

import numpy as np
import pandas as pd

from kurversc.peer_context import PeerContext
from kurversc.peer_history import PeerHistoryFamily
from kurversc.relational_inputs import key_parts, schema_paths
from kurversc.signal_challengers import ChallengerUnavailable


class DeepPeerExtension:
    def fit(self, history, *, tables, relationships, root_name, phase_boundary,
            entity="__event_entity", timestamp="__event_cutoff", target="__outcome",
            available_at="__available_at", max_paths=2, max_history_rows=50000):
        if not 1 <= max_paths <= 4 or max_history_rows < 1:
            raise ValueError("A bounded path and observed-history budget is required")
        edges = {(edge.parent,edge.child):edge for edge in relationships}
        # Two owned-event hops followed by an object lookup are an interaction
        # bridge. Repeated dimension lookups consume no slots in this bounded
        # extension: they do not recover the omitted event/object interaction.
        paths = [path for path,edge in schema_paths(root_name,relationships,3)
                 if len(path)==3 and not edge.reduce
                 and edges[(root_name,path[0])].reduce
                 and edges[(path[0],path[1])].reduce
                 and len(key_parts(tables[path[-1]].key))==1]
        paths = sorted(set(paths))[:max_paths]
        if not paths:
            raise ChallengerUnavailable("No eligible three-hop shared-object path")
        history = history.copy()
        history[timestamp] = pd.to_datetime(history[timestamp], errors="raise")
        history[available_at] = pd.to_datetime(history[available_at], errors="raise")
        full = history.loc[history[available_at].le(pd.Timestamp(phase_boundary))].copy()
        if full.empty:
            raise ChallengerUnavailable("No mature observed history for three-hop context")
        if len(full) > max_history_rows:
            hashes = pd.util.hash_pandas_object(full[[entity,timestamp]],index=False).to_numpy()
            indices = np.argsort(hashes,kind="stable")[:max_history_rows]
            selected = full.iloc[np.sort(indices)].copy()
        else:
            selected = full
        # Empty relationships initialize and validate the ordinary peer state
        # without extracting irrelevant shorter-path memberships.
        peer = PeerHistoryFamily(max_depth=3,max_paths=max_paths,max_neighbors=32).fit(
            selected,tables=tables,relationships=(),root_name=root_name,entity_key=entity,
            timestamp=timestamp,target=target,available_at=available_at)
        peer._membership_tables_ = peer._stable_membership_tables(tables,relationships,paths)
        for path in paths:
            rows = peer._members(peer.history_[[entity,timestamp]],tables,relationships,path)
            peer.memberships_[path] = rows.drop_duplicates(["__example_id","__neighbor_key_0"])
        peer.paths_ = tuple(paths)
        provider = PeerContext(mode="both",max_history_rows=max_history_rows,max_paths=max_paths,
                               max_neighbors=32,large_max_paths=max_paths,large_max_neighbors=32)
        provider.peer_,provider.tables_,provider.relationships_ = peer,tables,relationships
        provider.source_history_rows_,provider.history_rows_ = len(full),len(selected)
        self.provider,self.keys = provider,[entity,timestamp]
        self.audit = dict(paths=paths,depth=3,path_shape="two_owned_event_hops_then_referenced_object",
            max_paths=max_paths,max_neighbors=32,
            observed_history_rows=len(full),sampled_history_rows=len(selected),
            memberships={"/".join(path):len(rows) for path,rows in peer.memberships_.items()},
            phase_boundary=str(phase_boundary),maximum_extra_columns=25*max_paths,
            unchanged_existing_peer_columns=True,raw_ids_are_predictors=False)
        return self

    def transform(self, requests):
        query = requests[self.keys].reset_index(drop=True).copy()
        query[self.keys[1]] = pd.to_datetime(query[self.keys[1]], errors="raise")
        result = self.provider.transform(query)
        return result.drop(columns=self.keys).add_prefix("deep__")
