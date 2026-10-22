"""Observed-outcome peer neighborhoods, bounded and strictly point-in-time.

A historical peer's neighborhood is known through its target availability
time, not just the start of that already-completed outcome window. A query
can use a label only once it is available. Ordinary source rows remain STRICTLY
before their respective membership cutoff. No queried root contributes labels.
"""

import pandas as pd

from scripts.deep_peer_extension import DeepPeerExtension


class CompletedPeerExtension(DeepPeerExtension):
    def fit(self, history, *, entity="__event_entity", timestamp="__event_cutoff",
            target="__outcome", available_at="__available_at", **kwargs):
        if (pd.to_datetime(history[available_at]) <= pd.to_datetime(history[timestamp])).any():
            raise ValueError("Completion-time peer context requires strictly positive target horizons")
        super().fit(history,entity=entity,timestamp=timestamp,target=target,available_at=available_at,**kwargs)
        self.phase_boundary = pd.Timestamp(kwargs["phase_boundary"])
        provider = self.provider
        peer = provider.peer_
        requests = peer.history_[[entity,available_at]].rename(columns={available_at:timestamp})
        for path in peer.paths_:
            rows = peer._members(requests,provider.tables_,provider.relationships_,path)
            peer.memberships_[path] = rows.drop_duplicates(["__example_id","__neighbor_key_0"])
        # Existing peer reducers use strict availability. On the integer-ns
        # timeline, ready-1ns < query is exactly ready <= query. The TRUE
        # completion clock above still bounds all ordinary source extraction.
        peer.history_[available_at] = pd.to_datetime(peer.history_[available_at])-pd.Timedelta(1,"ns")
        provider._cache.clear()
        self.audit.update(membership_clock="true target availability",label_eligibility="available_at <= query",
            source_membership_cutoff="ordinary rows strictly before true completion time",
            label_eligibility_implementation="integer-nanosecond <= identity; no source clock shifted",
            queried_root_labels_excluded=True,
            memberships={"/".join(path):len(rows) for path,rows in peer.memberships_.items()})
        return self

    def transform(self, requests):
        query = requests[self.keys].copy()
        query[self.keys[1]] = pd.to_datetime(query[self.keys[1]]).clip(upper=self.phase_boundary)
        return super().transform(query).rename(columns=lambda c:c.replace("deep__","completed_peer__",1))
