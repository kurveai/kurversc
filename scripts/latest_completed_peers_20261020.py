"""Bounded existing peer readout with maturity/latest selection before sampling.

Older repeats of one root must not consume slots needed by another root's
latest known outcome. This changes neither the frozen schema-derived paths nor
the existing leave-own-root-out reducers. It is an optional research provider.
"""

import copy
from dataclasses import replace

import numpy as np
import pandas as pd

from scripts.completed_peer_extension import CompletedPeerExtension


def select_latest(history, *, entity, timestamp, available_at, cutoff, limit):
    keys = [entity, timestamp, available_at]
    if limit < 1 or history[keys].isna().any().any() or history.duplicated([entity, timestamp]).any():
        raise ValueError("Unique complete clocks/identities and a positive peer budget required")
    ready = pd.to_datetime(history[available_at])
    selected = history.loc[ready.le(pd.Timestamp(cutoff))].sort_values(
        [available_at, timestamp, entity], kind="stable").drop_duplicates(entity, keep="last")
    selected = selected.sort_values([timestamp, entity], kind="stable")
    if len(selected) > limit:
        hashes = pd.util.hash_pandas_object(selected[[entity, timestamp]], index=False).to_numpy()
        positions = np.argsort(hashes, kind="stable")[:limit]
        selected = selected.iloc[np.sort(positions)]
    return selected.reset_index(drop=True)


class LatestCompletedPeers(CompletedPeerExtension):
    def fit(self, history, **kwargs):
        super().fit(history, **kwargs)
        self.full_history = history.copy()
        self.maximum_history = kwargs.get("max_history_rows", 50_000)
        self.snapshot_audit = []
        peer = self.provider.peer_
        fields = [f"{name}__{stat}" for name in ("support", "mean", "std", "shrunk_mean", "supported")
            for stat in ("mean", "max", "min")] + ["shared_entities", "supported_entities"]
        weighted = ("weighted_mean", "weighted_std", "weighted_shrunk_mean", "weighted_mass",
            "weighted_group_count", "weighted_peer_support", "weighted_recentness_days", "weighted_max_group_mean")
        self.output_columns = [f"peer_p{i}__{n}" for i in range(len(peer.paths_)) for n in fields]
        self.output_columns += [f"weighted_p{i}__{n}" for i in range(len(peer.paths_)) for n in weighted]
        self.audit.update(history_sampling="maturity_then_latest_per_root_then_bounded_identity_hash",
            original_native_peer_columns_unchanged=True)
        return self

    def transform(self, requests):
        base = self.provider.peer_
        entity, timestamp, available = base.entity_key_, base.timestamp_, base.available_at_
        query = requests[self.keys].reset_index(drop=True).copy()
        query[timestamp] = pd.to_datetime(query[timestamp]).clip(upper=self.phase_boundary)
        if query.isna().any().any() or query.empty:
            raise ValueError("Complete nonempty query identities/clocks required")
        blocks = []
        for clock, positions in query.groupby(timestamp, sort=True).indices.items():
            history = select_latest(self.full_history, entity=entity, timestamp=timestamp,
                available_at=available, cutoff=clock, limit=self.maximum_history)
            peer = copy.copy(base)
            peer.history_, peer.memberships_, peer.audit = history.copy(), {}, []
            members_at = history[[entity, available]].rename(columns={available: timestamp})
            for path in base.paths_:
                if history.empty:
                    members = pd.DataFrame({"__example_id": pd.Series(dtype="int64"),
                        "__neighbor_key_0": pd.Series(dtype="object")})
                else:
                    members = peer._members(members_at, self.provider.tables_, self.provider.relationships_, path)
                peer.memberships_[path] = members.drop_duplicates(["__example_id", "__neighbor_key_0"])
            # Only label eligibility changes <= to the existing strict reducer;
            # actual source membership extraction above uses TRUE completion.
            peer.history_[available] = pd.to_datetime(peer.history_[available])-pd.Timedelta(1, "ns")
            provider = replace(self.provider)
            provider.peer_, provider.tables_, provider.relationships_ = peer, self.provider.tables_, self.provider.relationships_
            provider.source_history_rows_, provider.history_rows_ = len(self.full_history), len(history)
            values = provider.transform(query.iloc[positions].reset_index(drop=True)).drop(columns=self.keys)
            values = values.reindex(columns=self.output_columns)
            for column in values:
                if column.endswith(("weighted_mass", "weighted_group_count", "weighted_peer_support")):
                    values[column] = values[column].fillna(0.)
            values.index = positions
            blocks.append(values)
            self.snapshot_audit.append(dict(cutoff=str(clock), sampled_latest_roots=len(history),
                latest_complete_at=None if history.empty else str(pd.to_datetime(history[available]).max()),
                memberships=sum(len(v) for v in peer.memberships_.values())))
        return pd.concat(blocks).sort_index().reset_index(drop=True).add_prefix("latest_completed__")
