"""Opt-in cutoff-local sampling for an already frozen peer-context plan.

Keep the ordinary strict maturity and leave-own-root-out rules. Unlike the
legacy global history sample, unavailable labels cannot displace mature rows.
This provider does not add graph configurations or change default fitting.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .peer_context import PeerContext

_WEIGHTED_NAMES = (
    "weighted_mean",
    "weighted_std",
    "weighted_shrunk_mean",
    "weighted_mass",
    "weighted_group_count",
    "weighted_peer_support",
    "weighted_recentness_days",
    "weighted_max_group_mean",
)


def sample_mature_peer_history(
    history, *, entity, timestamp, available_at, cutoff, limit
):
    """Target-blind bottom-hash sample AFTER strict label eligibility.

    Canonical sorting also breaks hash ties reproducibly. Target values do not
    participate in selection, and changing input row order has no effect.
    """
    if limit < 1:
        raise ValueError("The peer history row budget must be positive")
    if history[[entity, timestamp, available_at]].isna().any().any():
        raise ValueError(
            "Complete peer identities and maturity timestamps are required"
        )
    if history.duplicated([entity, timestamp]).any():
        raise ValueError("Peer history requires unique entity/time labels")
    eligible = history.loc[
        pd.to_datetime(history[available_at]).lt(pd.Timestamp(cutoff))
    ].sort_values([timestamp, entity], kind="stable")
    if len(eligible) > limit:
        hashes = pd.util.hash_pandas_object(
            eligible[[entity, timestamp]], index=False
        ).to_numpy()
        positions = np.argsort(hashes, kind="stable")[:limit]
        eligible = eligible.iloc[np.sort(positions)]
    return eligible.reset_index(drop=True)


@dataclass
class CutoffLocalPeerContext:
    """Regenerate bounded memberships from strictly mature labels per cutoff.

    ``reference`` supplies a frozen schema-derived plan and fixed path/neighbor
    budgets, independent of the current query or future label additions. The
    full history stores only four metadata/label columns; memberships are
    materialized for at most ``reference.max_history_rows`` rows per snapshot.
    One snapshot is retained by default. Table sources are shared, not copied.

    This is an experimental reusable provider, not a default policy. Historical
    memberships are still extracted at the original prediction timestamp, and
    query roots are excluded by the unchanged PeerHistoryFamily implementation.
    """

    reference: PeerContext
    history: pd.DataFrame
    snapshot_cache_entries: int = 1
    audit: list = field(default_factory=list, init=False)
    _snapshots: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)

    def __post_init__(self):
        if not hasattr(self.reference, "peer_"):
            raise ValueError("A fitted, frozen peer-context reference is required")
        if self.snapshot_cache_entries < 1:
            raise ValueError("At least one bounded snapshot cache entry is required")
        peer = self.reference.peer_
        required = [peer.entity_key_, peer.timestamp_, peer.target_, peer.available_at_]
        self.history = self.history[required].copy()
        if self.history[required].isna().any().any():
            raise ValueError("Complete peer history is required")
        if self.history.duplicated([peer.entity_key_, peer.timestamp_]).any():
            raise ValueError("Peer history requires unique entity/time labels")
        if not np.isfinite(self.history[peer.target_].to_numpy(dtype=float)).all():
            raise ValueError("Historical targets must be finite")
        if (
            pd.to_datetime(self.history[peer.available_at_])
            < pd.to_datetime(self.history[peer.timestamp_])
        ).any():
            raise ValueError("Label availability cannot precede prediction time")

    def _snapshot(self, cutoff):
        cutoff = pd.Timestamp(cutoff)
        cached = self._snapshots.get(cutoff)
        if cached is not None:
            self._snapshots.move_to_end(cutoff)
            return cached
        base = self.reference.peer_
        selected = sample_mature_peer_history(
            self.history,
            entity=base.entity_key_,
            timestamp=base.timestamp_,
            available_at=base.available_at_,
            cutoff=cutoff,
            limit=self.reference.max_history_rows,
        )
        peer = copy.copy(base)
        peer.history_, peer.memberships_, peer.audit = selected, {}, []
        for path in base.paths_:
            rows = peer._members(
                selected[[base.entity_key_, base.timestamp_]],
                self.reference.tables_,
                self.reference.relationships_,
                path,
            ).drop_duplicates(["__example_id", "__neighbor_key_0"])
            peer.memberships_[path] = rows
            peer.audit.append({"path": path, "memberships": len(rows)})
        # dataclasses.replace gives this snapshot its own result cache; it
        # cannot reuse values computed with the reference's global sample.
        provider = replace(self.reference)
        provider.peer_ = peer
        provider.tables_ = self.reference.tables_
        provider.relationships_ = self.reference.relationships_
        provider.source_history_rows_ = len(self.history)
        provider.history_rows_ = len(selected)
        self.audit.append(
            {
                "cutoff": str(cutoff),
                "eligible_history_rows": int(
                    pd.to_datetime(self.history[base.available_at_]).lt(cutoff).sum()
                ),
                "selected_history_rows": len(selected),
                "selected_historical_roots": int(selected[base.entity_key_].nunique()),
                "memberships": sum(len(rows) for rows in peer.memberships_.values()),
            }
        )
        self._snapshots[cutoff] = provider
        while len(self._snapshots) > self.snapshot_cache_entries:
            self._snapshots.popitem(last=False)
        return provider

    def transform(self, examples):
        peer = self.reference.peer_
        keys = [peer.entity_key_, peer.timestamp_]
        query = examples[keys].reset_index(drop=True)
        if query[keys].isna().any().any():
            raise ValueError("Complete query identities and timestamps are required")
        if query.empty:
            return self.reference.transform(query)
        pieces = []
        for cutoff, positions in query.groupby(
            peer.timestamp_, sort=True
        ).indices.items():
            provider = self._snapshot(cutoff)
            block = provider.transform(query.iloc[positions].reset_index(drop=True))
            block.index = positions
            pieces.append(block)
        result = pd.concat(pieces).sort_index().reset_index(drop=True)
        if self.reference.mode == "both":
            # The legacy weighted helper skips paths with empty memberships.
            # A cutoff-local snapshot may be empty even though the frozen plan
            # has those columns. Preserve that plan's schema and zero-support
            # convention rather than dropping columns at cold-start cutoffs.
            weighted_columns = [
                f"weighted_p{i}__{name}"
                for i, path in enumerate(peer.paths_)
                if not peer.memberships_[path].empty
                for name in _WEIGHTED_NAMES
            ]
            for column in weighted_columns:
                if column not in result:
                    result[column] = (
                        0.0
                        if column.endswith(
                            (
                                "weighted_mass",
                                "weighted_group_count",
                                "weighted_peer_support",
                            )
                        )
                        else np.nan
                    )
            generated = {
                f"weighted_p{i}__{name}"
                for i in range(len(peer.paths_))
                for name in _WEIGHTED_NAMES
            }
            result = result.reindex(
                columns=[
                    *[c for c in result if c not in generated],
                    *weighted_columns,
                ]
            )
        return result

    def contract(self):
        return {
            **self.reference.contract(),
            "history_rows": max(
                (p.history_rows_ for p in self._snapshots.values()), default=0
            ),
            "history_sampling": "strict_maturity_before_bottom_hash",
            "source_history_rows": len(self.history),
            "snapshot_cache_entries": self.snapshot_cache_entries,
        }
