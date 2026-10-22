"""Bounded, point-in-time peer-label context for relational predictions.

The path plan is schema-derived and fitted only on observed labels. Historical
labels are sampled without looking at target values when a task is too large
to retain all peer memberships. Query roots never contribute their own labels.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib

import numpy as np
import pandas as pd

from .peer_history import PeerHistoryFamily


def weighted_peer_features(
    peer: PeerHistoryFamily,
    examples: pd.DataFrame,
    *,
    tables,
    relationships,
    half_life_days: float = 730.0,
    prior_strength: float = 5.0,
) -> pd.DataFrame:
    """Add recency/specificity-weighted moments of other roots' mature labels."""

    if half_life_days <= 0 or prior_strength <= 0:
        raise ValueError("Weighting constants must be positive")
    if not examples.index.equals(pd.RangeIndex(len(examples))):
        raise ValueError("Examples need a stable zero-based RangeIndex")
    entity, timestamp, target, available = (
        peer.entity_key_, peer.timestamp_, peer.target_, peer.available_at_
    )
    history = peer.history_
    parts = [examples[[entity, timestamp]].copy()]
    for path_index, path in enumerate(peer.paths_):
        history_members = peer.memberships_[path]
        if history_members.empty:
            continue
        queried = peer._members(examples, tables, relationships, path)
        queried = queried.drop_duplicates(["__example_id", "__neighbor_key_0"])
        names = (
            "weighted_mean", "weighted_std", "weighted_shrunk_mean",
            "weighted_mass", "weighted_group_count", "weighted_peer_support",
            "weighted_recentness_days", "weighted_max_group_mean",
        )
        output = pd.DataFrame(index=examples.index, columns=names, dtype="float64")
        output[["weighted_mass", "weighted_group_count", "weighted_peer_support"]] = 0.0
        for cutoff, query_frame in examples.groupby(timestamp, sort=True):
            cutoff = pd.Timestamp(cutoff)
            eligible = history.loc[pd.to_datetime(history[available]).lt(cutoff)]
            latest = (
                eligible.sort_values([available, timestamp], kind="stable")
                .drop_duplicates(entity, keep="last")
            )
            if latest.empty:
                continue
            links = history_members.loc[
                history_members.__example_id.isin(latest.index)
            ].copy()
            if links.empty:
                continue
            hist_ids = links.__example_id.to_numpy(dtype=int)
            links["root"] = history.loc[hist_ids, entity].to_numpy()
            links["y"] = history.loc[hist_ids, target].to_numpy(dtype=float)
            maturity = pd.to_datetime(history.loc[hist_ids, available])
            age_days = (cutoff - maturity).dt.total_seconds().to_numpy() / 86400
            links["age_days"] = age_days
            links["w"] = np.exp2(-age_days / half_life_days)
            links = links.drop_duplicates(["root", "__neighbor_key_0"])
            links["wy"] = links.w * links.y
            links["wy2"] = links.w * links.y.pow(2)
            links["wage"] = links.w * links.age_days
            key = "__neighbor_key_0"
            groups = links.groupby(key).agg(
                support=("root", "size"),
                total_w=("w", "sum"),
                total_wy=("wy", "sum"),
                total_wy2=("wy2", "sum"),
                total_wage=("wage", "sum"),
            )
            query = queried.loc[queried.__example_id.isin(query_frame.index)].copy()
            if query.empty:
                continue
            query["root"] = examples.loc[query.__example_id, entity].to_numpy()
            query = query.merge(groups, left_on=key, right_index=True, how="left")
            own = links[["root", key, "w", "wy", "wy2", "wage"]].rename(
                columns={
                    "w": "own_w", "wy": "own_wy", "wy2": "own_wy2",
                    "wage": "own_wage",
                }
            )
            query = query.merge(
                own, on=["root", key], how="left", validate="many_to_one"
            )
            own_w = query.own_w.fillna(0)
            support = query.support.fillna(0) - query.own_w.notna().astype(int)
            priority = np.where(support.gt(0), 1.0 / np.sqrt(support), 0.0)
            query["mass"] = priority * (query.total_w.fillna(0) - own_w)
            query["ymass"] = priority * (
                query.total_wy.fillna(0) - query.own_wy.fillna(0)
            )
            query["y2mass"] = priority * (
                query.total_wy2.fillna(0) - query.own_wy2.fillna(0)
            )
            query["agemass"] = priority * (
                query.total_wage.fillna(0) - query.own_wage.fillna(0)
            )
            query["supported"] = support.gt(0).astype(float)
            query["support"] = support.clip(lower=0)
            query["group_mean"] = (
                query.total_wy.fillna(0) - query.own_wy.fillna(0)
            ) / (query.total_w.fillna(0) - own_w).replace(0, np.nan)
            pooled = query.groupby("__example_id").agg(
                mass=("mass", "sum"), ymass=("ymass", "sum"),
                y2mass=("y2mass", "sum"), agemass=("agemass", "sum"),
                group_count=("supported", "sum"),
                peer_support=("support", "sum"),
                max_group_mean=("group_mean", "max"),
            )
            mean = pooled.ymass / pooled.mass.replace(0, np.nan)
            variance = pooled.y2mass / pooled.mass.replace(0, np.nan) - mean.pow(2)
            global_age = (
                cutoff - pd.to_datetime(latest[available])
            ).dt.total_seconds().to_numpy() / 86400
            global_w = np.exp2(-global_age / half_life_days)
            global_y = latest[target].to_numpy(dtype=float)
            global_mass = float(global_w.sum())
            global_ymass = float(np.dot(global_w, global_y))
            root_weights = pd.DataFrame(
                {
                    "root": latest[entity].to_numpy(), "w": global_w,
                    "wy": global_w * global_y,
                }
            ).set_index("root")
            query_roots = examples.loc[pooled.index, entity]
            own_global_w = query_roots.map(root_weights.w).fillna(0).to_numpy()
            own_global_wy = query_roots.map(root_weights.wy).fillna(0).to_numpy()
            prior = (global_ymass - own_global_wy) / np.where(
                global_mass - own_global_w > 0,
                global_mass - own_global_w,
                np.nan,
            )
            output.loc[pooled.index, "weighted_mean"] = mean.to_numpy()
            output.loc[pooled.index, "weighted_std"] = np.sqrt(
                variance.clip(lower=0)
            ).to_numpy()
            output.loc[pooled.index, "weighted_shrunk_mean"] = (
                pooled.ymass.to_numpy() + prior_strength * prior
            ) / (pooled.mass.to_numpy() + prior_strength)
            output.loc[pooled.index, "weighted_mass"] = pooled.mass.to_numpy()
            output.loc[pooled.index, "weighted_group_count"] = pooled.group_count.to_numpy()
            output.loc[pooled.index, "weighted_peer_support"] = pooled.peer_support.to_numpy()
            output.loc[pooled.index, "weighted_recentness_days"] = (
                pooled.agemass / pooled.mass.replace(0, np.nan)
            ).to_numpy()
            output.loc[pooled.index, "weighted_max_group_mean"] = (
                pooled.max_group_mean.to_numpy()
            )
        parts.append(output.add_prefix(f"weighted_p{path_index}__"))
    return pd.concat(parts, axis=1)


@dataclass
class PeerContext:
    """Reusable peer features for search, final fitting, and prediction.

    The fixed row/neighbor budgets are schema-generic. Label-history sampling
    depends on entity/time only, never the target, and is applied only when the
    full history exceeds the budget. Point-in-time and leave-own-root-out rules
    are enforced by :class:`PeerHistoryFamily` and ``weighted_peer_features``.
    """

    mode: str = "both"
    max_history_rows: int = 50_000
    max_paths: int = 8
    max_neighbors: int = 128
    large_max_paths: int = 4
    large_max_neighbors: int = 32
    batch_rows: int = 10_000
    cache_rows: int = 50_000
    cache_entries: int = 8
    _cache: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)

    def fit(self, training, *, tables, relationships, root_name, entity_key, timestamp, target, available_at):
        if self.mode not in {"unweighted", "both"}:
            raise ValueError("Peer context mode must be 'unweighted' or 'both'")
        if min(self.max_history_rows, self.batch_rows, self.max_paths, self.max_neighbors) < 1:
            raise ValueError("Peer context budgets must be positive")
        if training.duplicated([entity_key, timestamp]).any():
            raise ValueError("Peer history requires unique entity/time labels")
        self.source_history_rows_ = len(training)
        large = len(training) > self.max_history_rows
        if large:
            identities = training[[entity_key, timestamp]]
            hashed = pd.util.hash_pandas_object(identities, index=False).to_numpy()
            positions = np.argpartition(hashed, self.max_history_rows - 1)[
                : self.max_history_rows
            ]
            training = training.iloc[np.sort(positions)].copy()
        self.history_rows_ = len(training)
        self.tables_ = tables
        self.relationships_ = relationships
        self.peer_ = PeerHistoryFamily(
            max_paths=self.large_max_paths if large else self.max_paths,
            max_neighbors=self.large_max_neighbors if large else self.max_neighbors,
        ).fit(
            training,
            tables=tables,
            relationships=relationships,
            root_name=root_name,
            entity_key=entity_key,
            timestamp=timestamp,
            target=target,
            available_at=available_at,
        )
        self._cache.clear()
        return self

    def contract(self) -> dict:
        return {
            "mode": self.mode,
            "source_history_rows": self.source_history_rows_,
            "history_rows": self.history_rows_,
            "max_history_rows": self.max_history_rows,
            "max_paths": self.peer_.max_paths,
            "max_neighbors": self.peer_.max_neighbors,
            "batch_rows": self.batch_rows,
            "paths": tuple(self.peer_.paths_),
        }

    def transform(self, examples: pd.DataFrame) -> pd.DataFrame:
        peer = self.peer_
        keys = [peer.entity_key_, peer.timestamp_]
        query = examples[keys].reset_index(drop=True)
        fingerprint = None
        if len(query) <= self.cache_rows:
            row_hash = pd.util.hash_pandas_object(query, index=False).to_numpy()
            fingerprint = hashlib.sha256(row_hash.tobytes()).hexdigest()
            cached = self._cache.get(fingerprint)
            if cached is not None:
                self._cache.move_to_end(fingerprint)
                return cached.copy()
        pieces = []
        for start in range(0, len(query), self.batch_rows):
            chunk = query.iloc[start : start + self.batch_rows].reset_index(drop=True)
            plain = peer.transform(
                chunk, tables=self.tables_, relationships=self.relationships_
            )
            if self.mode == "both":
                weighted = weighted_peer_features(
                    peer, chunk, tables=self.tables_, relationships=self.relationships_
                )
                plain = pd.concat(
                    [plain, weighted.drop(columns=keys)], axis=1
                )
            pieces.append(plain)
        result = pd.concat(pieces, ignore_index=True) if pieces else query.copy()
        if fingerprint is not None:
            self._cache[fingerprint] = result
            while len(self._cache) > self.cache_entries:
                self._cache.popitem(last=False)
        return result
