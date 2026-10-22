"""Opt-in, bounded conjunctions of existing schema-derived peer contexts.

No topology search or default fitting behavior changes. Existing GraphReduce
membership extraction enforces source-row cutoffs; the cutoff-local provider
enforces label maturity before sampling. Keys remain join metadata, never
downstream features. Pair selection does not inspect labels or held-out scores.
"""

from dataclasses import dataclass, field
from itertools import combinations
import json

import numpy as np
import pandas as pd

from .peer_sampling import CutoffLocalPeerContext


def _bounded_memberships(rows, limit):
    result = rows[["__example_id", "__neighbor_key_0"]].dropna().drop_duplicates().copy()
    # Memberships originate from one declared key column. Canonical strings
    # accommodate numeric/string keys without ever exposing IDs to CatBoost.
    result["__neighbor_key_0"] = result["__neighbor_key_0"].map(
        lambda value: json.dumps(
            value.item() if isinstance(value, np.generic) else value,
            ensure_ascii=True, separators=(",", ":"), default=str,
        )
    )
    result["__rank"] = pd.util.hash_pandas_object(
        result[["__neighbor_key_0"]], index=False
    ).to_numpy()
    result = result.sort_values(
        ["__example_id", "__rank", "__neighbor_key_0"], kind="stable"
    ).groupby("__example_id", sort=False).head(limit)
    return result.drop(columns="__rank").reset_index(drop=True)


def _paired_memberships(left, right, limit):
    # Each input is bounded BEFORE this join: at most n*n rows per example,
    # never a global historical-root/query-root Cartesian join.
    result = left.merge(right, on="__example_id", suffixes=("_left", "_right"))
    result["__neighbor_key_0"] = [
        json.dumps([a, b], separators=(",", ":"))
        for a, b in zip(result.__neighbor_key_0_left, result.__neighbor_key_0_right)
    ]
    result = result[["__example_id", "__neighbor_key_0"]].drop_duplicates()
    result["__rank"] = pd.util.hash_pandas_object(
        result[["__neighbor_key_0"]], index=False
    ).to_numpy()
    result = result.sort_values(
        ["__example_id", "__rank", "__neighbor_key_0"], kind="stable"
    ).groupby("__example_id", sort=False).head(limit)
    return result.drop(columns="__rank").reset_index(drop=True)


def _reduce_peers(history_members, query_members, latest, query, *, entity, target,
                  prior_strength, minimum_support, prior=None):
    """Group-level leave-own-root-out readout, not a peer-pair materialization."""
    names = ("mean", "shrunk_mean", "std", "support_mean", "support_max",
             "groups", "supported_groups", "coverage")
    result = pd.DataFrame(np.nan, index=query.index, columns=names)
    result[["support_mean", "support_max", "groups", "supported_groups", "coverage"]] = 0.
    if query_members.empty or latest.empty:
        return result
    links = history_members.loc[history_members.__example_id.isin(latest.index)].copy()
    indices = links.__example_id.to_numpy(dtype=int)
    links["__root"] = latest.loc[indices, entity].to_numpy()
    links["__y"] = latest.loc[indices, target].to_numpy(dtype=float)
    links = links.drop_duplicates(["__root", "__neighbor_key_0"])
    links["__square"] = links.__y.pow(2)
    stats = links.groupby("__neighbor_key_0").agg(
        count=("__root", "size"), total=("__y", "sum"), square=("__square", "sum")
    )
    current = query_members.copy()
    current["__root"] = query.loc[current.__example_id, entity].to_numpy()
    current = current.merge(stats, left_on="__neighbor_key_0", right_index=True, how="left")
    own = links[["__root", "__neighbor_key_0", "__y"]].rename(columns={"__y": "__own"})
    current = current.merge(own, on=["__root", "__neighbor_key_0"], how="left", validate="many_to_one")
    count = current["count"].fillna(0) - current.__own.notna().astype(int)
    total = current.total.fillna(0) - current.__own.fillna(0)
    square = current.square.fillna(0) - current.__own.fillna(0).pow(2)
    global_own = current.__root.map(latest.set_index(entity)[target])
    global_prior = (latest[target].sum() - global_own.fillna(0)) / (
        len(latest) - global_own.notna().astype(int)
    ).replace(0, np.nan)
    broad_prior = global_prior if prior is None else current.__example_id.map(prior).fillna(global_prior)
    supported = count.ge(minimum_support)
    mean = total / count.replace(0, np.nan)
    shrunk = (total + prior_strength * broad_prior) / (count + prior_strength)
    # Unsupported conjunctions fall back to their already leave-own-out
    # single-context prior, rather than inventing a confident tiny-group mean.
    current["shrunk_mean"] = shrunk.where(supported, broad_prior)
    current["mean"] = mean.where(supported)
    current["std"] = np.sqrt((square / count.replace(0, np.nan) - mean.pow(2)).clip(lower=0)).where(supported)
    current["support"] = count
    current["supported"] = supported.astype(float)
    pooled = current.groupby("__example_id").agg(
        mean=("mean", "mean"), shrunk_mean=("shrunk_mean", "mean"), std=("std", "mean"),
        support_mean=("support", "mean"), support_max=("support", "max"),
        groups=("support", "size"), supported_groups=("supported", "sum"), coverage=("supported", "mean"),
    )
    result.loc[pooled.index, names] = pooled[list(names)].to_numpy()
    return result


@dataclass
class PairedPeerContext:
    """Matched single contexts plus their bounded pairwise intersections.

    This composes the existing peer provider, not a new graph feature family.
    The historical pool is filtered at each query cutoff before sampling;
    memberships use each label's original prediction timestamp, and only the
    latest eligible label per sampled entity contributes. Pair shrinkage uses
    the corresponding broader single-context priors. All budgets are explicit.
    """

    reference: object
    history: pd.DataFrame
    max_contexts: int = 4
    max_pairs: int = 6
    max_neighbors: int = 8
    max_pair_memberships: int = 32
    prior_strength: float = 5.
    minimum_support: int = 2
    audit: list = field(default_factory=list, init=False)

    def __post_init__(self):
        if min(self.max_contexts, self.max_pairs, self.max_neighbors,
               self.max_pair_memberships, self.minimum_support) < 1 or self.prior_strength <= 0:
            raise ValueError("Positive paired-context budgets required")
        self.local_ = CutoffLocalPeerContext(self.reference, self.history)
        peer = self.reference.peer_
        ranked = []
        for path in peer.paths_:
            rows = peer.memberships_[path].copy()
            rows["__root"] = peer.history_.loc[rows["__example_id"], peer.entity_key_].to_numpy()
            rows = rows.drop_duplicates(["__root", "__neighbor_key_0"])
            degree = rows.groupby("__neighbor_key_0")["__root"].nunique()
            shared = rows.loc[rows["__neighbor_key_0"].isin(degree.index[degree.gt(1)])]
            coverage = shared["__root"].nunique() / max(peer.history_[peer.entity_key_].nunique(), 1)
            ranked.append((-coverage, len(path), path))
        self.path_ranking_ = [dict(path=path, shared_root_coverage=-coverage)
                              for coverage, _, path in sorted(ranked)]
        self.paths_ = tuple(path for coverage, _, path in sorted(ranked)
                            if coverage < 0)[:self.max_contexts]
        if len(self.paths_) < 2:
            raise ValueError("At least two schema-derived peer contexts required")
        candidates = list(combinations(range(len(self.paths_)), 2))
        uses = [0] * len(self.paths_)
        pairs = []
        while candidates and len(pairs) < self.max_pairs:
            pair = min(candidates, key=lambda p: (uses[p[0]] + uses[p[1]], p))
            candidates.remove(pair)
            pairs.append(pair)
            uses[pair[0]] += 1
            uses[pair[1]] += 1
        self.pairs_ = tuple(pairs)

    def transform(self, examples):
        base = self.reference.peer_
        entity, timestamp, target, available = (
            base.entity_key_, base.timestamp_, base.target_, base.available_at_
        )
        query = examples[[entity, timestamp]].reset_index(drop=True)
        if query.isna().any().any():
            raise ValueError("Complete query identities and timestamps required")
        single_columns = [f"paired_single_p{i}__{name}" for i in range(len(self.paths_))
                          for name in ("mean", "shrunk_mean", "std", "support_mean", "support_max", "groups", "supported_groups", "coverage")]
        pair_columns = [f"paired_joint_p{i}__{name}" for i in range(len(self.pairs_))
                        for name in ("mean", "shrunk_mean", "std", "support_mean", "support_max", "groups", "supported_groups", "coverage")]
        if query.empty:
            return pd.DataFrame(index=query.index, columns=single_columns, dtype="float32"), pd.DataFrame(index=query.index, columns=pair_columns, dtype="float32")
        singles, joints = [], []
        for cutoff, positions in query.groupby(timestamp, sort=True).indices.items():
            provider = self.local_._snapshot(cutoff)
            peer = provider.peer_
            latest = peer.history_.loc[pd.to_datetime(peer.history_[available]).lt(pd.Timestamp(cutoff))]
            latest = latest.sort_values([available, timestamp], kind="stable").drop_duplicates(entity, keep="last")
            block = query.iloc[positions].reset_index(drop=True)
            hm, qm, readouts = [], [], []
            for path in self.paths_:
                hm.append(_bounded_memberships(peer.memberships_[path], self.max_neighbors))
                qm.append(_bounded_memberships(peer._members(block, provider.tables_, provider.relationships_, path), self.max_neighbors))
                readouts.append(_reduce_peers(hm[-1], qm[-1], latest, block, entity=entity, target=target,
                                             prior_strength=self.prior_strength, minimum_support=self.minimum_support))
            paired = []
            for left, right in self.pairs_:
                hist = _paired_memberships(hm[left], hm[right], self.max_pair_memberships)
                asked = _paired_memberships(qm[left], qm[right], self.max_pair_memberships)
                broad = pd.concat([readouts[left].shrunk_mean, readouts[right].shrunk_mean], axis=1).mean(axis=1)
                paired.append(_reduce_peers(hist, asked, latest, block, entity=entity, target=target,
                                            prior_strength=self.prior_strength, minimum_support=self.minimum_support, prior=broad))
            one = pd.concat([r.add_prefix(f"paired_single_p{i}__") for i, r in enumerate(readouts)], axis=1)
            two = pd.concat([r.add_prefix(f"paired_joint_p{i}__") for i, r in enumerate(paired)], axis=1)
            one.index = two.index = positions
            singles.append(one)
            joints.append(two)
            self.audit.append(dict(cutoff=str(cutoff), query_rows=len(block), history_rows=len(peer.history_),
                                   history_memberships=[len(m) for m in hm],
                                   joint_coverage=[int(r.supported_groups.gt(0).sum()) for r in paired]))
        return (pd.concat(singles).sort_index().reindex(columns=single_columns).astype("float32"),
                pd.concat(joints).sort_index().reindex(columns=pair_columns).astype("float32"))

    def contract(self):
        return dict(paths=self.paths_, path_ranking=self.path_ranking_, pairs=self.pairs_, max_neighbors=self.max_neighbors,
                    max_pair_memberships=self.max_pair_memberships, prior_strength=self.prior_strength,
                    minimum_support=self.minimum_support, history_sampling="strict_maturity_before_bottom_hash",
                    shrinkage="leave-own-root-out single-context priors", production_default=False)
