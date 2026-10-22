"""Schema-derived, strictly historical label transfer through shared entities."""

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .learned_messages import LearnedMessageFamily
from .relational_inputs import key_parts, memberships, schema_paths


@dataclass
class PeerHistoryFamily:
    """Summarize other roots' mature labels at shared terminal entities.

    Memberships are extracted at each label's ORIGINAL prediction time. Per
    query cutoff use only the latest mature label per historical root. Remove
    the query root from both the group statistics and the shrinkage prior.
    IDs are relational metadata only and are never emitted as model features.
    """

    max_depth: int = 2
    max_paths: int = 8
    max_neighbors: int = 128
    prior_strength: float = 20.0
    random_state: int = 0
    memory_limit: str = "4GB"
    audit: list = field(default_factory=list, init=False)

    as_table = LearnedMessageFamily.as_table

    @staticmethod
    def _stable_membership_tables(tables, relationships, paths):
        """Give unkeyed intermediate events a content-stable ranking key.

        GraphReduce bounds each hop with ``row_number``. Without a primary key
        it ranks on the incoming foreign key, which ties for every event of a
        busy entity; the selected neighbors then depend on DuckDB scan order.
        A hash of the structural event identity breaks those ties without
        looking at labels or payloads. Identical structural rows may still
        tie, but they lead to identical peer memberships.
        """

        intermediate = {name for path in paths for name in path[:-1]}
        stable = dict(tables)
        for name in sorted(intermediate):
            table = tables[name]
            frame = table.source
            if not isinstance(frame, pd.DataFrame):
                continue
            key = key_parts(table.key)
            if key and not frame.duplicated(list(key)).any():
                continue
            structural = set(key)
            if table.date is not None:
                structural.add(table.date)
            for edge in relationships:
                if edge.parent == name:
                    structural.update(key_parts(edge.parent_key))
                if edge.child == name:
                    structural.update(key_parts(edge.child_key))
            columns = sorted(structural.intersection(frame.columns))
            synthetic = "__kurversc_peer_event_key__"
            while synthetic in frame.columns:
                synthetic += "_"
            keyed = frame.copy(deep=False)
            keyed[synthetic] = pd.util.hash_pandas_object(
                keyed[columns], index=False
            ).to_numpy(dtype="uint64")
            stable[name] = replace(table, source=keyed, key=synthetic)
        return stable

    def _members(self, examples, tables, relationships, path):
        return memberships(
            examples,
            self._membership_tables_,
            relationships,
            root=self.root_name_,
            entity=self.entity_key_,
            timestamp=self.timestamp_,
            path=path,
            max_neighbors=self.max_neighbors,
            seed=self.random_state,
            memory_limit=self.memory_limit,
        )

    def fit(
        self,
        training,
        *,
        tables,
        relationships,
        root_name,
        entity_key,
        timestamp,
        target,
        available_at,
    ):
        if (
            min(self.max_depth, self.max_paths, self.max_neighbors) < 1
            or self.prior_strength <= 0
        ):
            raise ValueError("Peer budgets and prior strength must be positive")
        required = [entity_key, timestamp, target, available_at]
        history = training[required].copy()
        if history.empty or history[required].isna().any().any():
            raise ValueError("Complete nonempty historical labels are required")
        if history.duplicated([entity_key, timestamp]).any():
            raise ValueError(
                "One label per entity and prediction timestamp is required"
            )
        if not np.isfinite(history[target].to_numpy(dtype=float)).all():
            raise ValueError("Historical targets must be finite")
        if (
            pd.to_datetime(history[available_at]) < pd.to_datetime(history[timestamp])
        ).any():
            raise ValueError("Label availability cannot precede prediction time")
        self.root_name_, self.entity_key_, self.timestamp_ = (
            root_name,
            entity_key,
            timestamp,
        )
        self.root_key_, self.target_, self.available_at_ = (
            tables[root_name].key,
            target,
            available_at,
        )
        self.history_ = history.sort_values(
            [timestamp, entity_key], kind="stable"
        ).reset_index(drop=True)
        self.memberships_, self.audit = {}, []
        candidates = [
            (p, e)
            for p, e in schema_paths(root_name, relationships, self.max_depth)
            if not e.reduce
        ]
        self._membership_tables_ = self._stable_membership_tables(
            tables, relationships, [path for path, _ in candidates]
        )
        for path, edge in sorted(candidates, key=lambda p: (len(p[0]), p[0])):
            if len(key_parts(tables[path[-1]].key)) != 1:
                continue
            # Paths are schema-derived, never selected using target values or
            # future label availability. Empty paths retain an explicit schema.
            rows = self._members(
                self.history_[[entity_key, timestamp]], tables, relationships, path
            )
            rows = rows.drop_duplicates(["__example_id", "__neighbor_key_0"])
            self.memberships_[path] = rows
            self.audit.append(
                dict(
                    path=path,
                    memberships=len(rows),
                    historical_roots=int(rows["__example_id"].nunique()),
                )
            )
            if len(self.memberships_) >= self.max_paths:
                break
        self.paths_ = tuple(self.memberships_)
        return self

    def transform(self, examples, *, tables, relationships):
        if not hasattr(self, "history_"):
            raise ValueError("Fit the peer family before inference")
        if not examples.index.is_unique or examples[self.timestamp_].isna().any():
            raise ValueError("Examples need unique indexes and nonmissing timestamps")
        history = self.history_
        parts = [examples[[self.entity_key_, self.timestamp_]].copy()]
        for path_index, path in enumerate(self.paths_):
            query = self._members(
                examples, tables, relationships, path
            ).drop_duplicates()
            chunks = []
            for cutoff, frame in examples.groupby(self.timestamp_, sort=True):
                eligible = history.loc[
                    pd.to_datetime(history[self.available_at_]).lt(pd.Timestamp(cutoff))
                ]
                latest = eligible.sort_values(
                    [self.available_at_, self.timestamp_], kind="stable"
                ).drop_duplicates(self.entity_key_, keep="last")
                links = self.memberships_[path]
                links = links[links["__example_id"].isin(latest.index)].copy()
                links["__root"] = history.loc[
                    links["__example_id"], self.entity_key_
                ].to_numpy()
                links["__y"] = history.loc[
                    links["__example_id"], self.target_
                ].to_numpy(dtype=float)
                links["__y2"] = links["__y"] ** 2
                links = links.drop_duplicates(["__root", "__neighbor_key_0"])
                stats = links.groupby("__neighbor_key_0")["__y"].agg(["count", "sum"])
                stats["square_sum"] = links.groupby("__neighbor_key_0")["__y2"].sum()
                current = query[query["__example_id"].isin(frame.index)].copy()
                current["__root"] = examples.loc[
                    current["__example_id"], self.entity_key_
                ].to_numpy()
                current = current.merge(
                    stats, how="left", left_on="__neighbor_key_0", right_index=True
                )
                own = links[["__root", "__neighbor_key_0", "__y"]].rename(
                    columns={"__y": "own"}
                )
                current = current.merge(
                    own,
                    on=["__root", "__neighbor_key_0"],
                    how="left",
                    validate="many_to_one",
                )
                count = current["count"].fillna(0) - current.own.notna().astype(int)
                total = current["sum"].fillna(0) - current.own.fillna(0)
                square = current.square_sum.fillna(0) - current.own.fillna(0) ** 2
                global_own = current["__root"].map(
                    latest.set_index(self.entity_key_)[self.target_]
                )
                prior = (latest[self.target_].sum() - global_own.fillna(0)) / (
                    len(latest) - global_own.notna().astype(int)
                ).replace(0, np.nan)
                current["support"] = count
                current["mean"] = total / count.replace(0, np.nan)
                current["std"] = np.sqrt(
                    (square / count.replace(0, np.nan) - current["mean"] ** 2).clip(
                        lower=0
                    )
                )
                current["shrunk_mean"] = (total + self.prior_strength * prior) / (
                    count + self.prior_strength
                )
                current["supported"] = count.gt(0).astype(float)
                columns = ["support", "mean", "std", "shrunk_mean", "supported"]
                pooled = current.groupby("__example_id")[columns].agg(
                    ["mean", "max", "min"]
                )
                pooled.columns = [f"{col}__{stat}" for col, stat in pooled.columns]
                pooled = pooled.reindex(frame.index)
                pooled["shared_entities"] = (
                    current.groupby("__example_id")
                    .size()
                    .reindex(frame.index, fill_value=0)
                )
                pooled["supported_entities"] = (
                    current.groupby("__example_id")
                    .supported.sum()
                    .reindex(frame.index, fill_value=0)
                )
                chunks.append(pooled)
            parts.append(
                pd.concat(chunks)
                .reindex(examples.index)
                .astype("float64")
                .add_prefix(f"peer_p{path_index}__")
            )
        return pd.concat(parts, axis=1)

    def fit_transform(self, training, examples, **kwargs):
        self.fit(training, **kwargs)
        return self.transform(
            examples, tables=kwargs["tables"], relationships=kwargs["relationships"]
        )
