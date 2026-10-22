"""Bounded training-only compression of relational object co-occurrence.

Ordinary peer readouts average other roots' labels. This different view retains
which objects co-occur in a root's history, without labels, text tokens or raw
identifiers in the learner payload. A frozen TF-IDF/SVD basis emits <=32 scalar
state coordinates, reusing GraphReduce's strict, bounded path memberships.
"""

from collections import Counter
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

from kurversc.peer_history import PeerHistoryFamily
from kurversc.relational_inputs import key_parts, schema_paths
from kurversc.signal_challengers import ChallengerUnavailable


def canonical(values):
    return values.map(lambda v: json.dumps(v.item() if isinstance(v, np.generic) else v,
        ensure_ascii=True, default=str, separators=(",", ":")))


class IncidenceBasis:
    def __init__(self, *, dimensions=16, vocabulary=65536, maximum_nonzeros=5_000_000):
        if not 1 <= dimensions <= 16 or not 1 <= vocabulary <= 65536 or maximum_nonzeros < 1:
            raise ValueError("Bounded incidence representation budgets required")
        self.dimensions, self.maximum_vocabulary, self.maximum_nonzeros = dimensions, vocabulary, maximum_nonzeros

    def _matrix(self, rows, count):
        if rows.empty:
            return csr_matrix((count, len(self.vocabulary)), dtype=np.float32)
        if rows["__example_id"].min() < 0 or rows["__example_id"].max() >= count:
            raise ValueError("Memberships must refer to the declared query rows")
        columns = canonical(rows["__neighbor_key_0"]).map(self.vocabulary)
        known = columns.notna()
        ids = rows.loc[known, "__example_id"].to_numpy(dtype=int)
        if len(ids) > self.maximum_nonzeros:
            raise ChallengerUnavailable("Bounded association sparse capacity exceeded")
        matrix = csr_matrix((np.ones(len(ids), np.float32), (ids, columns.loc[known].to_numpy(dtype=int))),
            shape=(count, len(self.vocabulary)))
        matrix.sum_duplicates()
        # Membership duplicates are not additional events in this view.
        matrix.data[:] = 1.
        return matrix

    def fit(self, rows, count):
        rows = rows[["__example_id", "__neighbor_key_0"]].dropna().drop_duplicates()
        counts = Counter(canonical(rows["__neighbor_key_0"]))
        ranked = sorted(counts, key=lambda key: (-counts[key], key))[:self.maximum_vocabulary]
        self.vocabulary = {key: i for i, key in enumerate(ranked)}
        if count < 3 or len(ranked) < 3:
            raise ChallengerUnavailable("Insufficient reusable object incidence")
        matrix = self._matrix(rows, count)
        self.tfidf = TfidfTransformer(sublinear_tf=True).fit(matrix)
        weighted = self.tfidf.transform(matrix)
        width = min(self.dimensions, count-1, len(ranked)-1)
        self.reducer = TruncatedSVD(n_components=width, n_iter=7, random_state=0).fit(weighted)
        self.audit = dict(training_rows=count, vocabulary=len(ranked), nonzeros=matrix.nnz,
            dimensions=width, targets_used=False, query_corpus_used=False,
            identity_keys_are_grouping_metadata_not_predictor_columns=True)
        return self

    def transform(self, rows, count):
        matrix = self._matrix(rows, count)
        values = self.reducer.transform(self.tfidf.transform(matrix)).astype("float32")
        if not np.isfinite(values).all():
            raise ValueError("Finite bounded association state required")
        return values, np.asarray(matrix.getnnz(axis=1)>0)


class AssociationState:
    def fit_transform(self, train, query, *, tables, relationships, root_name, phase_boundary,
            entity="__event_entity", timestamp="__event_cutoff"):
        if len(train) > 250_000 or len(query) > 250_000:
            raise ChallengerUnavailable("Association-state row capacity exceeded")
        if any(getattr(t, "available_at", None) and t.available_at != t.date for t in tables.values()):
            raise ChallengerUnavailable("This isolated extractor requires ordinary event clocks without delayed arrival")
        phase = pd.Timestamp(phase_boundary)
        requests = [f[[entity, timestamp]].reset_index(drop=True).copy() for f in (train, query)]
        for frame in requests:
            frame[timestamp] = pd.to_datetime(frame[timestamp]).clip(upper=phase)
            if frame.isna().any().any():
                raise ValueError("Complete source routing clocks/identities required")
        edges = {(r.parent, r.child): r for r in relationships}
        candidates = []
        for path, last in schema_paths(root_name, relationships, 3):
            if not 2 <= len(path) <= 3 or last.reduce or len(key_parts(tables[path[-1]].key)) != 1:
                continue
            previous = root_name
            owned = True
            for node in path[:-1]:
                owned = owned and edges[(previous, node)].reduce
                previous = node
            if owned:
                candidates.append(path)
        # Prefer two owned event hops (interaction bridges), with deterministic
        # schema order for otherwise tied paths. No target/heldout rank lookup.
        candidates = sorted(set(candidates), key=lambda p: (-len(p), p))[:4]
        if not candidates:
            raise ChallengerUnavailable("No ordinary event-to-referenced-object path")
        peer = PeerHistoryFamily(max_depth=3, max_paths=2, max_neighbors=32, memory_limit="4GB")
        peer.root_name_, peer.entity_key_, peer.timestamp_ = root_name, entity, timestamp
        peer._membership_tables_ = peer._stable_membership_tables(tables, relationships, candidates)
        ranked = []
        for path in candidates:
            members = peer._members(requests[0], tables, relationships, path)
            members = members[["__example_id", "__neighbor_key_0"]].dropna().drop_duplicates()
            if members["__neighbor_key_0"].nunique() < 3:
                continue
            coverage = members["__example_id"].nunique()/max(1, len(train))
            ranked.append((-len(path), -coverage, path, members))
        selected = sorted(ranked, key=lambda r: r[:3])[:2]
        if not selected:
            raise ChallengerUnavailable("No usable training-visible association incidence")
        outputs, audits = [[], []], []
        support = [np.zeros(len(f), bool) for f in requests]
        for i, (_, coverage, path, members) in enumerate(selected):
            encoder = IncidenceBasis().fit(members, len(train))
            future = peer._members(requests[1], tables, relationships, path)
            for j, (rows, frame) in enumerate(((members, train), (future, query))):
                values, supported = encoder.transform(rows, len(frame))
                outputs[j].append(pd.DataFrame(values, columns=[f"association_state_p{i}_{k}" for k in range(values.shape[1])]))
                support[j] |= supported
            audits.append(dict(path=path, training_coverage=-coverage, encoder=encoder.audit))
        self.audit = dict(paths=audits, maximum_paths=2, maximum_members_per_query_per_hop=32,
            maximum_columns=32, strict_query_and_source_phase_clocks=True,
            graph_configurations_added=0, task_names_or_target_values_used_for_selection=False)
        return [pd.concat(parts, axis=1) for parts in outputs], support
