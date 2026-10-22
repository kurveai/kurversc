"""Opt-in, bounded preservation/encoding of relational payloads.

No benchmark-specific schema names. GraphReduce owns identity-preserving,
point-in-time extraction; a frozen training-only encoder owns the representation.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .learned_messages import LearnedMessageFamily


def key_parts(key):
    return (key,) if isinstance(key, str) else tuple(key or ())


def structural_columns(tables, relationships):
    columns = {
        name: set(key_parts(table.key)) | {table.date} | set(table.context_keys)
        for name, table in tables.items()
    }
    for edge in relationships:
        columns[edge.parent].update(key_parts(edge.parent_key))
        columns[edge.child].update(key_parts(edge.child_key))
    return columns


def payload_columns(table, structural):
    allowed = table.columns if table.columns is not None else table.source.columns
    return tuple(c for c in allowed if c not in structural)


def schema_paths(root, relationships, max_depth):
    frontier, found = [(root, ())], []
    for _ in range(max_depth):
        following = []
        for parent, path in frontier:
            for edge in sorted(relationships, key=lambda e: (e.parent, e.child)):
                if edge.parent == parent and edge.child not in (root, *path):
                    candidate = (*path, edge.child)
                    found.append((candidate, edge))
                    following.append((edge.child, candidate))
        frontier = following
    return found


def memberships(
    examples,
    tables,
    relationships,
    *,
    root,
    entity,
    timestamp,
    path,
    max_neighbors,
    seed,
    memory_limit,
):
    """Canonical root/terminal identities, including the zero-hop root path."""
    if not examples.index.is_unique or examples[timestamp].isna().any():
        raise ValueError("Examples need unique indexes and nonmissing timestamps")
    if len(key_parts(tables[root].key)) != 1:
        raise ValueError("Relational inputs currently require a single-column root key")
    if not path:
        table = tables[root]
        source = table.source.set_index(table.key)
        if not source.index.is_unique:
            raise ValueError("Root primary keys must be unique")
        keep = examples[entity].isin(source.index)
        if table.date:
            dates = examples[entity].map(source[table.date])
            keep &= pd.to_datetime(dates) < pd.to_datetime(examples[timestamp])
        return pd.DataFrame(
            {
                "__example_id": examples.index[keep],
                "__neighbor_key_0": examples.loc[keep, entity].to_numpy(),
            }
        )
    from graphreduce.learned_messages import extract_memberships

    return extract_memberships(
        examples,
        tables,
        relationships,
        root_name=root,
        entity_key=entity,
        timestamp=timestamp,
        path=path,
        max_neighbors=max_neighbors,
        seed=seed,
        memory_limit=memory_limit,
    )


def member_payload(table, rows, columns):
    if len(key_parts(table.key)) != 1:
        raise ValueError(
            "Payload encoders currently require single-column terminal keys"
        )
    unique = rows["__neighbor_key_0"].drop_duplicates().sort_values()
    frame = table.source.set_index(table.key)
    if not frame.index.is_unique:
        raise ValueError("Terminal primary keys must be unique")
    return frame.reindex(unique).reindex(columns=columns)


@dataclass
class PayloadEncoder:
    """Frozen lexical TF-IDF/SVD + numeric + bounded categorical indicators.

    This is a lexical representation, not a pretrained clinical language model.
    It never receives labels. Column budgets are separate by payload type so
    unique narratives cannot evict every structured attribute.
    """

    components: int = 64
    max_vocabulary: int = 12000
    max_text_columns: int = 8
    max_numeric_columns: int = 16
    max_category_indicators: int = 64
    max_category_values: int = 32
    max_characters: int = 2000
    max_fit_rows: int = 20000
    random_state: int = 0

    def _documents(self, frame):
        if not self.text_columns_:
            return pd.Series("", index=frame.index)
        blocks = [
            frame[c].fillna("").astype(str).str.slice(0, self.max_characters)
            for c in self.text_columns_
        ]
        result = blocks[0]
        for block in blocks[1:]:
            result = result.str.cat(block, sep=" ")
        return result

    def fit(self, frame):
        if min(self.components, self.max_vocabulary, self.max_fit_rows) < 1:
            raise ValueError("Encoder budgets must be positive")
        if frame.empty:
            raise ValueError("A nonempty historical payload is required")
        sample = frame
        if len(sample) > self.max_fit_rows:
            sample = sample.sample(
                n=self.max_fit_rows, random_state=self.random_state
            ).sort_index()
        self.numeric_columns_, text, self.categories_ = [], [], []
        for col in sample:
            values = sample[col].dropna()
            if values.empty:
                continue
            if pd.api.types.is_datetime64_any_dtype(values):
                continue
            if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(
                values
            ):
                if len(self.numeric_columns_) < self.max_numeric_columns:
                    self.numeric_columns_.append(col)
                continue
            counts = values.astype(str).value_counts()
            if len(counts) <= self.max_category_values:
                order = sorted(counts.index, key=lambda x: (-int(counts[x]), x))
                remaining = self.max_category_indicators - len(self.categories_)
                self.categories_.extend((col, value) for value in order[:remaining])
            # Every allowed nonnumeric field remains eligible for lexical input;
            # fit-time ordering gives narratives precedence over tiny enums.
            text.append((col, len(counts), float(values.astype(str).str.len().mean())))
        text.sort(key=lambda item: (-min(item[1], 100), -item[2], item[0]))
        self.text_columns_ = tuple(c for c, _, _ in text[: self.max_text_columns])
        self.columns_ = tuple(frame.columns)
        self.vectorizer_, self.reducer_, self.text_width_ = None, None, 0
        docs = self._documents(sample)
        if docs.str.strip().ne("").any():
            vectorizer = TfidfVectorizer(
                dtype=np.float32,
                max_features=self.max_vocabulary,
                min_df=1,
                ngram_range=(1, 2),
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b",
            )
            matrix = vectorizer.fit_transform(docs)
            if matrix.shape[1] >= 2 and matrix.shape[0] >= 2:
                width = min(self.components, matrix.shape[1] - 1, matrix.shape[0] - 1)
                reducer = TruncatedSVD(
                    n_components=width, n_iter=5, random_state=self.random_state
                )
                reducer.fit(matrix)
                self.vectorizer_, self.reducer_, self.text_width_ = (
                    vectorizer,
                    reducer,
                    width,
                )
        self.feature_names_ = (
            [f"numeric_{c}" for c in self.numeric_columns_]
            + [f"category_{i}" for i in range(len(self.categories_))]
            + [f"text_{i}" for i in range(self.text_width_)]
        )
        self.audit_ = dict(
            fit_rows=len(sample),
            numeric_columns=self.numeric_columns_,
            category_indicators=self.categories_,
            text_columns=self.text_columns_,
            text_components=self.text_width_,
            vocabulary_size=len(self.vectorizer_.vocabulary_)
            if self.vectorizer_
            else 0,
        )
        return self

    def transform(self, frame):
        frame = frame.reindex(columns=self.columns_)
        blocks = []
        if self.numeric_columns_:
            blocks.append(
                frame[self.numeric_columns_]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float32, na_value=np.nan)
            )
        if self.categories_:
            blocks.append(
                np.column_stack(
                    [
                        frame[c]
                        .astype("string")
                        .eq(value)
                        .fillna(False)
                        .to_numpy(dtype=np.float32)
                        for c, value in self.categories_
                    ]
                )
            )
        if self.text_width_:
            docs = self._documents(frame)
            encoded = []
            for start in range(0, len(frame), 2048):
                encoded.append(
                    self.reducer_.transform(
                        self.vectorizer_.transform(docs.iloc[start : start + 2048])
                    )
                )
            blocks.append(
                np.concatenate(encoded) if encoded else np.empty((0, self.text_width_))
            )
        matrix = np.column_stack(blocks) if blocks else np.empty((len(frame), 0))
        return pd.DataFrame(
            matrix, index=frame.index, columns=self.feature_names_
        ).replace([np.inf, -np.inf], np.nan)


@dataclass
class SemanticPayloadFamily:
    """Encode retained root/neighbor payloads before bounded pooling.

    Honor explicit Table.columns. To recover adapter-filtered descriptors,
    construct the opt-in problem with max_enrichment_columns=None. The encoder
    is frozen on the earliest supplied training frame for coordinate stability.
    """

    max_depth: int = 4
    max_paths: int = 12
    max_neighbors: int = 128
    components: int = 64
    random_state: int = 0
    memory_limit: str = "4GB"
    audit: list = field(default_factory=list, init=False)

    as_table = LearnedMessageFamily.as_table

    def _members(self, examples, tables, relationships, path):
        return memberships(
            examples,
            tables,
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
        training_examples,
        *,
        tables,
        relationships,
        root_name,
        entity_key,
        timestamp,
    ):
        if (
            training_examples.empty
            or min(self.max_paths, self.max_neighbors, self.max_depth) < 1
        ):
            raise ValueError("Training examples and positive budgets are required")
        self.root_name_, self.entity_key_, self.timestamp_ = (
            root_name,
            entity_key,
            timestamp,
        )
        self.root_key_ = tables[root_name].key
        self.fit_cutoff_ = pd.Timestamp(training_examples[timestamp].min())
        earliest = training_examples.loc[
            training_examples[timestamp].eq(self.fit_cutoff_), [entity_key, timestamp]
        ]
        structural = structural_columns(tables, relationships)
        candidates = schema_paths(root_name, relationships, self.max_depth)
        candidates.sort(key=lambda p: (p[1].reduce, len(p[0]), p[0]))
        self.encoders_, self.audit = {}, []
        for path in [(), *[p for p, _ in candidates]]:
            terminal = path[-1] if path else root_name
            table = tables[terminal]
            columns = payload_columns(table, structural[terminal])
            if not columns or len(key_parts(table.key)) != 1:
                continue
            rows = self._members(earliest, tables, relationships, path)
            if rows.empty:
                continue
            values = member_payload(table, rows, columns)
            encoder = PayloadEncoder(
                components=self.components, random_state=self.random_state
            ).fit(values)
            if not encoder.feature_names_:
                continue
            self.encoders_[path] = encoder
            self.audit.append(
                dict(path=path, fit_cutoff=str(self.fit_cutoff_), **encoder.audit_)
            )
            if len(self.encoders_) >= self.max_paths:
                break
        self.paths_ = tuple(self.encoders_)
        return self

    def transform(self, examples, *, tables, relationships):
        if not hasattr(self, "encoders_"):
            raise ValueError("Fit the payload family before inference")
        if (pd.to_datetime(examples[self.timestamp_]) < self.fit_cutoff_).any():
            raise ValueError(
                "Query cutoffs cannot precede the frozen encoder fit cutoff"
            )
        parts = [examples[[self.entity_key_, self.timestamp_]].copy()]
        for index, path in enumerate(self.paths_):
            table = tables[path[-1] if path else self.root_name_]
            encoder = self.encoders_[path]
            rows = self._members(examples, tables, relationships, path)
            values = member_payload(table, rows, encoder.columns_)
            encoded = encoder.transform(values)
            expanded = encoded.reindex(rows["__neighbor_key_0"]).reset_index(drop=True)
            expanded["__example_id"] = rows["__example_id"].to_numpy()
            grouped = expanded.groupby("__example_id", sort=True)
            if not path:
                pooled = grouped[encoder.feature_names_].first().reindex(examples.index)
            else:
                pooled = (
                    grouped[encoder.feature_names_]
                    .agg(["mean", "max"])
                    .reindex(examples.index)
                )
                pooled.columns = [f"{col}__{stat}" for col, stat in pooled.columns]
            pooled["neighbor_count"] = grouped.size().reindex(
                examples.index, fill_value=0
            )
            parts.append(pooled.add_prefix(f"payload_p{index}__"))
        return pd.concat(parts, axis=1)

    def fit_transform(self, training_examples, examples, **kwargs):
        self.fit(training_examples, **kwargs)
        return self.transform(
            examples, tables=kwargs["tables"], relationships=kwargs["relationships"]
        )
