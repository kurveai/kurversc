"""Opt-in, label-free parent encodings with separate, hard width budgets.

The input is the parent's own point-in-time payload, never a propagated frame.
Structured attributes retain their type; text uses field-aware lexical features,
not a pretrained semantic model. Fit once on training payloads and replay the
frozen encoder. This experimental module does not change production defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from .specs import Table


def parent_payload(
    examples: pd.DataFrame,
    table: Table,
    *,
    entity_key: str,
    timestamp: str,
    excluded: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Align a unique-key parent to examples, excluding unavailable source rows.

    Callers must supply a cutoff-censored source and exclude all structural FK
    columns. The date is retained only for typed, cutoff-relative encoding.
    Example labels are never inspected or copied into the returned payload.
    """
    if not isinstance(table.source, pd.DataFrame) or not isinstance(table.key, str):
        raise ValueError("Parent payload requires a DataFrame and a single key")
    if not examples.index.is_unique or examples[timestamp].isna().any():
        raise ValueError("Example indexes must be unique and timestamps complete")
    source = table.source.set_index(table.key, drop=False)
    if not source.index.is_unique:
        raise ValueError("Parent source key must be unique")
    columns = list(table.columns) if table.columns is not None else list(source)
    forbidden = {table.key, *excluded, *table.context_keys}
    columns = [c for c in columns if c not in forbidden]
    if table.date and table.date not in columns and table.date not in forbidden:
        columns.append(table.date)
    aligned = source.reindex(examples[entity_key])
    aligned.index = examples.index
    frame = aligned.loc[:, columns].copy()
    frame.index = examples.index
    if table.date:
        dates = pd.to_datetime(aligned[table.date], utc=True)
        cutoffs = pd.to_datetime(examples[timestamp], utc=True)
        frame.loc[~dates.lt(cutoffs), :] = pd.NA
    return frame


def _string_value(value):
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    if pd.isna(value):
        return None
    return str(value)


def _field_tokens(document):
    """Stable field identity prevents equal words in different fields merging."""
    for field_index, text in enumerate(document):
        words = re.findall(r"\b\w+\b", text.lower())
        for word in words:
            yield f"f{field_index}:{word}"
        for left, right in zip(words, words[1:]):
            yield f"f{field_index}:{left}_{right}"


@dataclass
class BoundedParentEncoder:
    """At most 64 structured + 64 lexical columns, regardless of cardinality.

    There is no target encoding, column interaction enumeration, or one-hot
    expansion. Category dictionaries, text work, sample size, and output batch
    size are separately bounded. All selection decisions are training-only.
    """

    structured_features: int = 64
    text_components: int = 64
    max_input_columns: int = 128
    max_text_columns: int = 8
    max_fit_rows: int = 20_000
    max_categories: int = 2048
    max_category_characters: int = 256
    max_text_characters: int = 4096
    max_document_characters: int = 16_384
    hash_features: int = 16_384
    batch_rows: int = 512
    random_state: int = 0

    def _strings(self, series):
        return series.map(_string_value)

    def _documents(self, frame):
        allowance = min(
            self.max_text_characters,
            self.max_document_characters // max(1, len(self.text_columns_)),
        )
        fields = [
            self._strings(frame[c]).fillna("").str.slice(0, allowance).tolist()
            for c in self.text_columns_
        ]
        return list(zip(*fields))

    def fit(self, frame: pd.DataFrame) -> BoundedParentEncoder:
        if frame.empty or not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError("Fit requires nonempty rows and unique indexes/columns")
        if not all(isinstance(c, str) for c in frame.columns):
            raise ValueError("Parent column names must be strings")
        positive = (
            self.max_input_columns,
            self.max_text_columns,
            self.max_fit_rows,
            self.max_categories,
            self.max_category_characters,
            self.max_text_characters,
            self.max_document_characters,
            self.hash_features,
            self.batch_rows,
        )
        if min(positive) < 1 or min(self.structured_features, self.text_components) < 0:
            raise ValueError("Feature widths must be nonnegative and budgets positive")
        # Stable under row/column reordering of the same indexed training data.
        sample = frame.sort_index()
        if len(sample) > self.max_fit_rows:
            sample = sample.sample(n=self.max_fit_rows, random_state=self.random_state)
            sample = sample.sort_index()
        coverage = sample.notna().mean()
        ordered = sorted(sample, key=lambda c: (-float(coverage[c]), c))
        self.columns_ = tuple(ordered[: self.max_input_columns])
        primary, secondary, texts, audit = [], [], [], []
        self.category_values_ = {}
        for column in self.columns_:
            values = sample[column]
            record = dict(column=column, coverage=float(coverage[column]), outputs=[])
            if values.notna().sum() == 0:
                record.update(kind="empty", reason="no observed training values")
                audit.append(record)
                continue
            if pd.api.types.is_datetime64_any_dtype(values.dtype):
                # A constant date can still have useful age across example times.
                record["kind"] = "datetime"
                primary.append((column, "age_days"))
                secondary.extend([(column, "month"), (column, "dayofweek")])
            elif pd.api.types.is_numeric_dtype(
                values.dtype
            ) and not pd.api.types.is_bool_dtype(values):
                record["kind"] = "numeric"
                if values.nunique(dropna=False) > 1:
                    primary.append((column, "numeric"))
            else:
                strings = self._strings(values)
                counts = strings.dropna().value_counts()
                length = float(strings.dropna().str.len().mean())
                narrative = length > 80 and len(counts) > 32
                record.update(kind="text" if narrative else "categorical")
                if not narrative and (len(counts) > 1 or strings.isna().any()):
                    categories = sorted(
                        counts.index, key=lambda v: (-int(counts[v]), v)
                    )
                    self.category_values_[column] = frozenset(
                        categories[: self.max_categories]
                    )
                    record["retained_categories"] = len(self.category_values_[column])
                    record["observed_categories"] = len(counts)
                    primary.append((column, "category"))
                if len(counts) > 1:
                    texts.append((column, float(coverage[column]), length, len(counts)))
            audit.append(record)
        self.plan_ = (primary + secondary)[: self.structured_features]
        self.structured_columns_ = tuple(
            f"parent__s{i}" for i in range(len(self.plan_))
        )
        self.categorical_columns_ = tuple(
            name
            for name, (_, kind) in zip(self.structured_columns_, self.plan_)
            if kind == "category"
        )
        # Prefer informative narratives; every field still has an independent tag.
        texts.sort(key=lambda x: (-min(x[3], 100), -x[2], -x[1], x[0]))
        self.text_columns_ = tuple(x[0] for x in texts[: self.max_text_columns])
        self.vectorizer_, self.tfidf_, self.reducer_ = None, None, None
        self.text_width_ = 0
        if self.text_components and self.text_columns_ and len(sample) > 1:
            vectorizer = HashingVectorizer(
                analyzer=_field_tokens,
                n_features=self.hash_features,
                alternate_sign=False,
                norm=None,
                dtype=np.float32,
            )
            counts = vectorizer.transform(self._documents(sample))
            if counts.nnz:
                tfidf = TfidfTransformer(sublinear_tf=True)
                matrix = tfidf.fit_transform(counts)
                width = min(
                    self.text_components, matrix.shape[0] - 1, matrix.shape[1] - 1
                )
                reducer = TruncatedSVD(
                    n_components=width, n_iter=5, random_state=self.random_state
                )
                reducer.fit(matrix)
                self.vectorizer_, self.tfidf_, self.reducer_ = (
                    vectorizer,
                    tfidf,
                    reducer,
                )
                self.text_width_ = width
        self.feature_names_ = (
            *self.structured_columns_,
            *(f"parent__text{i}" for i in range(self.text_width_)),
        )
        by_column = {row["column"]: row for row in audit}
        for name, (column, kind) in zip(self.structured_columns_, self.plan_):
            by_column[column]["outputs"].append(dict(feature=name, kind=kind))
        for column in self.text_columns_:
            by_column[column]["lexical_encoded"] = self.text_width_ > 0
        self.audit_ = dict(
            fit_rows=len(sample),
            supplied_columns=len(frame.columns),
            selected_input_columns=list(self.columns_),
            omitted_input_columns=[c for c in sorted(frame) if c not in self.columns_],
            structured_columns=len(self.structured_columns_),
            text_components=self.text_width_,
            total_columns=len(self.feature_names_),
            max_total_columns=self.structured_features + self.text_components,
            text_columns=list(self.text_columns_),
            categorical_columns=list(self.categorical_columns_),
            columns=audit,
            representation="typed parent payload + field-tagged hashed TF-IDF/SVD; no pretrained semantics",
        )
        return self

    def transform(self, frame: pd.DataFrame, *, timestamps: pd.Series) -> pd.DataFrame:
        if not hasattr(self, "plan_"):
            raise ValueError("Fit the parent encoder before transforming")
        if not frame.index.equals(timestamps.index) or timestamps.isna().any():
            raise ValueError(
                "Prediction timestamps must be complete and exactly aligned"
            )
        if not set(self.columns_).issubset(frame.columns):
            raise ValueError("Parent input schema is missing fitted columns")
        out = {}
        for name, (column, kind) in zip(self.structured_columns_, self.plan_):
            values = frame[column]
            if kind == "category":
                text = self._strings(values)
                known = self.category_values_[column]
                out[name] = text.map(
                    lambda v: (
                        "missing:"
                        if v is None
                        else "value:" + v[: self.max_category_characters]
                        if v in known
                        else "other:"
                    )
                )
            elif kind == "numeric":
                out[name] = (
                    pd.to_numeric(values, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .astype("float32")
                )
            else:
                date = pd.to_datetime(values, errors="coerce", utc=True)
                if kind == "age_days":
                    value = (
                        pd.to_datetime(timestamps, utc=True) - date
                    ).dt.total_seconds() / 86400
                else:
                    value = getattr(date.dt, kind)
                out[name] = value.astype("float32")
        result = pd.DataFrame(out, index=frame.index)
        if self.text_width_:
            batches = []
            for start in range(0, len(frame), self.batch_rows):
                block = frame.iloc[start : start + self.batch_rows]
                counts = self.vectorizer_.transform(self._documents(block))
                batches.append(
                    self.reducer_.transform(self.tfidf_.transform(counts)).astype(
                        "float32"
                    )
                )
            values = (
                np.concatenate(batches)
                if batches
                else np.empty((0, self.text_width_), dtype="float32")
            )
            text = pd.DataFrame(
                values,
                index=frame.index,
                columns=[f"parent__text{i}" for i in range(self.text_width_)],
            )
            result = pd.concat([result, text], axis=1)
        return result.loc[:, self.feature_names_]
