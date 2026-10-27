"""Bounded, causal, entity-cross-fitted lexical target-affinity annotation.

Instead of blending a text-only probability into predictions, expose one
learned annotation to the existing tabular learner. Every annotation model
sees only targets completed strictly before the earliest feature query and
excludes every historical label of entities routed to it. Structural keys
route cross-fitting only; they are never lexical/model predictors.
"""

import hashlib

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from scripts.gap_text_readout import FieldTextReadout


def entity_folds(values, count=3):
    return np.asarray([int.from_bytes(hashlib.blake2b(str(v).encode("utf-8"), digest_size=8).digest(), "big") % count
        for v in values], dtype=np.int8)


class CausalParentTextAnnotation:
    def __init__(self, *, maximum_per_field=4096, historical_frames=4):
        self.maximum_per_field = maximum_per_field
        self.historical_frames = historical_frames

    def fit(self, history, *, target, columns, first_query, horizon):
        keys = ["__event_entity", "__event_cutoff"]
        if not columns or len(columns) > 4 or set(keys + [target]) & set(columns):
            raise ValueError("One to four payload text columns required; structural keys/target forbidden")
        if history[keys].isna().any().any() or history.duplicated(keys).any():
            raise ValueError("Unique complete structural training keys required")
        first_query = pd.Timestamp(first_query)
        times = pd.to_datetime(history[keys[1]])
        eligible = times + pd.Timedelta(horizon) < first_query
        cutoffs = sorted(times.loc[eligible].unique())[-self.historical_frames:]
        if len(cutoffs) != self.historical_frames:
            raise ValueError("Complete historical text support required")
        selected = history.loc[eligible & times.isin(cutoffs)].copy()
        order = np.lexsort((selected[keys[0]].astype(str).to_numpy(), pd.to_datetime(selected[keys[1]]).astype("int64").to_numpy()))
        selected = selected.iloc[order]
        groups = entity_folds(selected[keys[0]])
        self.columns = tuple(columns)
        self.heads = []
        records = []
        for fold in range(3):
            train = selected.loc[groups != fold]
            vocabulary = train.loc[~train[keys[0]].duplicated(keep="last")]
            y = train[target].to_numpy(int)
            if set(np.unique(y)) != {0, 1} or len(vocabulary) < 20:
                raise ValueError("Reusable binary lexical annotation support required")
            vectorizers, matrices = {}, []
            nonzeros = 0
            for column in self.columns:
                vectorizer = TfidfVectorizer(max_features=self.maximum_per_field, ngram_range=(1, 2),
                    min_df=2, max_df=.98, sublinear_tf=True, strip_accents="unicode", dtype=np.float32)
                try:
                    vectorizer.fit(FieldTextReadout.documents(vocabulary, column))
                except ValueError as error:
                    if "empty vocabulary" not in str(error) and "After pruning" not in str(error):
                        raise
                    continue
                x = vectorizer.transform(FieldTextReadout.documents(train, column))
                nonzeros += x.nnz
                if nonzeros > 32000000:
                    raise MemoryError("Cross-fit lexical annotation sparse ceiling exceeded")
                vectorizers[column] = vectorizer
                matrices.append(x)
            if not matrices:
                raise ValueError("No reusable historical vocabulary")
            x = sparse.hstack(matrices, format="csr")
            model = LogisticRegression(C=1., solver="liblinear", max_iter=1000, random_state=0)
            model.fit(x, y)
            self.heads.append((vectorizers, model))
            records.append(dict(excluded_entity_fold=fold, training_rows=len(train), vocabulary_entities=len(vocabulary),
                fitted_nonzeros=nonzeros, vocabulary_sizes={c: len(v.vocabulary_) for c, v in vectorizers.items()}))
        self.first_query = first_query
        self.audit = dict(first_query=str(first_query), historical_cutoffs=[str(pd.Timestamp(t)) for t in cutoffs],
            latest_label_complete_at=str((pd.to_datetime(selected[keys[1]]) + pd.Timedelta(horizon)).max()),
            maximum_per_field=self.maximum_per_field, output_features=1, text_columns=self.columns,
            entity_crossfit_groups=3, historical_frames=self.historical_frames, heads=records,
            scope="Strictly older mature targets; all own entity labels excluded; keys only route cross-fitting")
        return self

    def transform(self, frame):
        if frame["__event_entity"].isna().any() or (pd.to_datetime(frame["__event_cutoff"]) < self.first_query).any():
            raise ValueError("Queries must follow the frozen historical annotation plan")
        groups = entity_folds(frame["__event_entity"])
        prediction = np.empty(len(frame), dtype=np.float64)
        for fold, (vectorizers, model) in enumerate(self.heads):
            positions = np.flatnonzero(groups == fold)
            if not len(positions):
                continue
            query = frame.iloc[positions]
            matrices = [v.transform(FieldTextReadout.documents(query, c)) for c, v in vectorizers.items()]
            prediction[positions] = model.predict_proba(sparse.hstack(matrices, format="csr"))[:, 1]
        return pd.DataFrame(dict(lexical_target_affinity=prediction))
