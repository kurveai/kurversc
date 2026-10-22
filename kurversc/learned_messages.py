"""Optional fitted relational messages with forward-only supervised training.

GraphReduce owns bounded path extraction and pooling. This module owns label
availability, row scorers, and replay. It deliberately does not expand the
ordinary unsupervised graph-search lattice automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import re

import numpy as np
import pandas as pd

from .specs import Table, Relationship


def encode_payload(rows: pd.DataFrame, *, buckets: int = 8) -> pd.DataFrame:
    """Stateless numeric/text encoding shared by learned and unlearned arms.

    Text and categoricals use per-column signed token hashing, with no fitted
    vocabulary or future corpus statistics. Structural identifiers never enter
    this function. The same encoded rows feed the scorer and the raw control.
    """
    output = {}
    for column in rows:
        if column == "__example_id":
            continue
        series = rows[column]
        if pd.api.types.is_numeric_dtype(series):
            output[column] = pd.to_numeric(series, errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            # Nonstructural datetime attributes are deliberately omitted: their
            # availability cannot be inferred from the cell value.
            continue
        else:
            codes, unique = pd.factorize(series.fillna("").astype(str), sort=True)
            vectors = np.zeros((len(unique), buckets), dtype=np.float32)
            for index, value in enumerate(unique):
                tokens = re.findall(r"\w+", value.lower()[:4000])
                for token in tokens:
                    digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                    number = int.from_bytes(digest, "little")
                    vectors[index, number % buckets] += 1 if number & 256 else -1
                if tokens:
                    vectors[index] /= np.sqrt(len(tokens))
            for bucket in range(buckets):
                output[f"{column}__hash{bucket}"] = vectors[codes, bucket]
    if not output:
        output["__constant"] = np.ones(len(rows))
    return (
        pd.DataFrame(output, index=rows.index)
        .replace([np.inf, -np.inf], np.nan)
        .astype("float32")
    )


@dataclass
class LearnedMessageFamily:
    """Fitted, bounded, schema-derived scalar-message feature family.

    Training labels must include an explicit availability timestamp, normally
    prediction timestamp + outcome horizon + reporting delay. For every query
    cutoff, only labels available *strictly before* that cutoff can train its
    scorer. The earliest cutoff may have no model; its message is missing.

    Repeated entities and shared neighbors are allowed across past/future dates:
    they are legitimate historical signal, while overlapping label windows are
    excluded. Input rows are assumed immutable as of their declared table date;
    versioned mutable attributes require an upstream as-of view.

    ``fit_transform`` returns exact-cutoff snapshots and stores fitted scorers;
    ``transform`` scores new unlabeled examples without retraining. Both also
    return ordinary pooling of the EXACT numeric inputs supplied to the scorer,
    enabling a controlled ablation. Pickle/joblib persists the fitted family.
    """

    max_depth: int = 2
    max_paths: int = 12
    max_neighbors: int = 128
    max_columns: int = 16
    iterations: int = 160
    thread_count: int = 4
    random_state: int = 0
    memory_limit: str = "4GB"
    neighbor_context_depth: int = 0
    neighbor_context_max_columns: int = 4
    neighbor_context_max_relations: int = 12
    audit: list[dict] = field(default_factory=list, init=False)
    models_: dict = field(default_factory=dict, init=False, repr=False)

    def _paths(self, root, relationships):
        paths, frontier = [], [(root, ())]
        for _ in range(self.max_depth):
            next_frontier = []
            for parent, path in frontier:
                for edge in sorted(
                    relationships, key=lambda edge: (edge.parent, edge.child)
                ):
                    if edge.parent == parent and edge.child not in (root, *path):
                        candidate = (*path, edge.child)
                        paths.append(candidate)
                        next_frontier.append((edge.child, candidate))
            frontier = next_frontier
        # Prefer terminal paths; association -> dimension paths retain payload
        # that the association-only prefix does not contain.
        terminals = [
            p
            for p in paths
            if not any(len(q) > len(p) and q[: len(p)] == p for q in paths)
        ]
        return tuple(sorted(terminals, key=lambda p: (len(p), p))[: self.max_paths])

    def _validate_examples(self, examples):
        if not examples.index.is_unique or examples[self.timestamp_].isna().any():
            raise ValueError("Examples need unique indexes and nonmissing timestamps")
        if examples.duplicated([self.entity_key_, self.timestamp_]).any():
            raise ValueError("Only one example per entity and cutoff is supported")

    def _message_names(self):
        return ("score",)

    def _scorer_columns(self, encoded):
        return tuple(c for c in encoded if encoded[c].nunique(dropna=False) > 1)

    def _fit_scorer(self, x, y, ids):
        from catboost import CatBoostClassifier, CatBoostRegressor

        counts = pd.Series(ids).value_counts()
        weights = 1.0 / pd.Series(ids).map(counts).to_numpy()
        cls = (
            CatBoostClassifier if self.task_ == "classification" else CatBoostRegressor
        )
        model = cls(
            iterations=self.iterations,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss" if self.task_ == "classification" else "MAE",
            random_seed=self.random_state,
            thread_count=self.thread_count,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x, y, sample_weight=weights)
        return model

    def _predict_scorer(self, model, x):
        values = (
            model.predict_proba(x)[:, 1]
            if self.task_ == "classification"
            else model.predict(x)
        )
        return np.asarray(values).reshape(-1, 1)

    def _finish_messages(self, message, examples, versions):
        return message

    def _fit_encoded_path(
        self, encoded, ids, training, examples, *, path, target, available_at
    ):
        """Shared temporal fitter, also usable with a cached extracted frame."""
        row_y = training.loc[ids, target].to_numpy()
        availability = pd.to_datetime(training.loc[ids, available_at]).to_numpy()
        versions = []
        cutoffs = sorted(set(pd.to_datetime(examples[self.timestamp_])))
        cutoffs.append(
            pd.Timestamp(training[available_at].max()) + pd.Timedelta(nanoseconds=1)
        )
        seen = set()
        for cutoff in sorted(set(cutoffs)):
            # np.datetime64(Timestamp) silently truncates nanoseconds to us.
            # Preserve the +1 ns final-training sentinel and exact boundaries.
            keep = availability < cutoff.to_datetime64()
            eligible = np.flatnonzero(keep)
            signature = tuple(np.unique(ids[keep]))
            if not len(eligible) or signature in seen:
                continue
            seen.add(signature)
            x, y = encoded.iloc[eligible], row_y[keep]
            usable = self._scorer_columns(x)
            model = None
            counts = pd.Series(ids[keep]).value_counts()
            weights = 1.0 / pd.Series(ids[keep]).map(counts).to_numpy()
            constant = float(np.average(y, weights=weights))
            if usable and len(np.unique(y)) > 1:
                model = self._fit_scorer(x[list(usable)], y, ids[keep])
            latest = pd.Timestamp(availability[keep].max())
            versions.append((latest, model, usable, constant))
            self.audit.append(
                dict(
                    path=list(path),
                    cutoff=str(cutoff),
                    latest_label_available=str(latest),
                    training_roots=len(signature),
                    training_neighbors=len(eligible),
                    columns=len(usable),
                    constant_model=model is None,
                    **getattr(model, "fit_audit_", {}),
                )
            )
        self.models_[path] = versions

    def _transform_encoded_path(self, rows, encoded, examples, *, path):
        from graphreduce.learned_messages import pool_messages

        names = self._message_names()
        scores = np.full((len(rows), len(names)), np.nan)
        row_times = examples.loc[rows["__example_id"], self.timestamp_].to_numpy()
        versions = self.models_[path]
        for cutoff in sorted(set(pd.to_datetime(examples[self.timestamp_]))):
            candidates = [v for v in versions if v[0] < cutoff]
            selected = np.flatnonzero(row_times == cutoff.to_datetime64())
            if not candidates or not len(selected):
                continue
            _, model, columns, constant = max(candidates, key=lambda v: v[0])
            if model is None:
                scores[selected] = constant
            else:
                x = encoded.iloc[selected].reindex(columns=columns)
                scores[selected] = self._predict_scorer(model, x)
        valid = np.isfinite(scores).all(axis=1)
        message = pool_messages(
            rows.loc[valid], pd.DataFrame(scores[valid], columns=names), examples.index
        )
        return self._finish_messages(message, examples, versions)

    def fit_transform(
        self,
        training,
        examples,
        *,
        tables,
        relationships,
        root_name,
        entity_key,
        timestamp,
        target,
        available_at,
        task="regression",
    ):
        """Fit from training only; examples may be entirely unlabeled.

        Training and query indexes need not overlap. Root/time is the identity.
        All fit rows are extracted at their OWN historical feature cutoff.
        """
        if task not in {"regression", "classification"}:
            raise ValueError("Only regression and binary classification are supported")
        if self.neighbor_context_depth < 0:
            raise ValueError("neighbor_context_depth must be nonnegative")
        if (
            min(
                self.max_depth,
                self.max_paths,
                self.max_neighbors,
                self.max_columns,
                self.iterations,
                self.thread_count,
            )
            < 1
        ):
            raise ValueError("All learned-message budgets must be positive")
        self.entity_key_, self.timestamp_, self.task_ = entity_key, timestamp, task
        self.root_name_ = root_name
        self.root_key_ = tables[root_name].key
        self.paths_ = self._paths(root_name, relationships)
        self._validate_examples(training)
        self._validate_examples(examples)
        training = (
            training.sort_values([timestamp, entity_key], kind="stable")
            .reset_index(drop=True)
            .copy()
        )
        if training[target].isna().any() or training[available_at].isna().any():
            raise ValueError(
                "Training targets and label availability must be nonmissing"
            )
        if (
            pd.to_datetime(training[available_at]) < pd.to_datetime(training[timestamp])
        ).any():
            raise ValueError("Label availability cannot precede its prediction cutoff")
        if task == "classification" and not set(training[target].unique()).issubset(
            {0, 1}
        ):
            raise ValueError("Binary targets must be encoded as 0/1")
        self.models_, self.audit, self.payload_columns_ = {}, [], {}
        self.context_plans_ = {}
        from graphreduce.learned_messages import extract_path

        for path in self.paths_:
            self.context_plans_[path] = {}
            rows = extract_path(
                training,
                tables,
                relationships,
                root_name=root_name,
                entity_key=entity_key,
                timestamp=timestamp,
                path=path,
                max_neighbors=self.max_neighbors,
                max_columns=self.max_columns,
                seed=self.random_state,
                memory_limit=self.memory_limit,
                context_depth=self.neighbor_context_depth,
                context_max_columns=self.neighbor_context_max_columns,
                context_max_relations=self.neighbor_context_max_relations,
                context_plan_cache=self.context_plans_[path],
            )
            encoded = encode_payload(rows)
            self.payload_columns_[path] = tuple(
                c for c in rows if c in tables[path[-1]].source.columns
            )
            ids = rows["__example_id"].to_numpy()
            self._fit_encoded_path(
                encoded,
                ids,
                training,
                examples,
                path=path,
                target=target,
                available_at=available_at,
            )
        return self.transform(examples, tables=tables, relationships=relationships)

    def transform(self, examples, *, tables, relationships):
        """Generate snapshots from frozen scorers; never consume query targets."""
        from graphreduce.learned_messages import extract_path, pool_messages

        if not hasattr(self, "paths_"):
            raise ValueError("Fit the message family before transforming")
        self._validate_examples(examples)
        raw, learned = [], []
        for index, path in enumerate(self.paths_):
            path_tables = dict(tables)
            manifest = getattr(self, "payload_columns_", {}).get(path)
            if manifest is not None:
                missing = set(manifest) - set(tables[path[-1]].source.columns)
                if missing:
                    raise ValueError(
                        f"Prediction payload is missing columns: {sorted(missing)}"
                    )
                path_tables[path[-1]] = replace(tables[path[-1]], columns=manifest)
            rows = extract_path(
                examples,
                path_tables,
                relationships,
                root_name=self.root_name_,
                entity_key=self.entity_key_,
                timestamp=self.timestamp_,
                path=path,
                max_neighbors=self.max_neighbors,
                max_columns=self.max_columns,
                seed=self.random_state,
                memory_limit=self.memory_limit,
                context_depth=self.neighbor_context_depth,
                context_max_columns=self.neighbor_context_max_columns,
                context_max_relations=self.neighbor_context_max_relations,
                context_plan_cache=self.context_plans_[path]
                if self.neighbor_context_depth
                else None,
            )
            encoded = encode_payload(rows)
            pooled = pool_messages(rows, encoded, examples.index)
            raw.append(pooled.add_prefix(f"raw_p{index}__"))
            message = self._transform_encoded_path(rows, encoded, examples, path=path)
            learned.append(message.add_prefix(f"learned_p{index}__"))
        identifiers = examples[[self.entity_key_, self.timestamp_]].copy()
        return (
            pd.concat([identifiers, *raw], axis=1),
            pd.concat([identifiers, *learned], axis=1),
        )

    def as_table(self, snapshots, *, name="learned_messages"):
        """Expose exact-cutoff snapshots to ordinary KurveRSC fit/predict.

        At prediction time call transform on the new queries, then supply this
        table with the same name. Save the family alongside the FitResult.
        """
        if snapshots.duplicated([self.entity_key_, self.timestamp_]).any():
            raise ValueError("Snapshot entity/cutoff pairs must be unique")
        table = Table(
            snapshots,
            key=self.entity_key_,
            date=self.timestamp_,
            name=name,
            prefix=name,
            sample_in_search=False,
            cutoff_boundary="exact",
            auto_base_predicate_max=0,
        )
        edge = Relationship(
            self.root_name_,
            name,
            getattr(self, "root_key_", self.entity_key_),
            self.entity_key_,
        )
        return table, edge
