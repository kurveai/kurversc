"""Experimental, schema-generic joint readout of bounded relation bags.

No benchmark names, query labels, or automatic production/search changes.
GraphReduce owns point-in-time extraction; callers supply an eligible training
prefix. Root attributes may be represented as a one-row bag, just like any
other relation. Each relation is compressed before a joint parent-level loss.
"""

import time

import numpy as np
import pandas as pd


class _Normalizer:
    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        sample = x[np.linspace(0, len(x) - 1, min(10000, len(x)), dtype=int)]
        finite = np.isfinite(sample)
        n = finite.sum(axis=0)
        safe = np.where(finite, sample, 0).astype(np.float64)
        self.center = (safe.sum(axis=0) / np.maximum(n, 1)).astype(np.float32)
        variance = np.where(finite, (safe - self.center) ** 2, 0).sum(
            axis=0
        ) / np.maximum(n, 1)
        self.scale = np.sqrt(np.maximum(variance, 0)).astype(np.float32)
        self.scale[self.scale < 1e-6] = 1
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        missing = ~np.isfinite(x)
        normalized = np.clip(
            (np.where(missing, self.center, x) - self.center) / self.scale, -8, 8
        )
        return np.concatenate([normalized, missing.astype(np.float32)], axis=1)


def _indexed_bag(frame, roots):
    if "__example_id" not in frame or not frame.columns.is_unique:
        raise ValueError("Every bag needs example metadata and unique payload columns")
    index = roots.get_indexer(frame.__example_id)
    keep = index >= 0
    index = index[keep]
    payload = frame.loc[keep].drop(columns="__example_id")
    if not len(payload.columns):
        raise ValueError("Bags need at least one numeric payload feature")
    if not all(pd.api.types.is_numeric_dtype(payload[c]) for c in payload):
        raise ValueError("Encode all bag payloads numerically before fitting")
    order = np.argsort(index, kind="stable")
    counts = np.bincount(index, minlength=len(roots))
    offsets = np.r_[0, np.cumsum(counts)]
    return payload.to_numpy(dtype=np.float32)[order], counts, offsets


class JointNeighborhoodEncoder:
    """Per-relation phi/mean followed by one joint root-level readout.

    Input is a mapping of arbitrary relation names to numeric DataFrames with
    an ``__example_id`` metadata column. Targets are a Series indexed by root
    example, not entity ID; repeated entities at different cutoffs are distinct.
    Every training root has equal loss weight regardless of relation fanout.
    Missing relations carry zero latent messages plus count/availability.
    """

    def __init__(
        self,
        *,
        width=8,
        hidden=32,
        epochs=30,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.0001,
        task="regression",
        random_state=0,
    ):
        if task not in {"regression", "classification"}:
            raise ValueError("Only regression and binary classification are supported")
        if (
            min(width, hidden, epochs, batch_size) < 1
            or learning_rate <= 0
            or weight_decay < 0
        ):
            raise ValueError("Neural budgets must be positive")
        self.width, self.hidden, self.epochs, self.batch_size = (
            width,
            hidden,
            epochs,
            batch_size,
        )
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.task, self.random_state = task, random_state

    def _prepared(self, bags, roots, *, fit=False):
        import torch

        if not roots.is_unique or roots.hasnans:
            raise ValueError("Root example indexes must be unique and nonmissing")
        if set(bags) != set(self.names_):
            raise ValueError("Relation schema must match the fitted schema")
        prepared = []
        for name in self.names_:
            columns = tuple(c for c in bags[name] if c != "__example_id")
            if not fit and columns != self.columns_[name]:
                raise ValueError("Payload feature schema/order changed")
            x, counts, offsets = _indexed_bag(bags[name], roots)
            if fit:
                if not len(x):
                    raise ValueError(
                        "Each fitted relation needs at least one training row"
                    )
                self.columns_[name] = columns
                self.normalizers_[name] = _Normalizer().fit(x)
            normalized = self.normalizers_[name].transform(x)
            prepared.append((torch.from_numpy(normalized), counts, offsets))
        return prepared

    def _latents(self, prepared, parents):
        import torch

        pooled = []
        for encoder, (x, counts, offsets) in zip(self.encoders_, prepared, strict=True):
            sizes = counts[parents]
            row_ids = np.concatenate(
                [np.arange(offsets[p], offsets[p + 1]) for p in parents]
            )
            groups = torch.from_numpy(np.repeat(np.arange(len(parents)), sizes))
            z = encoder(x[row_ids])
            total = torch.zeros((len(parents), self.width), dtype=z.dtype).index_add(
                0, groups, z
            )
            mean = total / torch.tensor(np.maximum(sizes, 1), dtype=z.dtype)[:, None]
            metadata = torch.tensor(
                np.column_stack([np.log1p(sizes), sizes > 0]), dtype=z.dtype
            )
            pooled.extend([mean, metadata])
        return torch.cat(pooled, dim=1)

    def fit(self, bags, y):
        import torch
        from torch import nn

        started = time.monotonic()
        if not isinstance(y, pd.Series) or not len(y) or not np.isfinite(y).all():
            raise ValueError("Training targets must be a finite indexed Series")
        if self.task == "classification" and not set(y.unique()).issubset({0, 1}):
            raise ValueError("Binary labels must be 0/1")
        self.names_ = tuple(sorted(bags))
        if not self.names_:
            raise ValueError("At least one relation is required")
        self.columns_, self.normalizers_ = {}, {}
        prepared = self._prepared(bags, y.index, fit=True)
        self.target_center_ = float(y.median()) if self.task == "regression" else 0.0
        self.target_scale_ = (
            max(float(np.std(y)), 1e-3) if self.task == "regression" else 1.0
        )
        ty = torch.tensor(
            (y.to_numpy() - self.target_center_) / self.target_scale_,
            dtype=torch.float32,
        )
        rng, trace = np.random.default_rng(self.random_state), []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            self.encoders_ = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(x.shape[1], self.hidden),
                        nn.ReLU(),
                        nn.Linear(self.hidden, self.width),
                        nn.Tanh(),
                    )
                    for x, _, _ in prepared
                ]
            )
            self.head_ = nn.Sequential(
                nn.Linear(len(prepared) * (self.width + 2), 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            optimizer = torch.optim.AdamW(
                [*self.encoders_.parameters(), *self.head_.parameters()],
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            loss_fn = (
                nn.BCEWithLogitsLoss() if self.task == "classification" else nn.L1Loss()
            )
            for epoch in range(self.epochs):
                order, total = rng.permutation(len(y)), 0.0
                for start in range(0, len(y), self.batch_size):
                    parents = order[start : start + self.batch_size]
                    prediction = self.head_(self._latents(prepared, parents)).flatten()
                    loss = loss_fn(prediction, ty[parents])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite joint-neighborhood loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [*self.encoders_.parameters(), *self.head_.parameters()], 5
                    )
                    optimizer.step()
                    total += float(loss.detach()) * len(parents)
                if epoch == 0 or (epoch + 1) % 5 == 0:
                    trace.append(dict(epoch=epoch + 1, loss=total / len(y)))
        self.encoders_.eval()
        self.head_.eval()
        self.fit_audit_ = dict(
            roots=len(y),
            relations=len(bags),
            epochs=self.epochs,
            neighbors={
                name: int(prepared[i][1].sum()) for i, name in enumerate(self.names_)
            },
            parameters=sum(p.numel() for p in self.encoders_.parameters())
            + sum(p.numel() for p in self.head_.parameters()),
            loss_trace=trace,
            fit_seconds=time.monotonic() - started,
        )
        return self

    def transform(self, bags, roots):
        import torch

        roots = pd.Index(roots)
        prepared = self._prepared(bags, roots)
        blocks = []
        with torch.no_grad():
            for start in range(0, len(roots), self.batch_size):
                blocks.append(
                    self._latents(
                        prepared,
                        np.arange(start, min(start + self.batch_size, len(roots))),
                    ).numpy()
                )
        return (
            np.concatenate(blocks)
            if blocks
            else np.empty((0, len(self.names_) * (self.width + 2)), dtype=np.float32)
        )

    def predict(self, bags, roots):
        import torch

        with torch.no_grad():
            logits = self.head_(torch.from_numpy(self.transform(bags, roots))).flatten()
            if self.task == "classification":
                return torch.sigmoid(logits).numpy()
            return logits.numpy() * self.target_scale_ + self.target_center_
