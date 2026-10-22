"""Optional parent-supervised encoders for GraphReduce neighborhoods.

No benchmark/schema names, no graph extraction of its own, and no query labels.
PyTorch is optional and imported only when fitting or applying an encoder.
"""

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

from .learned_messages import LearnedMessageFamily


class NeighborhoodEncoder:
    """A small phi/mean/rho network, with matched row- versus parent-loss arms."""

    def __init__(
        self,
        *,
        width=1,
        hidden=32,
        epochs=30,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.0001,
        objective="parent",
        task="regression",
        random_state=0,
    ):
        if objective not in {"row", "parent"}:
            raise ValueError("objective must be 'row' or 'parent'")
        if task not in {"regression", "classification"}:
            raise ValueError("Only regression and binary classification are supported")
        if (
            min(width, hidden, epochs, batch_size) < 1
            or learning_rate <= 0
            or weight_decay < 0
        ):
            raise ValueError("Encoder budgets must be positive")
        self.width, self.hidden, self.epochs = width, hidden, epochs
        self.batch_size, self.learning_rate = batch_size, learning_rate
        self.weight_decay, self.objective = weight_decay, objective
        self.task, self.random_state = task, random_state

    def _input(self, x):
        values = np.asarray(x, dtype=np.float32)
        missing = ~np.isfinite(values)
        values = np.where(missing, self.center_, values)
        scaled = np.clip((values - self.center_) / self.scale_, -8, 8)
        return np.concatenate([scaled, missing.astype(np.float32)], axis=1)

    @staticmethod
    def _pool(z, groups, counts):
        import torch

        pooled = torch.zeros((len(counts), z.shape[1]), dtype=z.dtype)
        return pooled.index_add(0, groups, z) / counts[:, None]

    def fit(self, x, y, groups):
        import torch
        from torch import nn

        started = time.monotonic()
        x, y, groups = (
            np.asarray(x, dtype=np.float32),
            np.asarray(y, dtype=float),
            np.asarray(groups),
        )
        if x.ndim != 2 or len(x) != len(y) or len(x) != len(groups) or not len(x):
            raise ValueError("Nonempty row features, labels and parent IDs must align")
        if not np.isfinite(y).all():
            raise ValueError("Training labels must be finite")
        _, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
        parent_y = y[first]
        if not np.array_equal(y, parent_y[inverse]):
            raise ValueError(
                "All rows in a parent neighborhood must have the same target"
            )
        if self.task == "classification" and not set(parent_y).issubset({0, 1}):
            raise ValueError("Binary targets must be encoded as 0/1")
        rng = np.random.default_rng(self.random_state)
        sample = x[np.linspace(0, len(x) - 1, min(10000, len(x)), dtype=int)]
        finite = np.isfinite(sample)
        n = finite.sum(axis=0)
        safe = np.where(finite, sample, 0).astype(np.float64)
        self.center_ = (safe.sum(axis=0) / np.maximum(n, 1)).astype(np.float32)
        variance = np.where(finite, (safe - self.center_) ** 2, 0).sum(
            axis=0
        ) / np.maximum(n, 1)
        self.scale_ = np.sqrt(np.maximum(variance, 0)).astype(np.float32)
        self.scale_[self.scale_ < 1e-6] = 1
        self.target_center_ = (
            float(np.median(parent_y)) if self.task == "regression" else 0.0
        )
        self.target_scale_ = (
            max(float(np.std(parent_y)), 1e-3) if self.task == "regression" else 1.0
        )
        tx = torch.from_numpy(self._input(x))
        ty = torch.tensor(
            (parent_y - self.target_center_) / self.target_scale_, dtype=torch.float32
        )
        counts = np.bincount(inverse)
        order = np.argsort(inverse, kind="stable")
        offsets = np.r_[0, np.cumsum(counts)]
        losses = []
        # Local RNG context does not perturb unrelated callers' initialization.
        # CPU + fixed parent batches avoids backend-dependent atomic GPU sums.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            self.encoder_ = nn.Sequential(
                nn.Linear(tx.shape[1], self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.width),
                nn.Tanh(),
            )
            self.head_ = nn.Sequential(
                nn.Linear(self.width, 16), nn.ReLU(), nn.Linear(16, 1)
            )
            optimizer = torch.optim.AdamW(
                [*self.encoder_.parameters(), *self.head_.parameters()],
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            loss_fn = (
                nn.BCEWithLogitsLoss(reduction="none")
                if self.task == "classification"
                else nn.L1Loss(reduction="none")
            )
            for epoch in range(self.epochs):
                permutation = rng.permutation(len(parent_y))
                total = 0.0
                for start in range(0, len(permutation), self.batch_size):
                    parents = permutation[start : start + self.batch_size]
                    row_ids = np.concatenate(
                        [order[offsets[p] : offsets[p + 1]] for p in parents]
                    )
                    batch_counts = counts[parents]
                    batch_groups = torch.from_numpy(
                        np.repeat(np.arange(len(parents)), batch_counts)
                    )
                    count_tensor = torch.tensor(batch_counts, dtype=torch.float32)
                    latent = self.encoder_(tx[row_ids])
                    if self.objective == "parent":
                        pooled = self._pool(latent, batch_groups, count_tensor)
                        error = loss_fn(self.head_(pooled).flatten(), ty[parents])
                    else:
                        row_loss = loss_fn(
                            self.head_(latent).flatten(), ty[parents][batch_groups]
                        )
                        error = (
                            torch.zeros(len(parents)).index_add(
                                0, batch_groups, row_loss
                            )
                            / count_tensor
                        )
                    loss = error.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite neighborhood training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [*self.encoder_.parameters(), *self.head_.parameters()], 5
                    )
                    optimizer.step()
                    total += float(loss.detach()) * len(parents)
                if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == self.epochs:
                    losses.append({"epoch": epoch + 1, "loss": total / len(parent_y)})
        self.encoder_.eval()
        self.head_.eval()
        latent = self.encode(x)
        sums = np.zeros((len(parent_y), self.width), dtype=np.float64)
        np.add.at(sums, inverse, latent)
        predicted = self.predict_pooled(sums / counts[:, None])
        self.fit_audit_ = dict(
            objective=self.objective,
            message_width=self.width,
            epochs=self.epochs,
            parent_batch_size=self.batch_size,
            loss_trace=losses,
            train_parent_mae=float(np.mean(np.abs(parent_y - predicted))),
            fit_seconds=time.monotonic() - started,
        )
        return self

    def encode(self, x):
        import torch

        chunks = []
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                batch = torch.from_numpy(self._input(x[start : start + 4096]))
                chunks.append(self.encoder_(batch).numpy())
        return (
            np.concatenate(chunks)
            if chunks
            else np.empty((0, self.width), dtype=np.float32)
        )

    def predict_pooled(self, means):
        import torch

        with torch.no_grad():
            values = self.head_(
                torch.tensor(np.asarray(means), dtype=torch.float32)
            ).flatten()
            if self.task == "classification":
                return torch.sigmoid(values).numpy()
            return values.numpy() * self.target_scale_ + self.target_center_


@dataclass
class PooledMessageFamily(LearnedMessageFamily):
    """Opt-in, forward-only vector messages trained on whole neighborhoods.

    ``objective='row'`` supplies the same-network weak-supervision control.
    Inference returns pooled latent features plus a calibrated parent prediction.
    Encoder initialization and all normalization are fitted independently inside
    each permitted temporal training prefix; no query labels are consulted.
    """

    message_width: int = 1
    encoder_hidden: int = 32
    encoder_epochs: int = 30
    parent_batch_size: int = 128
    encoder_learning_rate: float = 0.001
    encoder_weight_decay: float = 0.0001
    encoder_random_state: int | None = None
    objective: str = "parent"

    def _message_names(self):
        return tuple(f"latent{i}" for i in range(self.message_width))

    def _scorer_columns(self, encoded):
        # Retain the frozen, target-independent feature schema across dates.
        # Normalization still uses only rows with historically available labels.
        return tuple(encoded.columns)

    def _fit_scorer(self, x, y, ids):
        return NeighborhoodEncoder(
            width=self.message_width,
            hidden=self.encoder_hidden,
            epochs=self.encoder_epochs,
            batch_size=self.parent_batch_size,
            learning_rate=self.encoder_learning_rate,
            weight_decay=self.encoder_weight_decay,
            objective=self.objective,
            task=self.task_,
            random_state=self.random_state
            if self.encoder_random_state is None
            else self.encoder_random_state,
        ).fit(x.to_numpy(), y, ids)

    def _predict_scorer(self, model, x):
        return model.encode(x.to_numpy())

    def _finish_messages(self, message, examples, versions):
        message["parent_prediction"] = np.nan
        for cutoff in sorted(set(pd.to_datetime(examples[self.timestamp_]))):
            candidates = [v for v in versions if v[0] < cutoff]
            mask = (examples[self.timestamp_] == cutoff) & (message.neighbor_count > 0)
            if not candidates or not mask.any():
                continue
            _, model, _, constant = max(candidates, key=lambda v: v[0])
            if model is None:
                message.loc[mask, "parent_prediction"] = constant
            else:
                means = message.loc[mask, [f"{c}__mean" for c in self._message_names()]]
                message.loc[mask, "parent_prediction"] = model.predict_pooled(means)
        return message
