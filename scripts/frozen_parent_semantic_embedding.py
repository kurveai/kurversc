"""Research-only pretrained parent-text annotation with a bounded train-only basis.

No labels, entity identifiers, schema-specific rules, or validation documents
enter the fitted projection. Model weights are pinned and never fine-tuned.
Heavy libraries/weights are loaded lazily, allowing lightweight local tests.
"""

import html
import re

import numpy as np
import pandas as pd


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class FrozenParentSemanticEmbedding:
    def __init__(self, *, dimensions=32, batch_size=64, threads=4, backend=None):
        if not 1 <= dimensions <= 32 or not 1 <= batch_size <= 128 or not 1 <= threads <= 8:
            raise ValueError("Bounded dimensions, batches and CPU threads required")
        self.dimensions, self.batch_size, self.threads = dimensions, batch_size, threads
        self.backend = backend
        self.cache = {}

    @staticmethod
    def documents(frame, columns):
        if not 1 <= len(columns) <= 4 or any(c.startswith("__") for c in columns):
            raise ValueError("One to four nonstructural parent text fields required")
        fields = []
        for column in columns:
            values = frame[column].astype("string").fillna("").replace("__missing__", "")
            fields.append([re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", str(v)[:2048]))).strip()
                           for v in values])
        # Generic positional markers retain field boundaries without encoding
        # benchmark/schema-specific feature names or identifiers.
        return ["\n".join(f"Field {i + 1}: {v}" for i, v in enumerate(row) if v)
                for row in zip(*fields)]

    def _encode_new(self, documents):
        if self.backend is not None:
            return np.asarray(self.backend(documents), dtype="float32")
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(self.threads)
        if not hasattr(self, "model"):
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
            self.model = AutoModel.from_pretrained(MODEL, revision=REVISION,
                trust_remote_code=False, use_safetensors=True).to("cpu").eval()
        output = []
        for start in range(0, len(documents), self.batch_size):
            tokens = self.tokenizer(documents[start:start + self.batch_size], padding=True,
                truncation=True, max_length=256, return_tensors="pt")
            with torch.inference_mode():
                encoded = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(encoded.dtype)
                vectors = (encoded * mask).sum(1) / mask.sum(1).clamp(min=1.)
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            output.append(vectors.cpu().numpy().astype("float32"))
        return np.concatenate(output)

    def embeddings(self, documents):
        if not hasattr(self, "width") and self.cache:
            widths = {np.asarray(v).shape for v in self.cache.values()}
            if len(widths) != 1 or len(next(iter(widths))) != 1:
                raise ValueError("One consistent cached pretrained width required")
            self.width = next(iter(widths))[0]
        unseen = sorted(set(documents) - set(self.cache) - {""})
        if unseen:
            vectors = self._encode_new(unseen)
            if vectors.ndim != 2 or len(vectors) != len(unseen) or not np.isfinite(vectors).all():
                raise ValueError("Finite aligned pretrained vectors required")
            if hasattr(self, "width") and vectors.shape[1] != self.width:
                raise ValueError("Frozen encoder width changed")
            self.width = vectors.shape[1]
            self.cache.update(zip(unseen, vectors))
        if not hasattr(self, "width"):
            raise ValueError("Nonempty training text required")
        self.cache[""] = np.zeros(self.width, dtype="float32")
        return np.asarray([self.cache[d] for d in documents], dtype="float32")

    def fit(self, frame, *, columns):
        self.columns = list(columns)
        docs = self.documents(frame, self.columns)
        nonempty = sorted(set(docs) - {""})
        if len(nonempty) < 2:
            raise ValueError("At least two distinct training documents required")
        vectors = self.embeddings(nonempty).astype("float64")
        self.mean = vectors.mean(axis=0)
        # Unique documents, lexical ordering, float64 decomposition and fixed
        # eigenvector signs avoid duplicate/entity-order and sign ambiguities.
        _, _, basis = np.linalg.svd(vectors - self.mean, full_matrices=False)
        count = min(self.dimensions, len(nonempty) - 1, self.width)
        basis = basis[:count]
        for row in basis:
            if row[np.argmax(np.abs(row))] < 0:
                row *= -1
        self.basis = basis
        self.audit = dict(model=MODEL, revision=REVISION, dimensions=count,
            embedding_width=self.width, training_rows=len(frame), unique_training_documents=len(nonempty),
            maximum_fields=4, maximum_characters_per_field=2048, maximum_wordpieces=256,
            columns=self.columns, basis_scope="Distinct earliest-mature-frame training documents only",
            supervised=False, pretrained_weights_frozen=True, structural_keys_are_predictors=False,
            external_pretraining="Model card includes Stack Exchange text; no task-label fine-tuning")
        return self

    def transform(self, frame):
        if not hasattr(self, "basis"):
            raise ValueError("Fit the training-only semantic basis first")
        vectors = self.embeddings(self.documents(frame, self.columns)).astype("float64")
        result = ((vectors - self.mean) @ self.basis.T).astype("float32")
        return pd.DataFrame(result, columns=[f"frozen_parent_semantic_{i}" for i in range(len(self.basis))])
