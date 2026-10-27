import hashlib

import numpy as np
import pandas as pd
import pytest

from scripts.frozen_parent_semantic_embedding import FrozenParentSemanticEmbedding, REVISION


def backend(documents):
    return np.array([np.random.default_rng(int.from_bytes(hashlib.sha256(d.encode()).digest()[:8], "little"))
                     .normal(size=48) for d in documents], dtype="float32")


def test_basis_uses_only_training_documents_and_is_order_duplicate_invariant():
    train = pd.DataFrame({"description":["engineer", "teacher", "gardener", "engineer"]})
    first = FrozenParentSemanticEmbedding(backend=backend).fit(train, columns=["description"])
    second = FrozenParentSemanticEmbedding(backend=backend).fit(train.iloc[[2, 0, 1]], columns=["description"])
    np.testing.assert_array_equal(first.basis, second.basis)
    query = pd.DataFrame({"description":["previously unseen", None]})
    before = first.basis.copy()
    result = first.transform(query)
    np.testing.assert_array_equal(before, first.basis)
    assert result.shape == (2, 2) and np.isfinite(result).all().all()
    assert first.audit["unique_training_documents"] == 3
    assert first.audit["revision"] == REVISION
    assert first.audit["supervised"] is False


def test_payload_cleaning_and_bounds():
    frame = pd.DataFrame({"a":["<p>Hello &amp; hi</p>", "__missing__"], "b":[None, "place"]})
    assert FrozenParentSemanticEmbedding.documents(frame, ["a", "b"]) == ["Field 1: Hello & hi", "Field 2: place"]
    with pytest.raises(ValueError):
        FrozenParentSemanticEmbedding(dimensions=33)
    with pytest.raises(ValueError):
        FrozenParentSemanticEmbedding.documents(pd.DataFrame({"__event_entity":["id"]}), ["__event_entity"])
    with pytest.raises(ValueError):
        FrozenParentSemanticEmbedding(backend=backend).fit(pd.DataFrame({"text":[None, None]}), columns=["text"])


def test_new_temporal_basis_can_reuse_entirely_cached_embeddings():
    frame = pd.DataFrame({"text":["engineer", "teacher", "gardener"]})
    first = FrozenParentSemanticEmbedding(backend=backend).fit(frame, columns=["text"])
    second = FrozenParentSemanticEmbedding(backend=lambda _:pytest.fail("Already encoded documents must not be recomputed"))
    second.cache = first.cache
    second.fit(frame, columns=["text"])
    np.testing.assert_array_equal(first.basis, second.basis)
    np.testing.assert_array_equal(first.transform(frame), second.transform(frame))
