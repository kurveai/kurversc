from types import SimpleNamespace

import numpy as np
import pandas as pd

from kurversc.pooled_alignment import AlignedNeighborhoodEncoder, align_message_path


class IdentityEncoder:
    def encode(self, x):
        return np.asarray(x)

    def predict_pooled(self, x):
        return (np.asarray(x) ** 2).sum(axis=1)


def test_alignment_recovers_rotation_and_preserves_parent_predictions():
    rng = np.random.default_rng(0)
    local = rng.normal(size=(100, 4))
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    reference = local @ rotation + 3
    model = IdentityEncoder()
    aligned = AlignedNeighborhoodEncoder.from_anchor(model, local, reference)
    np.testing.assert_allclose(aligned.encode(local), reference, atol=1e-12)
    pooled = local.reshape(20, 5, 4).mean(axis=1)
    mapped = aligned.encode(local).reshape(20, 5, 4).mean(axis=1)
    np.testing.assert_allclose(
        aligned.predict_pooled(mapped), model.predict_pooled(pooled), atol=1e-12
    )


def test_alignment_ignores_later_rows_and_all_target_values():
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
    training = pd.DataFrame(
        {
            "ready": pd.to_datetime(["2001-01-01"] * 50 + ["2010-01-01"] * 50),
            "y": rng.normal(size=100),
        }
    )

    def align(frame, labels):
        family = SimpleNamespace(
            models_={
                "p": [
                    (pd.Timestamp("2001-01-01"), IdentityEncoder(), tuple(x), 0),
                    (pd.Timestamp("2010-01-01"), IdentityEncoder(), tuple(x), 0),
                ]
            }
        )
        audit = align_message_path(
            family,
            "p",
            frame,
            np.arange(100),
            labels,
            available_at="ready",
            max_rows=30,
        )
        assert audit[0]["anchor_latest_available"] == "2001-01-01 00:00:00"
        return family.models_["p"][1][1]

    first = align(x, training)
    changed = x.copy()
    changed.iloc[50:] = 9999
    labels = training.copy()
    labels["y"] = -9999
    second = align(changed, labels)
    np.testing.assert_allclose(first.rotation, second.rotation, atol=1e-12)
    np.testing.assert_allclose(first.offset, second.offset, atol=1e-12)
