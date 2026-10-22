"""Optional label-free alignment of temporal message coordinates.

Align to the first fitted encoder on input rows available to that encoder.
Supervised models and their parent predictions are unchanged. No query features
or target values are used to estimate the orthogonal maps.
"""

import numpy as np
import pandas as pd


class AlignedNeighborhoodEncoder:
    def __init__(self, model, rotation, offset):
        self.model = model
        self.rotation = np.asarray(rotation)
        self.offset = np.asarray(offset)
        self.fit_audit_ = getattr(model, "fit_audit_", {})

    def encode(self, x):
        return self.model.encode(x) @ self.rotation + self.offset

    def predict_pooled(self, means):
        local = (np.asarray(means) - self.offset) @ self.rotation.T
        return self.model.predict_pooled(local)

    @classmethod
    def from_anchor(cls, model, local, reference):
        local, reference = (
            np.asarray(local, dtype=float),
            np.asarray(reference, dtype=float),
        )
        if local.shape != reference.shape or local.ndim != 2 or not len(local):
            raise ValueError(
                "Aligned anchor matrices must have the same nonempty shape"
            )
        if not np.isfinite(local).all() or not np.isfinite(reference).all():
            raise ValueError("Alignment anchors must be finite")
        a, b = local.mean(axis=0), reference.mean(axis=0)
        u, _, vt = np.linalg.svd((local - a).T @ (reference - b), full_matrices=False)
        rotation = u @ vt
        return cls(model, rotation, b - a @ rotation)


def align_message_path(
    family, path, encoded, ids, training, *, available_at, max_rows=2048
):
    """Align a fitted path in place using only its earliest eligible inputs."""
    if max_rows < 1:
        raise ValueError("Alignment anchor budget must be positive")
    versions = family.models_[path]
    fitted = [v for v in versions if v[1] is not None]
    if not fitted:
        return []
    first_time, first_model, first_columns, _ = fitted[0]
    if isinstance(first_model, AlignedNeighborhoodEncoder):
        raise ValueError("This path has already been aligned")
    availability = pd.to_datetime(training.loc[ids, available_at]).to_numpy()
    positions = np.flatnonzero(availability <= pd.Timestamp(first_time).to_datetime64())
    if not len(positions):
        raise ValueError("No common historical rows are available for alignment")
    positions = positions[
        np.linspace(0, len(positions) - 1, min(max_rows, len(positions)), dtype=int)
    ]
    anchor = encoded.iloc[positions]
    reference = first_model.encode(anchor[list(first_columns)].to_numpy())
    audit, aligned = [], []
    for latest, model, columns, constant in versions:
        if model is None or model is first_model:
            aligned.append((latest, model, columns, constant))
            continue
        local = model.encode(anchor[list(columns)].to_numpy())
        wrapper = AlignedNeighborhoodEncoder.from_anchor(model, local, reference)
        mapped = local @ wrapper.rotation + wrapper.offset
        audit.append(
            dict(
                latest_label_available=str(latest),
                anchor_latest_available=str(first_time),
                anchor_rows=len(positions),
                unaligned_rmse=float(np.sqrt(np.mean((local - reference) ** 2))),
                aligned_rmse=float(np.sqrt(np.mean((mapped - reference) ** 2))),
                parent_prediction_max_change=float(
                    np.max(
                        abs(
                            model.predict_pooled(local) - wrapper.predict_pooled(mapped)
                        )
                    )
                ),
            )
        )
        aligned.append((latest, wrapper, columns, constant))
    family.models_[path] = aligned
    return audit
