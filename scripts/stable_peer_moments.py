"""Research-only canonicalization of cancellation noise in peer moments.

No schema/task names, target access or model choices. Not a default feature
change: a new representation must pass its own matched validation gate.
"""

import numpy as np
import pandas as pd


def stable_peer_moments(frame, *, categorical=()):
    output = frame.copy()
    protected = set(categorical)
    audit = []
    for name in frame:
        if name in protected or not pd.api.types.is_numeric_dtype(frame[name]):
            continue
        values = frame[name].to_numpy(dtype="float64", na_value=np.nan)
        if name.endswith("__weighted_std"):
            mean_name = name.removesuffix("__weighted_std") + "__weighted_mean"
            if mean_name not in frame:
                raise ValueError("Weighted standard deviation needs its paired mean")
            mean = frame[mean_name].to_numpy(dtype="float64", na_value=np.nan)
            # E[y²] - E[y]² loses all relative accuracy close to a constant.
            # The scale-dependent floor is conservative for double precision,
            # unlike a fixed absolute threshold that erases small-unit signal.
            floor = np.sqrt(32 * np.finfo("float64").eps) * np.abs(mean)
            noise = np.isfinite(values) & np.isfinite(mean) & (values >= 0) & (values <= floor)
            audit.append(dict(column=name, cancellation_noise_rows=int((noise & (values != 0)).sum())))
            values = values.copy()
            values[noise] = 0.
        # CatBoost consumes numeric payloads at float32 precision. Make that
        # implicit model boundary explicit; categorical values remain intact.
        output[name] = values.astype("float32")
    return output, audit
