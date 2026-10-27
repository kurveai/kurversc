"""Necessary-condition certificates for the existing temporal selector.

This module never admits a candidate. A base fit may be omitted only when
every shipped readout of that fit (including incumbent blends) already fails
a deterministic gate in ``select_temporal_policy``. All surviving candidates
still need the complete original origin/seed grid and confidence calculation.
"""

import hashlib

import numpy as np
import pandas as pd

from kurversc.temporal_policy import _score, add_incumbent_blends


def evidence_digest(row):
    digest = hashlib.sha256()
    for key in ("origin", "seed", "policy"):
        digest.update(str(row[key]).encode())
        digest.update(b"\0")
    # pandas hashing also supports native string labels and entity identifiers.
    for key in ("truth", "prediction", "groups"):
        digest.update(pd.util.hash_pandas_object(
            pd.Series(np.asarray(row[key])), index=False
        ).to_numpy().tobytes())
    return digest.hexdigest()


def selector_vetoes(evidence, *, origins, seeds, policies, task,
                    blend_weights=(), deployment_seed=0,
                    classification_loss_cap=.005,
                    regression_relative_loss_cap=.01):
    """Return base-policy vetoes sufficient regardless of uncomputed fits.

    Origins and seeds MUST be the full, predeclared experiment, not the subset
    observed so far. In particular, an old observed origin is never called the
    recent origin simply because the actual recent origin is still pending.
    The caller must authenticate checkpoint identity before supplying evidence.
    """
    origins = tuple(sorted(pd.Timestamp(v) for v in origins))
    seeds, policies = tuple(seeds), tuple(policies)
    if (task not in {"classification", "regression"} or len(origins) < 2
            or len(set(origins)) != len(origins) or len(set(seeds)) != len(seeds)
            or len(seeds) < 2 or deployment_seed not in seeds
            or len(set(policies)) != len(policies) or "incumbent" not in policies
            or min(classification_loss_cap, regression_relative_loss_cap) < 0):
        raise ValueError("A complete, distinct selector experiment is required")
    normalized, seen = [], set()
    for item in evidence:
        row = {**item, "origin": str(pd.Timestamp(item["origin"])),
               "seed": int(item["seed"])}
        key = (pd.Timestamp(row["origin"]), row["seed"], row["policy"])
        if key in seen:
            raise ValueError("Duplicate origin/seed/policy evidence")
        seen.add(key)
        if key[0] not in origins or key[1] not in seeds or key[2] not in policies:
            raise ValueError("Evidence is outside the declared experiment")
        arrays = [np.asarray(row[k]) for k in ("truth", "prediction", "groups")]
        y, prediction, groups = arrays
        if (y.ndim != 1 or not len(y) or y.shape != prediction.shape
                or y.shape != groups.shape or pd.isna(y).any()
                or pd.isna(groups).any() or not np.isfinite(prediction).all()
                or (task == "regression" and not np.isfinite(y).all())):
            raise ValueError("Aligned finite nonempty evidence is required")
        _score(y, prediction, task)  # Validate binary label support as the selector does.
        normalized.append(row)
    references = {}
    for row in normalized:
        if row["policy"] == "incumbent":
            reference = references.setdefault(row["origin"], row)
            if any(not np.array_equal(row[k], reference[k]) for k in ("truth", "groups")):
                raise ValueError("Paired policies must score identical ordered rows")
    expanded, recipes = add_incumbent_blends(normalized, weights=blend_weights)
    rows = {(pd.Timestamp(r["origin"]), r["seed"], r["policy"]): r
            for r in expanded}
    original_rows = {(pd.Timestamp(r["origin"]), r["seed"], r["policy"]): r
                     for r in normalized}
    # Verify alignment even when this pair would not be a useful veto witness.
    for row in normalized:
        control = rows.get((pd.Timestamp(row["origin"]), row["seed"], "incumbent"))
        if control is not None and any(not np.array_equal(row[k], control[k])
                                       for k in ("truth", "groups")):
            raise ValueError("Paired policies must score identical ordered rows")
    result = {}
    for base in policies:
        if base == "incumbent":
            continue
        names = [base, *[f"{base}__blend_{w:g}" for w in blend_weights]]
        witnesses = {}
        for name in names:
            # Most recent deployed seed first, then other recent seeds, then
            # older deployed-seed origins. No older nondeployment-seed veto.
            pairs = [(origins[-1], deployment_seed),
                     *[(origins[-1], s) for s in sorted(seeds) if s != deployment_seed],
                     *[(o, deployment_seed) for o in origins[:-1]]]
            for origin, seed in pairs:
                a, b = rows.get((origin, seed, "incumbent")), rows.get((origin, seed, name))
                if a is None or b is None:
                    continue
                av, bv = (_score(a["truth"], a["prediction"], task),
                          _score(b["truth"], b["prediction"], task))
                gain = bv-av if task == "classification" else av-bv
                cap = (classification_loss_cap if task == "classification"
                       else regression_relative_loss_cap*av)
                reason = None
                if origin == origins[-1] and gain <= 0:
                    reason = ("no_recent_improvement" if seed == deployment_seed
                              else "recent_gain_not_seed_stable")
                elif origin != origins[-1] and gain < -cap:
                    reason = "older_origin_regression"
                if reason:
                    witnesses[name] = dict(reason=reason, origin=str(origin), seed=seed,
                        control=av, candidate=bv, gain=gain, practical_loss_cap=cap,
                        control_evidence_sha256=evidence_digest(a),
                        base_evidence_sha256=evidence_digest(original_rows[(origin, seed, base)]),
                        blend_weight=recipes.get(name, {}).get("blend_weight", 1.))
                    break
        if len(witnesses) == len(names):
            result[base] = dict(all_readouts_vetoed=True, witnesses=witnesses)
    return result


def verify_selection_barrier(evidence, *, origins, seeds, policies, task,
                             blend_weights=(), unavailable=()):
    """Fail closed on missing work unless it has an exact veto or support error."""
    vetoes = selector_vetoes(evidence, origins=origins, seeds=seeds,
        policies=policies, task=task, blend_weights=blend_weights)
    observed = {(pd.Timestamp(r["origin"]), int(r["seed"]), r["policy"])
                for r in evidence}
    unsupported = {(pd.Timestamp(o), int(s), p) for o, s, p in unavailable}
    for origin in origins:
        for seed in seeds:
            for policy in policies:
                key = (pd.Timestamp(origin), seed, policy)
                if key not in observed and key not in unsupported and policy not in vetoes:
                    raise ValueError(f"Incomplete non-vetoed selector evidence: {key}")
    # Every veto contains real evidence for each base and blend, so the original
    # selector still sees the ENTIRE menu for its Bonferroni correction.
    expanded, recipes = add_incumbent_blends(evidence, weights=blend_weights)
    expected = {p for p in policies if p != "incumbent"}
    expected |= {f"{p}__blend_{w:g}" for p in policies if p != "incumbent"
                 for w in blend_weights}
    present = {r["policy"] for r in expanded} - {"incumbent"}
    # Completely unsupported policies have no test in the original selector.
    absent = expected-present
    if any(p.split("__blend_")[0] not in {k[2] for k in unsupported} for p in absent):
        raise ValueError("Selector multiplicity menu unexpectedly shrank")
    return vetoes
