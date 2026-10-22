"""Fixed equal-weight composition of fully authenticated matched raw heads.

Three bounded representations are combined without new fits, per-fold weight
optimization, new graph candidates, or any official scoring. Existing component
fallback predictions are respected. The unchanged family-wise gate decides.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_recovered_readouts import matched_gate
from scripts.benchmark_sequence_recovery_20261020 import verify
from scripts.engagement_local_comparison import freeze
from scripts.run_gap_followup_queue import digest, dump


WEIGHTS = (.125, .5, 1.)
RECIPES = tuple(f"bounded_view_mean_{w:g}" for w in WEIGHTS)


def compose(native, heads):
    native, heads = np.asarray(native, float), np.asarray(heads, float)
    if native.ndim != 1 or heads.shape != (3, len(native)) or not np.isfinite(heads).all() or not np.isfinite(native).all():
        raise ValueError("Exactly three aligned finite matched component heads required")
    mean = heads.mean(axis=0)
    unchanged = np.all(heads == native[None, :], axis=0)
    result = {}
    for name, w in zip(RECIPES, WEIGHTS):
        prediction = (1-w)*native+w*mean
        prediction[unchanged] = native[unchanged]
        result[name] = prediction
    return result


def run(args):
    if len(args.component) != 3 or len(set(args.component)) != 3:
        raise ValueError("Three distinct complete raw-head audits required")
    headers, done, files = [], [], []
    for root in args.component:
        m = json.loads((root / "manifest.json").read_text())
        r = json.loads((root / "results.json").read_text())
        if not r.get("complete_grid") or r.get("official_test_accessed") is not False:
            raise ValueError("Complete validation-only component evidence required")
        verify(m["sources"])
        verify(m["inputs"])
        headers.append(m)
        done.append(r)
        files += [root / "manifest.json", root / "results.json"]
    first = headers[0]
    grid = {(f, s) for f in first["folds"] for s in range(3)}
    for m, r in zip(headers, done):
        if any(m[k] != first[k] for k in ("metadata", "policy", "model_params", "folds", "seeds")) or m["seeds"] != [0, 1, 2]:
            raise ValueError("Same graph, native support, actual production learner and grid required")
        if len(r["records"]) != len(grid) or {(x["fold"], x["seed"]) for x in r["records"]} != grid:
            raise ValueError("Incomplete/duplicate component grid")
    paired, reports = [], []
    args.output.mkdir(parents=True, exist_ok=True)
    for fold, seed in sorted(grid):
        cases = []
        for root, m, r in zip(args.component, headers, done):
            row = next(x for x in r["records"] if (x["fold"], x["seed"]) == (fold, seed))
            path = root / f"fold{fold}/seed{seed}.npz"
            if row["vectors_sha256"] != digest(path) or row["manifest_sha256"] != digest(root / "manifest.json"):
                raise ValueError("Component vectors or producer identity changed")
            files.append(path)
            with np.load(path) as arrays:
                cases.append({k: arrays[k].copy() for k in ("truth", "groups", "control", "head")})
        reference = cases[0]
        for case in cases[1:]:
            for key in ("truth", "groups", "control"):
                np.testing.assert_array_equal(reference[key], case[key])
        vectors = compose(reference["control"], [c["head"] for c in cases])
        origin = next(x["origin"] for x in done[0]["records"] if (x["fold"], x["seed"]) == (fold, seed))
        paired.append(dict(origin=origin, seed=seed, truth=reference["truth"], groups=reference["groups"],
            control=reference["control"], vectors=vectors))
        path = args.output / f"fold{fold}/seed{seed}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, truth=reference["truth"], groups=reference["groups"], control=reference["control"], **vectors)
        reports.append(dict(fold=fold, seed=seed, origin=origin, vectors_sha256=digest(path)))
    identity = dict(profile="three_fixed_bounded_view_consensus_v1", metadata=first["metadata"],
        policy=first["policy"], model_params=first["model_params"], folds=first["folds"], seeds=[0, 1, 2],
        recipes=RECIPES, allowance=1024, new_fits=0, component_headers=headers,
        sources={str(Path(__file__)): digest(__file__)}, inputs={str(p): digest(p) for p in files},
        official_test_accessed=False, default_changed=False)
    freeze(args.output / "manifest.json", identity)
    gate = matched_gate(paired, metric="roc_auc", recipes=RECIPES, allowance=1024)
    verify(identity["sources"])
    verify(identity["inputs"])
    dump(args.output / "results.json", dict(selected=gate["selected"], gate=gate, records=reports,
        complete_grid=True, official_test_accessed=False, default_changed=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--component", type=Path, required=True, action="append")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
