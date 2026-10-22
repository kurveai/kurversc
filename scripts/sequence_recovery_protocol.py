"""Finite, validation-only sequence readout selection for arbitrary schemas.

The two-stage menu is the already qualified procedure, not a task-name lookup.
All components must be independently fitted under one production contract.
"""

import numpy as np

from scripts.benchmark_recovered_readouts import matched_gate


BASES = {
    "sequence_basis_eighth": (.875, .125, 0.),
    "sequence_basis_and_exact_time": (.765625, .125, .109375),
}
HEADS = {"basis": (0., 1., 0.), "exact": (0., 0., 1.), "paired": (0., .5, .5)}
WEIGHTS = (.125, .5, 1.)
ALLOWANCE = 1024


def combine(native, basis, exact, coefficients):
    components = np.asarray([native, basis, exact], dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if (components.ndim != 2 or components.shape[0] != 3 or not components.shape[1]
            or not np.isfinite(components).all() or np.any((components < 0) | (components > 1))
            or coefficients.shape != (3,) or not np.isfinite(coefficients).all()
            or np.any(coefficients < 0) or coefficients.sum() != 1):
        raise ValueError("Aligned probabilities and fixed convex sequence coefficients required")
    # Match the qualified operation order, not a BLAS-dependent matrix product.
    return coefficients[0]*components[0]+coefficients[1]*components[1]+coefficients[2]*components[2]


def choose(records):
    """Select from complete (origin, seed, truth, groups, native, basis, exact).

    No query features, official answers, task names, or published scores enter.
    Missing or duplicate cases are not silently replaced with another seed.
    """
    first = []
    for row in records:
        first.append({k: row[k] for k in ("origin", "seed", "truth", "groups")})
        first[-1].update(control=np.asarray(row["native"], float), vectors={
            n: combine(row["native"], row["basis"], row["exact"], c) for n, c in BASES.items()})
    primary = matched_gate(first, metric="roc_auc", recipes=tuple(BASES), allowance=ALLOWANCE)
    base = primary["selected"]
    coefficients = np.asarray((1., 0., 0.) if base == "incumbent" else BASES[base])
    second, menu = [], {}
    for name, head in HEADS.items():
        for weight in WEIGHTS:
            menu[f"sequence_weight_{name}_{weight:g}"] = (1-weight)*coefficients+weight*np.asarray(head)
    for row, protected in zip(records, first):
        control = protected["control"] if base == "incumbent" else protected["vectors"][base]
        heads = dict(basis=np.asarray(row["basis"], float), exact=np.asarray(row["exact"], float),
            paired=.5*np.asarray(row["basis"], float)+.5*np.asarray(row["exact"], float))
        second.append({**{k: row[k] for k in ("origin", "seed", "truth", "groups")},
            "control": control, "vectors": {f"sequence_weight_{name}_{weight:g}": (1-weight)*control+weight*head
                for name, head in heads.items() for weight in WEIGHTS}})
    refinement = matched_gate(second, metric="roc_auc", recipes=tuple(menu), allowance=ALLOWANCE)
    selected = refinement["selected"]
    if selected != "incumbent":
        coefficients = menu[selected]
    else:
        selected = base
    return dict(selected=selected, protected_procedure=base, coefficients=coefficients.tolist(),
        primary_gate=primary, refinement_gate=refinement, deployment_seed=0,
        official_targets_used_for_selection=False)
