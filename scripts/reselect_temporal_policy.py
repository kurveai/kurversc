"""Rescore immutable retrospective predictions with a versioned generic gate.

No graph/model fitting, outer split, official predictions or test scores.
Useful for checking a selector correction without repeating expensive fits.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from kurversc.temporal_policy import (
    _score,
    add_incumbent_blends,
    select_temporal_policy,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blend-weights", default="")
    args = parser.parse_args()
    original = json.loads((args.input / "results.json").read_text())
    manifest = json.loads((args.input / "manifest.json").read_text())
    if manifest["metadata"]["validation_cutoffs"] > 1 and not manifest.get(
        "phase_frozen_cache"
    ):
        raise ValueError(
            "Multi-cutoff audit is not phase-frozen and cannot promote a policy"
        )
    evidence = []
    for number, fold in enumerate(manifest["folds"]):
        for seed in manifest["seeds"]:
            for policy in manifest["policies"]:
                name = f"fold{number}-{policy['name']}-seed{seed}.npz"
                path = args.input / name
                if not path.exists():
                    continue
                with np.load(path) as arrays:
                    evidence.append(
                        dict(
                            origin=fold["origin"],
                            seed=seed,
                            policy=policy["name"],
                            **{
                                key: arrays[key].copy()
                                for key in ("truth", "prediction", "groups")
                            },
                        )
                    )
    task = "classification" if original["metric"] == "roc_auc" else "regression"
    weights = tuple(float(value) for value in args.blend_weights.split(",") if value)
    evidence, recipes = add_incumbent_blends(evidence, weights=weights)
    decision = select_temporal_policy(evidence, task=task, deployment_seed=0)
    if decision["selected"] in recipes:
        recipe = recipes[decision["selected"]]
        selected = {
            **next(
                p for p in manifest["policies"] if p["name"] == recipe["base_policy"]
            ),
            "name": decision["selected"],
            "blend_weight": recipe["blend_weight"],
        }
    else:
        selected = next(
            p for p in manifest["policies"] if p["name"] == decision["selected"]
        )
    result = {
        **original,
        "decision": decision,
        "selected_policy": selected,
        "source_prediction_audit": str(args.input),
        "selection_source_sha256": hashlib.sha256(
            Path("kurversc/temporal_policy.py").read_bytes()
        ).hexdigest(),
    }
    result["records"] = [
        *original["records"],
        *[
            dict(
                origin=row["origin"],
                seed=row["seed"],
                policy=row["policy"],
                score=_score(row["truth"], row["prediction"], task),
                recipe=recipes[row["policy"]],
            )
            for row in evidence
            if row["policy"] in recipes
        ],
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    new_manifest = {
        **manifest,
        "selection_replay": dict(
            source=str(args.input),
            prediction_audit_sha256=hashlib.sha256(
                (args.input / "results.json").read_bytes()
            ).hexdigest(),
            selection_source_sha256=result["selection_source_sha256"],
            deployment_seed=0,
            blend_weights=list(weights),
        ),
    }
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != new_manifest:
        raise ValueError("Existing selector replay has different immutable inputs")
    path.write_text(json.dumps(new_manifest, indent=2) + "\n")
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
