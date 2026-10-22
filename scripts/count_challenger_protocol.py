"""Validate a bounded conditional-count challenger before official replay.

No task names or official scores enter this protocol. Strict research evidence
must improve every case; an explicitly declared production-rule audit instead
requires recent seed stability and the existing 1% older-fold cap (all seeds).
Both require independently replayed controls and both paired confidence gates.
"""

import math


WEIGHTS = (.125, .5, 1.)
RECIPES = tuple(f"count_likelihood_{w:g}" for w in WEIGHTS)


def validated_count_recipe(result, manifest, metadata):
    if result.get("selected") == "incumbent":
        return None
    chosen = result.get("selected")
    strict = manifest.get("profile") == "conditional_poisson_median"
    production = (manifest.get("profile") == "complete_conditional_count_production_rule"
        and manifest.get("selection_rule") == "existing_production_temporal_veto"
        and result.get("selection_rule") == manifest["selection_rule"]
        and manifest.get("older_all_seed_relative_loss_cap") == .01)
    if (chosen not in RECIPES or not result.get("complete_grid")
            or result.get("official_test_accessed") is not False
            or result.get("default_changed") is not False
            or not (strict or production)
            or manifest.get("metadata") != metadata
            or metadata.get("metric") != "mae"
            or metadata.get("fold") != "current"
            or metadata.get("atomic_date_replay") is not True
            or tuple(manifest.get("weights", ())) != WEIGHTS
            or manifest.get("seeds") != [0, 1, 2]
            or manifest.get("allowance") != 1024):
        raise ValueError("A complete, validation-only declared count recipe is required")
    folds = manifest.get("folds", [])
    expected = {(f, s) for f in folds for s in (0, 1, 2)}
    records = result.get("records", [])
    observed = [(r["fold"], r["seed"]) for r in records]
    if (len(folds) < 2 or len(folds) != len(set(folds))
            or len(observed) != len(set(observed)) or set(observed) != expected):
        raise ValueError("Every declared origin and seed is required exactly once")
    for record in records:
        error = record.get("native_archive_max_error", float("inf"))
        if not math.isfinite(error) or not 0 <= error <= 1e-9:
            raise ValueError("Native control must independently reproduce")
        for panel in ("fresh", "archive"):
            control = record["scores"][panel]
            candidate = record["scores"][panel + "_" + chosen]
            must_improve = strict or record["fold"] == max(folds)
            if (not math.isfinite(control) or not math.isfinite(candidate)
                    or (must_improve and candidate >= control)
                    or (not must_improve and candidate-control > .01*control+1e-12)):
                raise ValueError("Recent improvement and older-fold safety must hold")
    if set(result.get("gates", {})) != {"fresh", "archive"}:
        raise ValueError("Both independent and archived confidence gates are required")
    for gate in result["gates"].values():
        candidate = gate.get("candidates", {}).get(chosen, {})
        latest = (candidate.get("folds") or [{}])[-1]
        recent, temporal = latest.get("recent_lower_bound"), latest.get("temporal_gain_lower_bound")
        if (candidate.get("reason") != "eligible" or recent is None or temporal is None
                or not math.isfinite(recent) or not math.isfinite(temporal)
                or min(recent, temporal) <= 0):
            raise ValueError("Recent and temporal paired confidence must both pass")
    return dict(selected=chosen, weight=float(chosen.rsplit("_", 1)[1]),
                origins=len(folds), seeds=[0, 1, 2], official_targets_used_for_selection=False)


def count_stage_eligible(raw, policy, recovered):
    """Conservative generic scope: never displace another admitted readout."""
    meta = raw.get("metadata", {})
    return (meta.get("metric") == "mae" and meta.get("fold") == "current"
        and meta.get("atomic_date_replay") is True and raw.get("seeds") == [0, 1, 2]
        and len(raw.get("folds", [])) >= 2 and policy.get("mode") == "frame_ensemble"
        and policy.get("readout") == "direct" and policy.get("blend_weight") == 1
        and recovered.get("selected") == "integer_counts"
        and recovered.get("projection") == "integer_counts"
        and recovered.get("baseline_signal") == "incumbent")
