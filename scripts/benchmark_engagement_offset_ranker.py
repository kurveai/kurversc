"""Isolated, keyed three-origin/three-seed frozen-offset ranking experiment.

Reuse verified completed entity-disjoint OOF DATA, not a previously rejected
decision. Fit a new bounded correction; protect the admitted support3 graph,
cohorts, model and its legacy component. No official labels or default changes.
"""

import argparse
import gc
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from kurversc.temporal_policy import _score
from scripts.benchmark_reliability_blending import group_masks
from scripts.engagement_local_comparison import (
    SUPPORT, TASK, context, freeze, protected, supplemental,
)
from scripts.fast_matched_bootstrap import accelerated_select_temporal_policy
from scripts.frozen_offset_ranker import (
    FrozenOffsetRanker, probability_logits, replace_strong_half,
)
from scripts.run_gap_followup_queue import digest, dump


MENU_ALLOWANCE = 132
RECIPES = {"offset_rank_full": 1., "offset_rank_quarter": .25}


def completed_oof_source(previous):
    path = previous / "results.json"
    if not path.exists():
        raise ValueError("Only a completed, fully audited source may be reused")
    report = json.loads(path.read_text())
    records = report.get("records", [])
    cases = [(r["fold"], r["seed"]) for r in records]
    expected = {(fold, seed) for fold in range(3) for seed in range(3)}
    if not report.get("complete_grid") or len(cases) != 9 or set(cases) != expected:
        raise ValueError("Complete unique three-fold/three-seed OOF source required")
    if report.get("official_test_accessed") is not False:
        raise ValueError("Validation-only OOF evidence required")
    return digest(path)


def checked_oof(ctx, previous):
    manifest = json.loads((previous / "manifest.json").read_text())
    if not manifest["engagement"] or Path(manifest["task_root"]).resolve() != TASK.resolve():
        raise ValueError("Exact protected engagement OOF source required")
    completed_oof_source(previous)
    for name, value in manifest["sources"].items():
        path = Path(name)
        if not path.exists() or digest(path) != value:
            raise ValueError(f"OOF source implementation changed: {name}")
    for name, value in manifest["evidence_sources"].items():
        if digest(name) != value:
            raise ValueError("OOF graph/training identity changed")
    source = previous / f"fold{ctx['fold']}"
    y = ctx["train"][ctx["metadata"]["target"]].to_numpy()
    base, text = np.empty(len(y)), np.empty(len(y))
    covered = np.zeros(len(y), dtype=np.int8)
    identities = []
    for number, (fit, query) in enumerate(group_masks(ctx["train"].__event_entity)):
        path = source / f"oof-group{number}.npz"
        with np.load(path) as a:
            positions = np.flatnonzero(query)
            np.testing.assert_array_equal(positions, a["positions"])
            np.testing.assert_array_equal(y[query], a["truth"])
            base[query], text[query] = a["base"], a["extra"]
        covered[query] += 1
        if set(ctx["train"].__event_entity[fit]) & set(ctx["train"].__event_entity[query]):
            raise ValueError("Own entities entered OOF fitting")
        identities.append(dict(path=str(path), sha256=digest(path),
            query_rows=int(query.sum()), entity_disjoint=True))
    np.testing.assert_array_equal(covered, np.ones(len(y)))
    probability_logits(base)
    probability_logits(text)
    control, legacy = protected(ctx, 0)
    head = 2 * control - legacy
    with np.load(source / "seed0.npz") as a:
        np.testing.assert_array_equal(a["truth"], ctx["held"][ctx["metadata"]["target"]])
        np.testing.assert_array_equal(a["groups"], ctx["held"].__event_entity)
        np.testing.assert_array_equal(a["control"], control)
        # The old predeclared fixed-half formula was C + .125 * (text - head).
        validation_text = head + 8 * (a["fixed_half"] - control)
    probability_logits(validation_text)
    return base, text, validation_text, identities


def correction_frame(state, baseline, text):
    frame = pd.DataFrame(state).select_dtypes(include=["number"]).reset_index(drop=True)
    if {"__event_entity", "__event_cutoff"} & set(frame):
        raise ValueError("Routing keys cannot enter the correction learner")
    if len(frame) != len(baseline) or len(frame) != len(text):
        raise ValueError("Aligned fixed state/prediction payload required")
    extra = pd.DataFrame(dict(offset_base_logit=probability_logits(baseline),
        offset_text_logit=probability_logits(text),
        offset_logit_difference=probability_logits(text) - probability_logits(baseline)))
    if set(extra) & set(frame):
        raise ValueError("Correction payload column collision")
    return pd.concat([frame, extra], axis=1)


def fit_fold(args):
    began = time.monotonic()
    ctx = context(args.fold)
    folder = args.output / f"fold{args.fold}"
    folder.mkdir(parents=True, exist_ok=True)
    base, text, validation_text, identities = checked_oof(ctx, args.oof_source)
    x = correction_frame(supplemental(ctx, "training"), base, text)
    vstate = supplemental(ctx, "validation").select_dtypes(include=["number"])
    manifest = dict(fold=args.fold, origin=ctx["phase"]["origin"], recipes=RECIPES,
        oof_source_manifest_sha256=digest(args.oof_source / "manifest.json"),
        oof_source_completed_result_sha256=completed_oof_source(args.oof_source),
        oof_source_data=identities, control_manifest_sha256=digest(SUPPORT / "manifest.json"),
        control_selection_sha256=digest(SUPPORT / "selection.json"),
        frozen_training_support_sha256=digest(SUPPORT / f"fold{args.fold}/support.json"),
        graph_selection_sha256=digest(TASK / "selection/results.json"),
        phase_manifest_sha256=digest(ctx["snapshots"] / f"fold{args.fold}/manifest.json"),
        seed0_oof_used_for_all_correction_seeds=True, maximum_columns=64,
        maximum_pairs=100_000, iterations=240, depth=3, thread_count=4,
        clipped_log_odds_correction=[-1., 1.], correction_affects="strong supplemental half only",
        full_unsampled_training_cohorts=list(map(str, ctx["cuts"])),
        menu_allowance=MENU_ALLOWANCE, validation_only=True,
        sources={name: digest(name) for name in (
            __file__, "scripts/frozen_offset_ranker.py", "scripts/engagement_local_comparison.py",
            "scripts/benchmark_reliability_blending.py", "scripts/fast_matched_bootstrap.py",
            "scripts/bounded_pairwise_readout.py", "kurversc/temporal_policy.py")})
    freeze(folder / "manifest.json", manifest)
    fit_args = dict(entities=ctx["train"].__event_entity.to_numpy(),
        timestamps=ctx["train"].__event_cutoff.to_numpy(), origin=ctx["phase"]["origin"],
        horizon=ctx["metadata"]["label_horizon"])
    truth = ctx["held"][ctx["metadata"]["target"]].to_numpy()
    records = []
    for seed in range(3):
        path = folder / f"seed{seed}.json"
        if path.exists():
            records.append(json.loads(path.read_text()))
            continue
        control, legacy = protected(ctx, seed)
        head = 2 * control - legacy
        vx = correction_frame(vstate, head, validation_text)
        print(json.dumps(dict(phase="fit_offset_ranker", fold=args.fold, seed=seed,
            rows=len(x), payload_columns=len(x.columns))), flush=True)
        model = FrozenOffsetRanker(seed=seed).fit(x, ctx["train"][ctx["metadata"]["target"]], base, **fit_args)
        arrays = dict(truth=truth, groups=ctx["held"].__event_entity.to_numpy(),
            forecast_cutoffs=pd.to_datetime(ctx["held"].__event_cutoff).to_numpy(), control=control)
        for name, strength in RECIPES.items():
            corrected = model.predict_values(vx, head, strength=strength)
            arrays[name] = replace_strong_half(control, legacy, corrected)
            restored = pickle.loads(pickle.dumps(model))
            np.testing.assert_array_equal(corrected, restored.predict_values(vx, head, strength=strength))
            del restored
        repeated = False
        if args.fold == 2 and seed == 0:
            other = FrozenOffsetRanker(seed=seed).fit(x, ctx["train"][ctx["metadata"]["target"]], base, **fit_args)
            for name, strength in RECIPES.items():
                again = replace_strong_half(control, legacy, other.predict_values(vx, head, strength=strength))
                np.testing.assert_allclose(arrays[name], again, rtol=0, atol=1e-9)
            repeated = True
            del other
        scores = {name: _score(truth, arrays[name], "classification") for name in ("control", *RECIPES)}
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        record = dict(fold=args.fold, seed=seed, origin=ctx["phase"]["origin"], scores=scores,
            gains={name: score - scores["control"] for name, score in scores.items() if name != "control"},
            audit=model.audit, independent_fit_repeated=repeated, serialization_repeated=True,
            manifest_sha256=digest(folder / "manifest.json"), vector_sha256=digest(path.with_suffix(".npz")))
        dump(path, record)
        records.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != "audit"}), flush=True)
        del model, vx
        gc.collect()
    dump(folder / "results.json", dict(records=records, elapsed_seconds=time.monotonic() - began,
        official_test_accessed=False, default_changed=False, manifest_sha256=digest(folder / "manifest.json")))


def finalize(args):
    records = []
    for fold in range(3):
        folder = args.output / f"fold{fold}"
        manifest = folder / "manifest.json"
        for seed in range(3):
            path = folder / f"seed{seed}.json"
            report = json.loads(path.read_text())
            if (report["fold"], report["seed"]) != (fold, seed) or report["manifest_sha256"] != digest(manifest):
                raise ValueError("Complete unchanged keyed evidence required")
            if report["vector_sha256"] != digest(path.with_suffix(".npz")):
                raise ValueError("Frozen candidate predictions changed")
            records.append(report)
    if not next(r for r in records if r["fold"] == 2 and r["seed"] == 0)["independent_fit_repeated"]:
        raise ValueError("Independent newest-origin fit reproduction required")
    gates = {}
    for name in RECIPES:
        if not all(r["gains"][name] > 0 for r in records):
            gates[name] = dict(selected="incumbent", reason="failed_required_all_case_positive_gate")
            continue
        evidence = []
        for r in records:
            with np.load(args.output / f"fold{r['fold']}/seed{r['seed']}.npz") as a:
                for policy, key in (("incumbent", "control"), (name, name)):
                    evidence.append(dict(origin=r["origin"], seed=r["seed"], policy=policy,
                        truth=a["truth"].copy(), groups=a["groups"].copy(), prediction=a[key].copy()))
        gates[name] = accelerated_select_temporal_policy(evidence, task="classification",
            minimum_origins=3, minimum_seeds=3, deployment_seed=0, bootstrap_repeats=1600,
            alpha=.05 / MENU_ALLOWANCE)
    admitted = [name for name, gate in gates.items() if gate["selected"] == name]
    latest = next(r for r in records if r["fold"] == 2 and r["seed"] == 0)
    selected = max(admitted, key=lambda name: latest["scores"][name]) if admitted else "incumbent"
    dump(args.output / "results.json", dict(decision=dict(selected=selected), gates=gates,
        records=records, menu_allowance=MENU_ALLOWANCE, official_test_accessed=False,
        default_changed=False, scope="Isolated validation-only frozen-model ranking correction",
        next_step="Only an admitted, independently reproduced recipe may be locked for official confirmation"))
    print(json.dumps(dict(phase="completed", selected=selected, latest_scores=latest["scores"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        finalize(args)
    elif args.fold is not None:
        fit_fold(args)
    else:
        parser.error("Supply exactly one fold or finalize")


if __name__ == "__main__":
    main()
