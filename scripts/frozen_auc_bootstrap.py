"""Research-only cached ROC layouts for the unchanged paired confidence gate.

Scores are fixed during a bootstrap: sort them once, then recompute cumulative
class weights. RNG, entity clustering, resample counts, score direction and
policy logic are unchanged. Reject fractional weights; the current gate uses
integer resampling multiplicities. No production monkeypatch/default changes.
"""

import ast
import inspect

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from kurversc.temporal_policy import select_temporal_policy
from scripts.fast_matched_bootstrap import EntityBlocks, sample_positions


class FrozenAucLayout:
    def __init__(self, truth, prediction):
        self.truth, self.prediction = np.asarray(truth), np.asarray(prediction)
        if (self.truth.shape != self.prediction.shape or self.truth.ndim != 1
                or set(np.unique(self.truth)) != {0, 1} or not np.isfinite(self.prediction).all()):
            raise ValueError("Aligned finite binary ROC inputs required")
        self.order = np.argsort(self.prediction, kind="mergesort")[::-1]
        self.labels = self.truth[self.order].astype("float64")
        score = self.prediction[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]

    def score(self, weights):
        weights = np.asarray(weights)
        if (weights.shape != self.truth.shape or not np.isfinite(weights).all()
                or np.any(weights < 0) or np.any(weights != np.floor(weights))):
            raise ValueError("Aligned nonnegative integer bootstrap multiplicities required")
        weight = weights[self.order].astype("float64")
        tp = np.cumsum(weight * self.labels, dtype="float64")[self.ends]
        fp = np.cumsum(weight, dtype="float64")[self.ends] - tp
        keep = np.diff(np.r_[0., tp + fp]) > 0
        tp, fp = tp[keep], fp[keep]
        if not len(tp) or min(tp[-1], fp[-1]) <= 0:
            raise ValueError("Both weighted target classes required")
        if len(fp) > 2:
            # Same redundant-point elimination as sklearn.metrics.roc_curve.
            keep = np.r_[True, np.logical_or(np.diff(fp, 2), np.diff(tp, 2)), True]
            tp, fp = tp[keep], fp[keep]
        tpr, fpr = np.r_[0., tp] / tp[-1], np.r_[0., fp] / fp[-1]
        return float(np.trapz(tpr, fpr))


def temporal_lower_bound(pairs, *, task, alpha, repeats):
    codes, unique = pd.factorize(np.concatenate([p[3] for p in pairs]), sort=True)
    if len(unique) < 2:
        return None
    ordered, offset = [], 0
    for truth, control, candidate, _ in pairs:
        ids = codes[offset:offset + len(truth)]
        offset += len(truth)
        positions = np.lexsort((candidate, control, truth, ids))
        y, a, b, ids = truth[positions], control[positions], candidate[positions], ids[positions]
        if task == "classification":
            ordered.append((y, FrozenAucLayout(y, a), FrozenAucLayout(y, b), ids))
        else:
            ordered.append((y, a, b, ids))
    fold_weights = np.full(len(pairs), .5 / max(1, len(pairs) - 1))
    fold_weights[-1] = .5
    rng, gains = np.random.default_rng(43), []
    for _ in range(repeats):
        multipliers = rng.poisson(1., len(unique))
        differences = []
        for y, a, b, ids in ordered:
            weight = multipliers[ids]
            if not weight.sum() or (task == "classification" and np.unique(y[weight > 0]).size < 2):
                break
            if task == "classification":
                differences.append(b.score(weight) - a.score(weight))
            else:
                differences.append(mean_absolute_error(y, a, sample_weight=weight)
                                   - mean_absolute_error(y, b, sample_weight=weight))
        if len(differences) == len(pairs):
            gains.append(float(fold_weights @ differences))
    return float(np.quantile(gains, alpha)) if gains else None


def frozen_auc_select_temporal_policy(evidence, **kwargs):
    """Fail-closed AST translation of the installed, otherwise unchanged rule."""
    source = inspect.getsource(select_temporal_policy)
    replacements = {
        "[np.flatnonzero(codes == index) for index in range(len(unique))]":"EntityBlocks(codes, len(unique))",
        "np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])":
            "sample_positions(blocks, rng.integers(0, len(blocks), len(blocks)))",
        "_score(y[pos], control[pos], task)":"resampled_score(y, control, pos, task)",
        "_score(y[pos], candidate[pos], task)":"resampled_score(y, candidate, pos, task)",
    }
    fingerprints = {key:ast.dump(ast.parse(key, mode="eval").body) for key in replacements}
    counts = dict.fromkeys(replacements, 0)

    class Rewrite(ast.NodeTransformer):
        def generic_visit(self, node):
            fingerprint = ast.dump(node)
            for key, expected in fingerprints.items():
                if fingerprint == expected:
                    counts[key] += 1
                    return ast.copy_location(ast.parse(replacements[key], mode="eval").body, node)
            return super().generic_visit(node)

    layouts = {}

    def resampled_score(y, prediction, positions, task):
        if task != "classification":
            return select_temporal_policy.__globals__["_score"](y[positions], prediction[positions], task)
        key = (id(y), id(prediction))
        if key not in layouts:
            # Store arrays in the layout, preventing object-ID reuse.
            layouts[key] = FrozenAucLayout(y, prediction)
        return layouts[key].score(np.bincount(positions, minlength=len(y)))

    tree = ast.fix_missing_locations(Rewrite().visit(ast.parse(source)))
    if set(counts.values()) != {1}:
        raise ValueError("Installed selector changed; cached-layout acceleration not applicable")
    namespace = dict(select_temporal_policy.__globals__, EntityBlocks=EntityBlocks,
        sample_positions=sample_positions, resampled_score=resampled_score,
        _temporal_gain_lower_bound=temporal_lower_bound)
    exec(compile(tree, "<reviewed-frozen-auc-bootstrap>", "exec"), namespace)
    return namespace["select_temporal_policy"](evidence, **kwargs)
