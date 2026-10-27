"""Repeat the streamed count experiment with conditional Poisson medians.

The negative-binomial marginal dispersion produced an all-zero median, exactly
the cheap shrinkage control. This one scalar-loss ablation avoids estimating
conditional dispersion from unconditioned heterogeneous target variance.
The previous runner and all completed evidence remain unmodified.
"""

import argparse
from pathlib import Path

from scripts import benchmark_streamed_count_likelihood as runner
from scripts.poisson_count_median import PoissonCountMedian
from scripts.run_gap_followup_queue import digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    old_model, old_freeze = runner.CountLikelihoodRegressor, runner.freeze

    def freeze(path, identity):
        identity.update(profile="conditional_poisson_median", runner_unchanged=True)
        identity["sources"].update({n: digest(n) for n in (__file__, "scripts/poisson_count_median.py")})
        return old_freeze(path, identity)

    runner.CountLikelihoodRegressor, runner.freeze = PoissonCountMedian, freeze
    try:
        runner.run(args)
    finally:
        runner.CountLikelihoodRegressor, runner.freeze = old_model, old_freeze
