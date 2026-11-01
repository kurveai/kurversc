"""Compatibility entry point for the generic, unified candidate evaluator.

The former exposure-specific implementation is preserved only in the dated
research snapshot; it is not part of the runnable submission.
"""
from scripts.generic_staged_helpers import (
    dump, prepared, routing_headers, values, incumbent_evidence, materialize_outer,
)
from scripts.benchmark_generic_readouts import validate, official, main, fit_signal

fit_candidate = fit_signal

if __name__ == "__main__":
    main()
