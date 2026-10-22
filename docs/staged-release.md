# Complete staged release candidate

`0.2.0rc1` packages the full staged prediction procedure separately from the
existing lower-level `fit` / `predict` APIs. This candidate is under review;
it does not claim a completed 21-task reproduction or an Elo ranking.

## Installation and public entry point

Build both this repository and the companion GraphReduce `1.10.18rc1` branch:

```bash
python -m build
pip install 'dist/kurversc-0.2.0rc1-py3-none-any.whl[staged]'
kurversc-staged --help
```

Install the companion GraphReduce candidate first when it has not yet been
published to the package index. Do not substitute an older artifact carrying
the same nominal version as an experiment.

The `staged` extra pins the score-producing numerical/learner versions:
NumPy 1.26.4, pandas 2.3.3, DuckDB 1.2.2, CatBoost 1.2.10, scikit-learn 1.6.1,
PyArrow 20.0.0, RelBench 2.1.2, PyTorch 2.13.0 and Transformers 4.57.6.
Use a CPU PyTorch index for CPU-only reproduction. The optional semantic model
is `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; code disables remote model code and
loads safetensors. Its external pretraining remains subject to benchmark review.

Supply a `TaskContext` built from explicit source tables and declared keys,
event/availability clocks, observed training and validation labels, and
target-free forecast requests. `TaskContext.from_protocol_splits` accepts
already-censored caller-owned splits without reloading a benchmark registry.
`TaskContext.from_tables` supports new schemas and composite or string keys.

```python
import kurversc

# context is a caller-constructed TaskContext, or its exported context.json.
result = kurversc.run_staged_task(
    context, output="artifacts/task", task_hours=23.75,
    minimum_available_memory_gib=8,
)
predictions = result.predictions  # original request keys/order + prediction
```

The equivalent command is:

```bash
kurversc-staged --context input/context.json --output artifacts/task --hours 23.75
```

The output includes checksummed context/source manifests, stage decisions,
execution logs, and `predictions.parquet`. Final holdout scoring belongs to the
caller. No hidden test answers are part of the task bundle.

## Frozen procedure

- Schema-derived graph search screens a bounded panel at 10K rows, promotes
  contenders through 50K and full-frame evidence, and retains higher-order
  depth/family, history, annotation and bounded cyclic alternatives.
- Full-fidelity finalists use the production frame-ensemble lifecycle. The
  selected GraphReduce operation plan is frozen, rather than rediscovered
  during the reporting refit.
- CDF/raw representations, state/context and the other applicable capabilities
  compete within the existing graph search. Temporal support and optional
  readouts are selected only from available retrospective validation evidence.
- All available validation origins and fixed seeds 0/1/2 must satisfy the
  existing admission rules. Deployment uses seed 0; other deployment seeds are
  rejected explicitly rather than silently ignored.
- The original incumbent remains available. This is a selection safeguard,
  **not a guarantee of non-regression on future held-out data**.
- One original absolute deadline below 24 hours includes preparation, search,
  training and prediction. Resuming an output directory does not reset it.
- Resource stops preserve checkpoints and raise; partial results are never
  represented as complete predictions. Checkpoints require matching inputs,
  source/runtime identities and configuration.

The wheel contains the 106-module transitive stage closure under the private
`kurversc._staged` namespace, not the independent hypothesis scripts, trained
models, benchmark data, or experiment result archive. The source distribution
includes the source modules required to rebuild that wheel.

## Qualification and known blockers

Before release version stamping, all 46 public package source files and all
106 relocated stage modules matched the previously qualified candidate wheel
byte-for-byte (SHA-256
`b0c94fe8782886902cccab8337f4586e9439ec45e6f7a8d675952e9421d046e0`).
The companion metadata-only GraphReduce wheel has SHA-256
`7023d60dfbe69b1ab64c338169d9bc4cef8740d81f6e6eb83b03cb26769e0ce4`.
Changing release metadata does not imply that old checkpoint identities may
be forged or forced into the new version.

The most recent campaign completed **12 of 21** official tasks. Nine remain
uncompleted; no previous scores are substituted into that count. Remaining
execution issues are:

1. Raw rematerialized-frame checks fail on low floating-point aggregate bits
   or list/category representation differences. Assertions remain enabled;
   canonical feature/input equivalence still needs repair and qualification.
2. A staged resource allocator sets DuckDB working memory to
   `min(128, floor(host_GiB / 4)) GiB`. On a roughly 496-GiB host this becomes
   123 GiB and can fail while substantial host RAM is still free. This is a
   DuckDB working-memory policy, not a process RSS limit; consistent explicit
   resource propagation remains a blocker.
3. The widest H&M churn candidate reached roughly 414 GiB of cgroup memory on
   that host and was stopped by the campaign's host reserve. More RAM or a
   separately validated execution repair is needed; the graph/model evidence
   has not been weakened to fit it.

The release must not be marked fully qualified until the unfinished tasks,
end-to-end runtime limit, per-task peak memory, and current-harness reproduction
have been verified. Historical or projected pairwise wins are not fresh results.
