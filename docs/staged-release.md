# Complete staged pipeline: KurveRSC 0.2.0

This release packages the complete staged prediction procedure separately from
the lower-level `fit` / `predict` interfaces. The frozen fresh campaign has
**21/21 tasks completed**, with no failed tasks or historical score
substitutions. See the [final results](kurversc-staged-results.md) and the
[technical report](kurversc-staged-technical-report.md).

## Installation and entry point

The review branch uses KurveRSC **0.2.0** and GraphReduce **1.10.18**. Exact
companion Git commits are pinned in the RelArena PR until the corresponding
versions are published. An older PyPI artifact sharing an experimental version
does not identify the qualified code. Build both companion repositories with
`python -m build`, install the GraphReduce wheel, then the KurveRSC wheel with
its `staged` extra. RelArena's locked workspace supplies the exact review commits.

The staged extra pins NumPy 1.26.4, pandas 2.3.3, DuckDB 1.2.2, CatBoost 1.2.10,
scikit-learn 1.6.1, PyArrow 20.0.0, RelBench 2.1.2, PyTorch 2.13.0 and
Transformers 4.57.6. Use the CPU PyTorch index for CPU-only installation.
The qualified Linux runtime uses Python 3.11.17. Use an isolated environment;
in particular the qualified pandas version is 2.3.3, not pandas 3.

```python
import kurversc

# A TaskContext, or an exported context.json, supplied by the caller.
result = kurversc.run_staged_task(
    context, output="artifacts/task", task_hours=23.75,
    minimum_available_memory_gib=8,
)
predictions = result.predictions
```

```bash
kurversc-staged --context input/context.json --output artifacts/task --hours 23.75
```

`TaskContext.from_tables` declares keys, event/availability clocks, observed
labels and target-free forecast requests for an unregistered schema, including
string/composite keys. `TaskContext.from_protocol_splits` preserves caller-owned
censored databases and masked requests. The caller owns held-out scoring.

## Procedure

- Bounded schema-derived 10K/50K graph screening and full-source finalist
  evaluation retain alternative depth/family/history/topology representations.
- Full-fidelity comparisons use the production fitting lifecycle and freeze
  the selected feature-operation plan for replay. Base CDF sketches are enabled
  in this frozen recipe; they are not an exhaustive CDF on/off search dimension.
- Temporal support and bounded optional readouts are selected using eligible
  retrospective validation origins and fixed seeds 0/1/2. Deployment seed is 0.
- An original absolute deadline below 24 hours includes preparation, search,
  training and prediction; resuming does not reset it. Resource stops preserve
  checkpoints and raise, rather than returning a partial prediction as complete.
- Source, feature, model-input and runtime identities remain checked. Retaining
  an incumbent is a selection safeguard, not a universal no-regression guarantee.

The wheel contains **46 native modules and 108 transitive stage modules** under
`kurversc._staged`. It does not need the independent research checkout, model
checkpoints or benchmark data. The sdist contains the source closure needed to
rebuild the wheel. Scripts retain their source names for a direct audit against
the qualified execution; packaging relocates imports and paths only.

## Execution qualification

The source payload matches the score-producing frozen 0.1.1/GraphReduce 1.10.17
candidate, except release-version metadata. That private candidate is identified
by wheel SHA-256 values `04d066bb09185bed1bc687e1bcf30bfcc0349afdcc5800540f391e1e6a890692`
and `dbf00d70a340e3707f6315ffb350caf746a040b738a0c1b60d60120c913450fc`.
Version changes create new artifact/cache identities; old identities are not
forged to reuse fitted state across releases.

The qualified changes reduce unnecessary work: disk-backed historical capture,
bounded normalization/concatenation, detached routing headers, narrow text and
exposure projections, equivalent cache comparisons and scoped wide-aggregate
SQL thread use. Feature budgets, training evidence and validation rules are
preserved. The earlier cache-consistency/resource failures remain documented
historically; fresh completed tasks use the repaired installed-wheel runtime.

The optional semantic encoder is `sentence-transformers/all-MiniLM-L6-v2`, pinned
to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It uses fixed weights,
no remote model code and safetensors. Its external pretraining includes
StackExchange data; this remains an explicit benchmark-review disclosure.

See the [resource report](kurversc-staged-resources.md) for every task's runtime,
memory and coverage, and for parallel admission on larger machines.
