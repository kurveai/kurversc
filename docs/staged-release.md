# Generic staged pipeline — October 29 revision

The runnable submission has been narrowed following maintainer review. The
previous October 27 results and resource tables describe a superseded pipeline;
they do not establish performance or Elo for this revision. Fresh evaluation is
in progress.

## Retained procedure

1. Schema-derived graph search with successive fidelity, CDF/base representations,
   peer context, and bounded graph/path/feature budgets.
2. Full-fidelity graph reranking and the bounded production-faithful finalist
   comparison using the native fit/refit lifecycle.
3. Historical temporal-training-policy selection on up to three origins and
   seeds 0, 1, and 2; deployment seed 0.
4. A single generic candidate menu: parent text heads, three/five-frame ensembles,
   target-domain projection, zero-aware distributional readout, and object breadth.
5. Validation-locked final fitting and keyed prediction replay.

The optional menu uses one implementation of `select_temporal_policy`, with the
same three-seed grid, recent-origin criterion, older-origin safety veto, paired
entity-cluster temporal uncertainty, and menu-size multiplicity correction.
Candidate eligibility depends on required inputs and declared capacity, independently
of earlier candidate admissions. Text identification has one shared definition.

The global defaults use 600 bootstrap draws and alpha 0.05; older-origin practical
loss limits are 0.005 AUC or 1% relative MAE. The admitted deployed seed must improve
on the most recent origin and all latest-origin seeds must agree. The temporal
lower bound must be positive. The separately reported recent-only interval is
diagnostic for every candidate. These are selection safeguards, not a guarantee
of improved future performance or a claim to correct all development overfitting.

Blend weights, quantile levels and model budgets are bounded global settings,
shared across schemas. They are recorded in candidate manifests before evaluation.
A fresh evaluation reports their actual effect; historical benchmark development
is not described as untouched test evaluation.

## Removed from the submitted runtime

- Exposure-role reconstruction and specialized exposure-quality transforms.
- Recurrence/semantic teacher and offset corrections.
- Conditional-count recovery.
- Production-sequence recovery.
- Bounded-view recovery.
- Benchmark test-win campaign gates.

Historical research snapshots are preserved. None of these removed stage modules
is in the current wheel's transitive execution closure. The wheel contains 32
stage modules. Its staged extra no longer requests PyTorch, Transformers or
Safetensors for the removed semantic encoder.

## Data boundaries

Explicit strict source tables use event time `< query cutoff` and `< phase cutoff`.
Independent availability clocks use `<= phase cutoff`. RelBench timed-table
adapters declare strict event boundaries. A legacy compatibility offset cannot
relax an explicit strict boundary. Filtering precedes annotation/materialization,
and phase exports are censored before provider discovery.

Forecast requests survive when a strict cutoff makes their root payload
unavailable; they receive missing features. Existing request keys/order are kept.
Training labels remain subject to horizon maturity and declared availability.
Held-out answers are absent from prediction contexts and are scored externally
only after selection and predictions have been saved.

## Entry point

```python
import kurversc

result = kurversc.run_staged_task(
    context,
    output="artifacts/task",
    task_hours=23.75,
    minimum_available_memory_gib=8,
)
predictions = result.predictions
```

`TaskContext.from_tables` accepts declared keys, clocks and observed labels for
an unregistered schema. `TaskContext.from_protocol_splits` accepts caller-owned
censored phases and masked prediction requests. Resumes retain the original
absolute task deadline; incomplete resource-stopped runs are not scored.

## Current validation and local pilot

The clean review checkout passes 710 tests with one skip. The current RelArena
adapter/discovery checks pass 19 tests. Installed-wheel classification and
regression smokes each complete the revised procedure on a fresh, unregistered
48-request schema through the current registered class and actual core split
types. No hidden targets are supplied. These checks use an existing dependency
environment; they are not a cold resolution or a full benchmark reproduction.
The wheel contains 46 native and 32 stage modules, verified against the frozen
pilot sources apart from release-version stamping and import/path relocation.

The local frozen run is `results/generic-staged-local-20261029`. Six representative
tasks are admitted: F1 driver-dnf/driver-position, Avito user-clicks, Trial
study-adverse/study-outcome, and Stack user-engagement. The other fifteen tasks
are held. Each task records a fresh score, selection evidence, runtime and peak
process-tree RSS. The machine retains an 8 GiB RAM reserve.

No replacement 21-task result or updated Elo is claimed until measured.

See the [review changes](generic-staged-review.md), [current pilot table](generic-staged-pilot.md),
and [checksummed pilot admission bundle](generic-staged-pilot-admissions.tar.gz).
