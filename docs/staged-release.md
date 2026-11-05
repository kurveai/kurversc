# Generic staged pipeline — published 0.2.0

The runnable submission has been narrowed following maintainer review. The
previous October 27 results and resource tables describe a superseded pipeline;
they do not establish performance or Elo for this revision. RelArena maintainers
have now reproduced all 21 tasks with the public packages and merged the
system. The [reproduced table](kurversc-reproduced-results.md) is the reference:
1913.4 Elo, second overall and first among systems in the November 5 combined
16-method snapshot, with 11/21 wins against the latest TabPFN-Rel client.

## Retained procedure

1. Schema-derived graph search with successive fidelity, CDF/base representations,
   peer context, and bounded graph/path/feature budgets.
2. Full-fidelity graph reranking and the bounded production-faithful finalist
   comparison for both binary classification and regression using the native
   fit/refit lifecycle. Its shared 1% relative-loss cap uses `1 - AUC` or MAE.
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

## November 4 finalist-check revision

The S2 check now applies to both supported task types with the same 1% relative
validation-loss cap, using MAE or `1 - AUC`. Native metric uncertainty, matching
origins/seeds and unanimous seed selection are shared. Thirty MAE comparisons
exactly reproduce the reviewed decisions and evidence; regression behavior is
unchanged. No research heads or downstream learner settings were added.

Current qualification: **727 source tests passed, one skipped; 19 RelArena
adapter/discovery tests passed**. Installed-wheel classification and regression
runs each completed the full registered procedure on a fresh unregistered
48-request schema with actual core split types and no hidden targets. Source
and wheel Python payloads match. See [the current qualification receipt](generic-staged-validation.json).
The maintainer's completed 21-task reproduction is the reference result;
see the [published scores and provenance](kurversc-reproduced-results.md).

## Prior qualification and local pilot

The November 1 reproduction candidate retains the narrowed generic recipe and
adds the [source-graph replay correction](generic-staged-review.md#source-graph-replay-correction).
Its preceding qualification is recorded in [the archived receipt](archive/generic-staged-validation-20261101.json):
716 source tests pass with one skip, and 14 current RelArena adapter/discovery
checks pass. Installed-wheel classification and regression smokes each complete
the entire procedure on fresh unregistered 48-request schemas through the current
registered class and actual core split types. No hidden targets are supplied.
Wheel/sdist contents and source parity pass. These checks use existing numerical
dependencies and the exact pinned GraphReduce source; they do not establish cold
installation or a full-panel reproduction.

The new wheel contains 47 native and 32 stage modules. The extra native module
stores and verifies source bindings; it does not add a prediction head. Earlier
wheel hashes and predictive-source parity describe the preceding pilot revision.

The local frozen run is `results/generic-staged-local-20261029`. Six representative
tasks are admitted: F1 driver-dnf/driver-position, Avito user-clicks, Trial
study-adverse/study-outcome, and Stack user-engagement. The other fifteen tasks
are held. Each task records a fresh score, selection evidence, runtime and peak
process-tree RSS. The machine retains an 8 GiB RAM reserve.

The November 5 maintainer reproduction supersedes this earlier pilot as the
current performance reference. The dated qualification receipts remain
pre-release records, not retroactively rewritten result claims.

See the [review changes](generic-staged-review.md), [current pilot table](generic-staged-pilot.md),
and [checksummed pilot admission bundle](generic-staged-pilot-admissions.tar.gz).

The six-task pilot has completed. The shared gate admitted the zero-aware head
for study-adverse and the generic text head for study-outcome; the other four
retained the incumbent. User-engagement scores lower after removing the disputed
correction stack. All score changes appear in the pilot table.
