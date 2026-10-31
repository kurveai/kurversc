# Maintainer-review revision

The submitted execution path is now `run_staged_task` →
`run_portable_staged` → `run_admissible_staged`. Its profile is
`kurversc-generic-staged-v1`. This replaces the earlier recovery-stage cascade.

| Review item | Revision |
|---|---|
| Exposure-role reconstruction | Removed from the runnable recipe and packaged dependency closure. |
| Recurrence/semantic teacher and offset ranker | Removed, including their pretrained encoder dependencies. |
| Production-sequence and bounded-view recovery | Removed. |
| Conditional-count recovery | Removed. Integer-domain projection below is rounding/clipping, not a Poisson head. |
| Different gates and cascade eligibility | One `select_temporal_policy` call compares every applicable optional head against the same S3 incumbent. Earlier admissions do not control eligibility. |
| Campaign test-win gates | Absent from the portable recipe and its packaged dependency closure. External scoring happens after predictions and decisions are saved. |

Core S1–S3 remains: schema-derived graph search, production-faithful finalist
comparison, and temporal training-policy selection. The retained optional menu
contains the generic components identified in the review:

| Component | Inputs and global bounds | Validation candidates |
|---|---|---|
| Parent text | Mature training strings; mean nonempty length ≥32 and ≥20 distinct values under one field-selection definition; at most 8 fields, 2,048 characters per field, 16,384 TF-IDF terms and 100,000 training rows. Logistic or ridge head according to task type. | 0.25 or 0.5 blend. |
| Training-frame ensembles | Three or five eligible initial cutoffs; the same refit lifecycle in validation and final prediction. Unavailable when historical support is insufficient. | Three or five frames. |
| Target-domain projection | Training-only verification of nonnegative or nonnegative integer targets. | Clipping or integer projection. |
| Zero-aware readout | Nonnegative regression targets with at least 20 zero and 20 positive examples; occurrence classifier plus positive log-target quantiles at 0.05, 0.25, 0.5, 0.75 and 0.95. | 0.125, 0.5 or 1 blend. |
| Object breadth | Declared foreign keys to a root and keyed objects, with timed events and ≥64 training-visible objects. At most two paths, 42 features and 730 days of history. No inferred click/position/prior roles. | Full head or 0.5 blend. |

These are bounded global defaults, not per-task settings or claims of optimal
hyperparameters. The small blend grids regularize additional heads; vocabulary,
path and feature-cell limits bound cost. All optional payload learners share
the 500-million-cell capacity limit. Candidate manifests and hashed source definitions fix the settings before
candidate evaluation. The text head uses global logistic C=1 or ridge alpha=10;
these are regularized baselines, not per-database model choices. The text heuristic can also admit opaque strings; it
does not claim to recognize their semantic meaning and does not enable a
recurrence or teacher stack.

The shared gate uses deployment seed 0 and validation seeds 0/1/2, at least
two eligible origins, latest-origin positive gains for every seed, older-origin
safety caps of 0.005 AUC or 1% relative MAE, and a positive paired temporal
bootstrap lower bound (600 draws, alpha 0.05 with menu-size correction).
The recent-only interval is diagnostic for every candidate. Candidate
selection and support budgets are shared across schemas.

The wheel contains 32 stage modules rather than 108. The removed experiment implementations remain in the repository as historical
research source, but are excluded from the revised portable procedure and its
packaged dependency closure. Generic utilities can retain historical filenames;
they do not restore the removed prediction branches. The package-build tests check
their exclusion and prediction routing ignores their old admission files.

See [the current procedure](staged-release.md) for data boundaries and the
six-task pilot. The pilot uses externally scored, keyed predictions, with
strict source cutoffs. It is a revised author-run evaluation, not evidence of
an independent full current-harness reproduction or untouched test evaluation.
The prior 21-task scores and Elo are not assigned to this revision.

The completed pilot and its admission bundle report all six outcomes, including
regressions. These results do not erase the development history or establish
untouched-test generalization. This revision follows the narrower accepted
component set; it does not seek to retain the disputed components on the basis
of the earlier transfer experiments.
