# KurvePlanner training scaffold

KurvePlanner learns to rank safe GraphReduce plans from metadata supplied by an
explicit profiler. It does not crawl raw tables. In the Kurve product, Kurve is
the profiler; the open-source planner consumes its versioned aggregate output.

```text
Kurve profiler -> DatabaseProfile + TaskProfile
KurveRSC search -> candidate outcomes (teacher)
                -> PlannerEpisode JSONL
                -> database-grouped CatBoost ranker
                -> KurvePlanner artifact
```

The first model is deliberately a small CatBoost meta-ranker. Its purpose is to
validate the training signal and split protocol before replacing its aggregate
schema encoding with a graph transformer.

## Profile contract

`DatabaseProfile` is a structured and JSON-serializable schema snapshot. Its
`as_of` timestamp states the latest data the profiler was allowed to inspect.
It contains:

- table and value-free column profiles;
- relationships, cardinalities, coverage, fan-out and join expansion;
- event counts, durations and inter-event intervals;
- sampling and profile provenance; and
- optional scalar extension fields.

The event-frequency field names match the values Kurve currently persists on
compute-graph edges: `avg_num_events`, `median_num_events`,
`avg_duration_seconds`, `median_duration_seconds`,
`avg_time_between_updates_seconds`, and
`median_time_between_updates_seconds`.

Existing persisted Kurve payloads may also contain edge identifiers and the SQL
used to calculate the aggregates. Use
`EventFrequencyProfile.from_kurve_payload(payload)` to retain only the public,
value-free aggregate fields; SQL and product identifiers are not copied.

`TaskProfile` describes the label head and the support set visible at planning
time. `RuntimeProfile` describes the downstream estimator and compute limits.
No raw rows or actual categorical values are stored in a planner trace.

```python
import kurversc

profile = kurversc.DatabaseProfile(
    database_id="rel-amazon",
    as_of="2025-01-01T00:00:00+00:00",
    extractor="kurve",
    tables=(...),
    relationships=(...),
)

task = kurversc.TaskProfile(
    task_id="user-churn",
    task_type="classification",
    metric="roc_auc",
    root_table="users",
    target_id="churn",
    support_rows=50_000,
    horizon_seconds=90 * 86_400,
    positive_fraction=0.14,
    probes={"recent_event_count_auc": 0.61},
)

runtime = kurversc.RuntimeProfile(
    downstream_model="catboost",
    execution_backend="duckdb",
    cpu_count=16,
    memory_bytes=64_000_000_000,
    max_features=8_000,
)
```

Profiling must be explicit. A future open-source reference profiler may produce
the same object, but `plan` and `predict` must not perform an invisible database
scan when this object is absent.

## Collecting full-fidelity teacher traces

Use `collect_planner_search` for planner data generation. It is deliberately
different from the latency-conscious production search:

- source tables are not row-sampled;
- three training anchor frames are selected by default;
- the forward beam is disabled;
- 24 capability-valid configurations are chosen using a balanced design;
- depths 1 through 4 are represented, with the deepest candidates reserved
  for the compact base, temporal and sequence families;
- the default design spans 2, 4 and 8 source-column budgets;
- every selected candidate is evaluated at the same fidelity; and
- execution returns after teacher evaluation instead of fitting a redundant
  production model.

The required family/depth/budget coverage is stable, while higher-order family
combinations rotate deterministically from the database ID, task ID and support
cutoff. Across many episodes this explores substantially more of the lattice
without making every individual search exhaustive.

For the public RelBench training corpus, the repository includes an explicit
offline profiler and resumable collector. It derives only cutoff-safe aggregate
statistics from the public frames; it is not invoked by `fit`, `plan`, or
`predict`:

```bash
python examples/relbench_planner_collection.py rel-f1 \
  --output planner-traces \
  --candidates 24 \
  --training-frames 3 \
  --max-depth 4 \
  --schema-depth 4
```

Omit task names to collect every binary-classification and regression task for
the database, or list task names after the dataset. Each task is written to an
independent JSONL file, and existing files are skipped unless `--overwrite` is
passed. This makes database-level jobs safe to resume and avoids concurrent
writes to one corpus file.

To resume all seven original RelBench databases sequentially and train the
initial CatBoost ranker after collection finishes:

```bash
python examples/run_relbench_planner_corpus.py \
  --output planner-traces \
  --artifact artifacts/kurveplanner-catboost-v0
```

The queue skips completed task files. By default it trains and evaluates only
the validation partition; `--evaluate-test` is deliberately opt-in so the two
held-out databases remain untouched while the planner design is still changing.

```python
collected = kurversc.collect_planner_search(
    problem.parent_node,
    problem.label_node,
    tables=problem.tables,
    relationships=problem.relationships,
    task=problem.fit_kwargs()["task"],
    database_profile=profile,
    task_profile=task,
    runtime_profile=runtime,
    trace_path="planner-traces/rel-amazon-user-churn.jsonl",
    episode_key="support-cutoff-2025-01-01",
)

print(collected.fit_result.results)
print(collected.episodes[0].observations[0].cutoff_scores)
```

Supply `planner_collection_candidates=None` to evaluate the complete supported
lattice, or provide explicit `graph_configs` to evaluate an exact experimental
design. Complete-lattice collection is substantially more expensive.

Each completed candidate retains both its aggregate validation objective and
the score at each available validation cutoff. The initial ranker uses the
aggregate objective; the individual cutoff scores support later stability and
distribution-shift objectives.

## Importing ordinary search traces

Run an ordinary KurveRSC search, then convert its `FitResult`. Screening,
confirmation and reranking are stored as separate ranking episodes because
their validation scores come from different data fidelities. Higher-fidelity
episodes receive larger training weights.

```python
result = kurversc.fit(...)

episodes = kurversc.episodes_from_fit_result(
    result,
    database_profile=profile,
    task_profile=task,
    runtime_profile=runtime,
    episode_key="support-cutoff-2025-01-01",
)

kurversc.write_episodes_jsonl(
    "planner-traces/rel-amazon-user-churn.jsonl",
    episodes,
)
```

Each candidate record contains the proposed plan, objective score, uncertainty,
preflight feature estimate, observed cost and failure type. Only information
available before execution becomes a model feature. Observed validation scores,
materialized feature counts and runtimes remain supervision targets.

Each episode also preserves KurveRSC's source-column funnel audit in
`extras.column_selection`. It contains the table-local utility rank, actual
selection rank, eligible feature families, per-family ranks, budget tiers and
selection strategy for every audited column. The initial CatBoost plan ranker
deliberately ignores this nested, schema-specific payload. It is the supervision
contract for a future graph-transformer pointer head, which will score the
columns present in each input node rather than learn a fixed vocabulary of
table or column names.

Candidates skipped only because they fell outside the current forward beam are
marked `censored`, not assigned a losing score. Measured execution failures and
plans rejected by an explicit resource limit remain supervised infeasible
examples. This prevents the planner from merely reproducing the teacher's
existing search bias.

The current `GraphConfig` traces vary families, depth, annotations and feature
budgets. Training the planner to choose `ts_periods`, categorical policies or
schema-revisit policies requires deliberately varying those fields in teacher
searches and including them in the candidate plan.

## Database-grouped training

Install the package, then train from one or more JSONL files:

```bash
kurveplanner-train planner-traces/*.jsonl \
  --output artifacts/kurveplanner-v0 \
  --train-fraction 0.8 \
  --test-size 2
```

The automatic final-test selection requires those two databases, together, to
contain both classification and regression tasks. Override
`--required-test-task-types` only for a deliberately single-task-type corpus.
Add `--dry-run` to validate the trace corpus and write `dataset_summary.json`
and `split.json` without fitting a model.

For a reproducible benchmark, name the complete validation and final-test
databases explicitly:

```bash
kurveplanner-train planner-traces/*.jsonl \
  --output artifacts/kurveplanner-v0 \
  --validation-databases rel-f1 rel-stack \
  --test-databases rel-amazon rel-event
```

All tasks, heads, cutoffs and candidate results from a database remain in one
partition. Database identifiers, task identifiers, target names, table names
and column names are retained for audit but excluded from the baseline model's
features.

By default the command evaluates only validation databases. This prevents the
two final databases from becoming another tuning set. Once model and training
choices have been frozen, run once with `--evaluate-test`:

```bash
kurveplanner-train planner-traces/*.jsonl \
  --output artifacts/kurveplanner-v0-final \
  --validation-databases rel-f1 rel-stack \
  --test-databases rel-amazon rel-event \
  --evaluate-test
```

The artifact directory contains:

```text
ranker.cbm
manifest.json
split.json
dataset_summary.json
validation_metrics.json
test_metrics.json       # only with --evaluate-test
```

Evaluation reports mean and median normalized regret, the fraction selected
within 1% of the searched response surface, top-three oracle recall, and failed
selection rate.

## Remote training checklist

Before sending a run to the remote server:

1. Export a cutoff-safe `DatabaseProfile` for every task episode.
2. Retain all evaluated candidate outcomes, including failures—not only the
   winning configuration.
3. Include classification and regression heads from every training database.
4. Keep all episodes from the two final databases out of training and model
   selection.
5. Sync the JSONL traces and repository, then run `kurveplanner-train`.
6. Inspect validation regret before authorizing the single final-test run.

This baseline proves whether the metadata contains enough signal to amortize
search. The next model can consume the same JSONL records while replacing
aggregate schema features with table, relationship, task and candidate tokens.
