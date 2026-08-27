# KurveRSC

KurveRSC is a small validation-guided layer over
[GraphReduce](https://github.com/wesmadrigal/graphreduce). Its main API is one
function:

The [Kurve RSC white paper](docs/kurversc-white-paper.md)
([PDF](docs/kurversc-white-paper.pdf)) explains how graph configuration and
downstream learner performance participate in one validation loop. The
companion benchmark technical report contains the detailed system,
feature-family, and empirical treatment.

```bash
pip install -e "/path/to/kurve-rsc[relbench]"  # omit [relbench] for ordinary tables
```

```python
import kurversc

result = kurversc.fit(
    parent_node="customers.parquet",
    label_node="churn_labels.parquet",
    parent_key="customer_id",
    label_key="customer_id",
    target="churn",
    split_column="split",  # values: train / validation
)

print(result.best_config)         # highest validation ROC AUC or lowest MAE
print(result.recommended_config)  # simpler config when the gain is negligible
print(result.results)
```

Both node arguments accept a pandas DataFrame, CSV/Parquet path, or the name of
a table/view on a supplied DuckDB connection. For explicit metadata, use
`Table` and `Labels`:

```python
result = kurversc.fit(
    parent_node=kurversc.Table(
        "users", name="users", key="Id", date="CreationDate"
    ),
    label_node=kurversc.Labels(
        "user_labels",
        key="user_id",
        target="will_return",
        timestamp="timestamp",
        split="split",
    ),
    tables=[
        kurversc.Table(
            "posts", name="posts", key="Id", date="CreationDate"
        ),
        kurversc.Table(
            "comments", name="comments", key="Id", date="CreationDate"
        ),
    ],
    relationships=[
        kurversc.Relationship(
            parent="users",
            child="posts",
            parent_key="Id",
            child_key="OwnerUserId",
        ),
        kurversc.Relationship(
            parent="posts",
            child="comments",
            parent_key="Id",
            child_key="PostId",
        ),
    ],
    connection=duckdb_connection,
)
```

When the target is a future aggregation over one of the graph's event tables,
let GraphReduce generate it natively instead of supplying a materialized label
table:

```python
label_node=kurversc.GraphLabels(
    table="orders",
    field="id",
    operation="bool",
    period_days=365,
    train_cutoffs=("2023-01-01", "2024-01-01"),
    validation_cutoffs=("2025-01-01",),
    test_cutoffs=("2026-01-01",),
    target="will_order",
)
```

This executes GraphReduce's `prep_for_labels()` and automatic `do_labels`
aggregation at every cutoff. `Labels` remains the correct interface for
authoritative external targets such as official RelBench task tables.

Relationships are required when the compute graph contains feature tables:
file names alone cannot determine foreign-key direction or whether a join is
one-to-many. The two label/entity keys are also explicit so label attachment is
never guessed.

## What `fit` searches

The default search is deterministic and starts with:

```python
feature_families=("base",)
feature_family_max_columns=None
depth=1
auto_annotate_features=True
```

It then evaluates depths 2 and 3, the annotation switch, and the cumulative
family stages `base`, `base + temporal`, and
`base + temporal + conditional`. Customize these with `max_depth`,
`auto_annotate_options`, and `feature_family_stages`.

Every candidate holds the remaining node policy fixed: time-series periods
7/30/90 days, categorical cardinality threshold 20, categorical top-k 5,
automatic text features disabled, and annotation bounds 10 categorical
columns, 4 gated numeric columns, and top-k 3. These settings are assigned to
each node explicitly so they are effective with GraphReduce 1.10.

Each search source—including labels—is exposed to GraphReduce through a
temporary DuckDB view capped at `sample_rows=100_000`. Adapters can attach a
separate connected `search_source` while retaining their uncapped production
source. A new graph is created for every candidate because GraphReduce
execution mutates node state. If labels contain a timestamp, features are
built at each label cutoff; otherwise labels are split randomly (or by
`split_column`) and the current time is used as the feature cutoff.

Classification candidates use CatBoost and validation ROC AUC. Regression
candidates use CatBoost and validation MAE. The highest-performing candidate
is always retained as `best_trial`. KurveRSC also records feature count and
feature/model time. Trials that add at least 2x as many features for no more
than 0.002 AUC or 0.5% relative MAE improvement are marked in
`result.complexity_notes`; the simplest candidate within that tolerance is
`recommended_trial`.

## What the returned fitted model means

`fit` has a seven-stage lifecycle:

1. Build every candidate from source views capped at `sample_rows`.
2. Select the configuration by sampled validation ROC AUC or MAE.
3. Rebuild only the winner from the uncapped source tables.
4. Learn and freeze GraphReduce's exact SQL feature-operation plan on full
   training data.
5. Replay that plan and its training-only feature schema on full validation,
   then record `result.full_validation_score`.
6. Refit the production CatBoost estimator on full train plus validation
   without changing the frozen graph plan or feature schema.
7. Replay the plan with `GraphReduce(train=False)` at test cutoffs and expose
   predictions as `result.test_predictions`. If an external `Labels` test
   split contains targets, KurveRSC also records a test score.

The resulting production artifact is `result.fitted_model`: selected
`GraphConfig`, frozen execution plan, ordered feature schema, CatBoost model,
and validation/test metadata. `result.model` returns its final CatBoost model;
`result.execution_plan` returns the production GraphReduce plan. Validation
and test never run feature inference or annotation again.

Replay the fitted artifact on another timestamped entity frame with the same
declarative graph metadata:

```python
predictions = kurversc.predict(
    result,
    parent_node=parent,
    prediction_node=kurversc.Labels(
        scoring_rows, key="customer_id", timestamp="timestamp"
    ),
    tables=tables,
    relationships=relationships,
)
```

The output preserves prediction-row order and adds a `prediction` column.

Point-in-time production training can use many frames. Configuration search is
always performed on one sampled frame at the latest eligible training cutoff.
Supply all valid training cutoffs through `GraphLabels.train_cutoffs`, or all
timestamped rows through `Labels`, then choose how many full-refit frames are
used:

```python
result = kurversc.fit(
    ...,
    full_training_frames=15,   # 15 evenly spaced available train cutoffs
)
```

`full_training_frames=None` (the default) uses every available training
cutoff. These are point-in-time graph frames, not partitions of raw event
tables: every frame sees the complete history allowed by its cutoff, and all
frames replay one operation plan learned on the latest full-training frame.
When `full_training_frames=1`, KurveRSC always selects the latest eligible
training cutoff.

## Official RelBench tasks

`load_relbench_problem` uses the production RelBench dataset, task tables,
date keys, primary keys, and foreign keys without adding task-specific feature
expressions:

```python
import kurversc

problem = kurversc.load_relbench_problem(
    "rel-stack",
    "user-badge",
    sample_rows=10_000,
    max_train_timestamps=1,
)
result = kurversc.fit(**problem.fit_kwargs(), sample_rows=10_000)
```

Install the optional adapter with `pip install -e ".[relbench]"`. The object
adapter `relbench_problem_from_objects(...)` accepts a task, an already-censored
RelBench database, and its train/validation tables; RelArena uses this path so
its official inner and outer database cutoffs remain authoritative.

Relational schemas do require keys. This adapter reads them from official
RelBench metadata; for ordinary files or database tables, provide them with
`Table` and `Relationship`. Self-referential/cyclic foreign keys are omitted
because GraphReduce currently uses an acyclic `DiGraph`; every reachable
acyclic foreign-key path is represented as its own node instance.

For temporally meaningful relational evaluation, provide `Labels.timestamp`
and date columns on event tables. A dated parent is always filtered with
`parent.date <= Labels.timestamp` before feature inference and again through
GraphReduce's `do_filters_ops`. If the parent is a genuinely timeless entity
table, declare `Table(..., timeless=True)` explicitly; an omitted parent date
is otherwise rejected for temporal labels. Without event dates, KurveRSC
cannot distinguish historical features from future data.

## Local customer example

[`examples/cust_data_future_order.py`](examples/cust_data_future_order.py)
contains a complete run for `/usr/local/lake/cust_data`. It derives train and
validation labels through GraphReduce for “places an order in the following
365 days,” declares
the customer root as explicitly timeless, supplies every primary/foreign key
and event date, and runs the default 18-candidate search. The example enables
the `kurversc` logger at `INFO`, showing every attempted configuration, its
score/feature count/timing, and the selected configuration.
