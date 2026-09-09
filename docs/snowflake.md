# Snowflake-native joint optimization

`kurversc.fit_snowflake` treats the relational program and Snowflake ML model
as one validation-guided search. For every candidate `GraphConfig`, KurveRSC:

1. runs GraphReduce using `ComputeLayerEnum.snowflake`;
2. freezes uncertain operations on the latest training cutoff;
3. replays that exact plan over earlier training and validation cutoffs;
4. materializes temporary training and validation relations;
5. fits the selected Snowflake learner in the configured warehouse;
6. scores binary classification with ROC AUC or regression with MAE; and
7. removes the candidate relations before evaluating the next configuration.

The recommended configuration is materialized once more from its frozen plan,
refit on the combined training and validation cutoffs, and stored as the
learner's native Snowflake model object. The returned `FitResult` uses the same
trial and configuration objects as the DuckDB implementation. Its fitted
estimator is a `SnowflakeModelReference` containing the database, schema, model
name, learner, model type, task, and exact feature columns.

## Selectable learners

The learner is a first-class part of the joint search boundary:

```python
result = kurversc.fit_snowflake(..., learner="snowflake_native")
result = kurversc.fit_snowflake(..., learner="snowpark_xgboost")
```

- `snowflake_native` uses `SNOWFLAKE.ML.CLASSIFICATION`, evaluates every
  candidate against KurveRSC's explicit validation cutoffs, and persists the
  winner as a Snowflake classification object. Candidate model objects are
  dropped after scoring.
- `snowpark_xgboost` uses Snowpark ML `XGBClassifier` or `XGBRegressor` and
  persists the winner in Snowflake Model Registry.

Snowflake currently has no general tabular SQL ML regression class equivalent
to `SNOWFLAKE.ML.CLASSIFICATION`. The `SNOWFLAKE.ML.FORECAST` class is intended
for time-series forecasting and is not used as a silent regression substitute.
Choose `snowpark_xgboost` for general regression.

## Dependencies

```bash
pip install "kurversc[snowflake]"
```

The extra currently installs `snowflake-ml-python==1.47.0`. Snowflake imports
are lazy, so ordinary `kurversc.fit` users do not need the Snowflake runtime.

## Object lifecycle

- Input tables and views are never modified.
- Source aliases, cutoff frames, and candidate train/validation relations are
  Snowflake temporary objects scoped to the supplied connection.
- Failed candidates are recorded in `FitResult.trials` when
  `continue_on_error=True`.
- Candidate Snowpark models are not registered; candidate native SQL model
  objects are dropped immediately after their explicit validation score is
  calculated.
- Only the final refit winner remains as a persistent Snowflake model object.
- The frozen GraphReduce plan records the learner, model type, persistence
  mechanism, model FQN, and version (where applicable) under
  `execution_plan["kurversc_snowflake"]`.

Keep the supplied connector session alive for the duration of `fit_snowflake`.
Snowpark is created over that same connection so it can see the temporary
feature relations.

## Required access

The runtime role needs usage on the database, schema, and warehouse; read
access to every input relation; permission to create temporary tables and
views; and permission to create models in the output schema. A representative
grant set is:

```sql
GRANT USAGE ON WAREHOUSE KURVE_SNOWPARK_WH TO ROLE KURVE_ML_ROLE;
GRANT USAGE ON DATABASE KURVE TO ROLE KURVE_ML_ROLE;
GRANT USAGE ON SCHEMA KURVE.ML_OUTPUT TO ROLE KURVE_ML_ROLE;
GRANT CREATE TABLE, CREATE VIEW ON SCHEMA KURVE.ML_OUTPUT TO ROLE KURVE_ML_ROLE;
GRANT CREATE MODEL ON SCHEMA KURVE.ML_OUTPUT TO ROLE KURVE_ML_ROLE;
GRANT CREATE SNOWFLAKE.ML.CLASSIFICATION ON SCHEMA KURVE.ML_OUTPUT TO ROLE KURVE_ML_ROLE;
```

Grant `USAGE` and `SELECT` for the actual source database, schemas, tables, and
views separately. Account policies may require a dedicated role or additional
model-registry privileges.

For larger searches, use a dedicated Snowpark-optimized warehouse and pass its
name through `SnowflakeBackend.warehouse`. `sample_rows=None` evaluates full
source relations; a positive value limits each source alias during search.

## Current scope

The Snowflake execution path supports binary classification with either
learner and regression with Snowpark XGBoost, all with `GraphLabels`. External
`Labels` frames and Snowflake-native prediction replay will be added as
separate surfaces so the initial API does not silently move label or feature
matrices through the client process.
