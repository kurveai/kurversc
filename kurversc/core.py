"""Implementation of the one-call KurveRSC interface."""

from __future__ import annotations

import copy
import gc
import hashlib
import io
import multiprocessing
import re
from dataclasses import asdict, replace
from contextlib import ExitStack, contextmanager, nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime
from multiprocessing.connection import wait as wait_for_connections
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Iterable, Iterator, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd
import structlog

from .modeling import (
    FrameEnsemble,
    _binary_classes,
    fit_catboost,
    fit_final_catboost,
    fit_final_tabicl,
    fit_final_tabfm,
    fit_final_tabpfn_v3,
    fit_final_tabpfn_v3_5,
    fit_incremental_catboost_frame,
    fit_tabicl,
    fit_tabfm,
    fit_tabpfn_v3,
    fit_tabpfn_v3_5,
    prepare_features,
    prepare_prediction_features,
    prepare_training_features,
    sample_training_rows,
)
from .feature_audit import (
    ConfigFeatureEstimate,
    FeaturePreflight,
    available_feature_families,
    estimate_config_features,
    rank_feature_tables,
)
from .checkpointing import SearchCheckpoint
from .search import (
    DEFAULT_FAMILY_STAGES,
    FittedModel,
    FitResult,
    GraphConfig,
    SearchStageBudgets,
    Trial,
    adaptive_depth_candidate_allowed,
    annotate_complexity,
    confidence_frontier_trials,
    diverse_confirmation_trials,
    diverse_collection_configs,
    diverse_fidelity_trials,
    forward_candidate_allowed,
    graph_config_sort_key,
    incremental_configs,
    published_compatible_configs,
    resolve_feature_family_column_budgets,
    staged_challenger_configs,
    staged_modern_seed_configs,
)
from .specs import (
    GraphLabels,
    Key,
    Labels,
    Relationship,
    Source,
    Table,
    TARGET_HISTORY_TABLE,
    TARGET_HISTORY_TIME,
    TARGET_HISTORY_VALUE,
    coerce_labels,
    coerce_relationship,
    coerce_table,
)


_NAME = re.compile(r"[^A-Za-z0-9_]+")
logger = structlog.get_logger("kurversc")
_UNTIMED_LABEL_CUTOFF = datetime(2262, 1, 1)


def _with_graph_feature_options(
    configs: Sequence[GraphConfig],
    *,
    auto_text_features: bool = False,
    auto_annotate_max_text_columns: int | None = None,
    high_cardinality_policy: str = "skip",
    categorical_low_cardinality_policy: str = "top_k",
    text_categorical_fallback: bool = False,
) -> tuple[GraphConfig, ...]:
    """Apply nondefault fit options without resetting unrelated graph choices.

    The ordinary defaults mean no override for an explicit candidate. In
    particular, enabling high-cardinality handling must not demote its existing
    low-cardinality policy or disable its independently configured text view.
    Use the declarative graph itself to explicitly request a default-valued
    policy for a particular candidate.
    """
    overrides: dict[str, Any] = {}
    if auto_text_features:
        overrides["auto_text_features"] = True
    if auto_annotate_max_text_columns is not None:
        overrides["auto_annotate_max_text_columns"] = auto_annotate_max_text_columns
    if high_cardinality_policy != "skip":
        overrides["high_cardinality_policy"] = high_cardinality_policy
    if categorical_low_cardinality_policy != "top_k":
        overrides["categorical_low_cardinality_policy"] = (
            categorical_low_cardinality_policy
        )
    if text_categorical_fallback:
        overrides["text_categorical_fallback"] = True
    if not overrides:
        return tuple(configs)
    return tuple(replace(config, **overrides) for config in configs)


def _categorical_refinement_lineage(config: GraphConfig) -> GraphConfig:
    """Return the structural parent shared by categorical encoding variants."""

    return replace(
        config,
        categorical_tie_break="value",
        categorical_tie_break_seed=0,
        legacy_training_row_order=False,
    )


def _fork_process_map(
    function: Any,
    items: Sequence[Any],
    *,
    max_workers: int,
) -> list[Any]:
    """Map a closure in isolated forked workers while preserving input order.

    Search trials close over large, read-only relational frames and are not
    safely or cheaply serializable. POSIX ``fork`` shares those pages
    copy-on-write while giving every trial its own Python interpreter, DuckDB
    runtime, allocator, and model thread pool. This avoids the GIL and native
    lock contention produced by an in-process thread pool.
    """

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    values = tuple(items)
    if not values:
        return []
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError(
            "Parallel graph search requires a POSIX platform with the "
            "multiprocessing 'fork' start method"
        )

    context = multiprocessing.get_context("fork")
    results: list[Any] = [None] * len(values)
    active: dict[Any, tuple[int, Any]] = {}
    next_index = 0

    def child_main(index: int, item: Any, sender: Any) -> None:
        try:
            sender.send((True, function(item)))
        except BaseException as exc:
            sender.send((False, type(exc).__name__, str(exc)))
        finally:
            sender.close()

    def start_one(index: int) -> None:
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=child_main,
            args=(index, values[index], sender),
            name=f"kurversc-search-{index + 1}",
            daemon=True,
        )
        process.start()
        sender.close()
        active[receiver] = (index, process)

    try:
        while next_index < len(values) and len(active) < max_workers:
            start_one(next_index)
            next_index += 1
        while active:
            ready = wait_for_connections(tuple(active))
            for receiver in ready:
                index, process = active.pop(receiver)
                try:
                    payload = receiver.recv()
                except EOFError as exc:
                    process.join()
                    raise RuntimeError(
                        "Graph-search worker "
                        f"{index + 1} exited without returning a result "
                        f"(exitcode={process.exitcode})"
                    ) from exc
                finally:
                    receiver.close()
                process.join()
                if not payload[0]:
                    raise RuntimeError(
                        f"Graph-search worker {index + 1} failed with "
                        f"{payload[1]}: {payload[2]}"
                    )
                results[index] = payload[1]
                if next_index < len(values):
                    start_one(next_index)
                    next_index += 1
    except BaseException:
        for receiver, (_index, process) in active.items():
            receiver.close()
            if process.is_alive():
                process.terminate()
        for _receiver, (_index, process) in active.items():
            process.join()
        raise
    return results


def _is_classification(task: str) -> bool:
    return task in {"classification", "multiclass_classification"}


def _metric_for_task(task: str) -> str:
    if task == "classification":
        return "roc_auc"
    if task == "multiclass_classification":
        return "accuracy"
    return "mae"


def _objective_score(metric: str, score: float) -> float:
    return score if metric in {"roc_auc", "accuracy"} else -score


def _rerank_objective(
    metric: str,
    mean_score: float,
    score_std: float,
    fold_count: int,
    stability_penalty: float,
) -> tuple[float, float]:
    """Return the rerank objective and uncertainty of its estimated mean.

    Fold dispersion is useful evidence, but the selected quantity is the mean
    score. Penalizing the raw standard deviation overweights temporal
    heterogeneity, especially for the default three-fold rerank. Scale it by
    ``sqrt(fold_count)`` so the penalty represents uncertainty in that mean.
    """

    if fold_count < 1:
        raise ValueError("fold_count must be positive")
    score_standard_error = score_std / np.sqrt(fold_count)
    return (
        _objective_score(metric, mean_score) - stability_penalty * score_standard_error,
        score_standard_error,
    )


def _fit_validation_estimator(*args: Any, model_backend: str, **kwargs: Any):
    if model_backend == "catboost":
        return fit_catboost(*args, **kwargs)
    if model_backend == "tabpfn_v3":
        return fit_tabpfn_v3(*args, **kwargs)
    if model_backend == "tabpfn_v3_5":
        return fit_tabpfn_v3_5(*args, **kwargs)
    if model_backend == "tabicl":
        return fit_tabicl(*args, **kwargs)
    if model_backend == "tabfm":
        return fit_tabfm(*args, **kwargs)
    raise ValueError(
        "model_backend must be 'catboost', 'tabpfn_v3', 'tabpfn_v3_5', 'tabicl', or 'tabfm'"
    )


def _fit_final_estimator(*args: Any, model_backend: str, **kwargs: Any):
    if model_backend == "catboost":
        return fit_final_catboost(*args, **kwargs)
    if model_backend == "tabpfn_v3":
        return fit_final_tabpfn_v3(*args, **kwargs)
    if model_backend == "tabpfn_v3_5":
        return fit_final_tabpfn_v3_5(*args, **kwargs)
    if model_backend == "tabicl":
        return fit_final_tabicl(*args, **kwargs)
    if model_backend == "tabfm":
        return fit_final_tabfm(*args, **kwargs)
    raise ValueError(
        "model_backend must be 'catboost', 'tabpfn_v3', 'tabpfn_v3_5', 'tabicl', or 'tabfm'"
    )


def _fit_temporal_frame_estimator(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    model_backend: str,
    task: str,
    categorical: list[int],
    random_state: int,
    model_params: Mapping[str, Any] | None,
    target_classes: tuple[Any, ...] = (),
    iterations: int | None = None,
) -> tuple[Any, float, tuple[Any, ...]]:
    """Fit one independent estimator exactly as the frame ensemble does."""

    if model_backend == "catboost":
        return fit_incremental_catboost_frame(
            features,
            labels,
            task=task,
            categorical=categorical,
            random_state=random_state,
            model_params=model_params,
            init_model=None,
            target_classes=target_classes or None,
            iterations=iterations,
        )
    return _fit_final_estimator(
        features,
        labels,
        model_backend=model_backend,
        task=task,
        categorical=categorical,
        random_state=random_state,
        model_params=model_params,
    )


@contextmanager
def _connection_scope(
    connection: duckdb.DuckDBPyConnection | None,
    *,
    memory_limit: str | None = "128GB",
    max_temp_directory_size: str | None = "128GB",
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Own an in-memory DuckDB connection and its optional resource bounds."""

    if connection is not None:
        yield connection
        return
    with TemporaryDirectory(prefix="kurversc-duckdb-") as temporary_directory:
        con = duckdb.connect(":memory:")
        escaped = temporary_directory.replace("'", "''")
        con.sql(f"SET temp_directory='{escaped}'")
        if memory_limit is not None:
            escaped_memory_limit = memory_limit.replace("'", "''")
            con.sql(f"SET memory_limit='{escaped_memory_limit}'")
        # RelArena may run one isolated task per host core group. Allowing each
        # DuckDB connection to discover every host CPU multiplies worker stacks
        # and allocator arenas across parallel tasks without improving aggregate
        # throughput.
        con.sql("SET threads=8")
        # A sampled candidate must fail cleanly instead of filling the host
        # filesystem with an unbounded intermediate join. Callers supplying
        # their own connection retain complete control over DuckDB settings.
        if max_temp_directory_size is not None:
            escaped_limit = max_temp_directory_size.replace("'", "''")
            con.sql(f"SET max_temp_directory_size='{escaped_limit}'")
        try:
            yield con
        finally:
            con.close()


def _freeze_execution_plan(graph: Any, cut_date: datetime) -> dict[str, Any]:
    plan = graph.freeze_execution_plan()
    plan["kurversc_ts_periods"] = {
        node.prefix: tuple(getattr(node, "ts_periods", ()) or ())
        for node in graph.nodes()
    }
    # GraphReduce's external-label cut date is shifted by one microsecond to
    # make relation history inclusive. Root filters deliberately use the exact
    # task cutoff, so retain that second date for correct replay as well.
    plan["kurversc_root_cut_date"] = pd.Timestamp(cut_date).to_pydatetime()
    # Newly learned plans must preserve temporal windows when replay moves
    # backward onto an original window boundary. Retain legacy semantics for
    # unmarked historical artifacts, which callers may explicitly upgrade.
    plan["kurversc_date_rebinding"] = "atomic"
    return plan


def _prepare_execution_plan(
    plan: Mapping[str, Any], *, root_prefix: str, cut_date: datetime
) -> dict[str, Any]:
    if plan.get("kurversc_date_rebinding") == "atomic":
        from .plan_dates import prepare_atomic_plan

        return prepare_atomic_plan(plan, root_prefix=root_prefix, cut_date=cut_date)
    prepared = copy.deepcopy(dict(plan))
    original = prepared.get("kurversc_root_cut_date")
    if original is None:
        return prepared
    old = str(pd.Timestamp(original).to_pydatetime())
    new = str(pd.Timestamp(cut_date).to_pydatetime())
    for record in prepared.get("records", []):
        if record.get("node_prefix") != root_prefix:
            continue
        for field in ("ops", "method_ops", "date_filter_ops"):
            for operation in record.get(field, []):
                operation.opval = operation.opval.replace(old, new)
    return prepared


def _execution_plan_fingerprint(plan: Mapping[str, Any]) -> str:
    records: list[Any] = [
        tuple(
            sorted(
                (prefix, tuple(periods))
                for prefix, periods in plan.get("kurversc_ts_periods", {}).items()
            )
        )
    ]
    if plan.get("kurversc_date_rebinding") == "atomic":
        records.append(("kurversc_date_rebinding", "atomic"))
    for record in plan.get("records", []):
        records.append(
            (
                record.get("node_prefix"),
                record.get("method_name"),
                record.get("reduce_key"),
                record.get("edge"),
                tuple(
                    (getattr(operation, "optype", None), operation.opval)
                    for operation in record.get("ops", [])
                ),
            )
        )
    return hashlib.sha256(repr(records).encode()).hexdigest()


def _slug(value: str, fallback: str) -> str:
    result = _NAME.sub("_", value).strip("_").lower()
    if not result:
        result = fallback
    if result[0].isdigit():
        result = f"t_{result}"
    return result


def _source_name(source: Source, fallback: str) -> str:
    if isinstance(source, (str, Path)):
        value = str(source)
        path = Path(value)
        return _slug(path.stem if path.suffix else value.split(".")[-1], fallback)
    return fallback


def _key_parts(key: Key) -> list[str]:
    return [key] if isinstance(key, str) else list(key)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _quote_table(identifier: str) -> str:
    return ".".join(_quote_identifier(part) for part in identifier.split("."))


def _duckdb_compatible_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Present pandas 3 string columns to the GraphReduce-pinned DuckDB."""

    compatible = frame.copy(deep=False)
    string_columns = [
        column
        for column in compatible.columns
        if pd.api.types.is_string_dtype(compatible[column].dtype)
        and not pd.api.types.is_object_dtype(compatible[column].dtype)
    ]
    if string_columns:
        compatible = compatible.copy()
        for column in string_columns:
            compatible[column] = compatible[column].astype(object)
    return compatible


def _nested_sample_frame(
    frame: pd.DataFrame,
    *,
    limit: int,
    random_state: int,
    name: str,
) -> pd.DataFrame:
    """Return a deterministic, representative prefix of a nested row order.

    ``LIMIT`` is not a statistical sample.  In particular, RelBench search
    labels are grouped train-first and path tables are restored to source
    order after adapter sampling.  Taking their first rows can therefore omit
    an entire split or over-represent the earliest physical records.

    A fixed permutation for each named source makes the selected rows
    representative and ensures that a 10K sample is a subset of the same
    source's 50K sample.  Sorting the chosen positions restores stable source
    order after selection; relational aggregation itself is order-insensitive.
    """

    if limit < 1:
        raise ValueError("sample limit must be positive")
    if len(frame) <= limit:
        return frame
    name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    seed = (int(random_state) ^ name_seed) % (2**63 - 1)
    positions = np.random.default_rng(seed).permutation(len(frame))[:limit]
    positions.sort()
    return frame.iloc[positions]


class _Workspace:
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        sample_rows: int,
        random_state: int,
    ) -> None:
        self.connection = connection
        self.sample_rows = sample_rows
        self.random_state = random_state
        self.views: dict[str, str] = {}
        self.columns: dict[str, list[str]] = {}
        self._registered: list[str] = []
        self._auxiliary_views: list[str] = []

    def _source_query(self, source: Source, name: str) -> str:
        if isinstance(source, pd.DataFrame):
            registration = f"kurversc_frame_{name}_{len(self._registered)}"
            # DuckDB 1.2 predates pandas 3's dedicated ``str`` dtype. Keep the
            # public API compatible with both by presenting string columns as
            # the object dtype understood by that GraphReduce-pinned release.
            compatible = _duckdb_compatible_frame(source)
            self.connection.register(registration, compatible)
            self._registered.append(registration)
            return f"SELECT * FROM {_quote_identifier(registration)}"
        value = str(source)
        path = Path(value).expanduser()
        if path.is_file():
            escaped = str(path.resolve()).replace("'", "''")
            if path.suffix.lower() in {".parquet", ".pq"}:
                return f"SELECT * FROM read_parquet('{escaped}')"
            if path.suffix.lower() in {".csv", ".csv.gz"}:
                return f"SELECT * FROM read_csv_auto('{escaped}')"
            raise ValueError(f"Unsupported source file type: {path.suffix}")
        return f"SELECT * FROM {_quote_table(value)}"

    def add(self, name: str, source: Source, *, sample: bool = True) -> str:
        if name in self.views:
            raise ValueError(f"Duplicate table name: {name}")
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        view = f"kurversc_{_slug(name, 'table')}_{digest}"
        effective_source = source
        sampled_in_memory = sample and isinstance(source, pd.DataFrame)
        if sampled_in_memory:
            effective_source = _nested_sample_frame(
                source,
                limit=self.sample_rows,
                random_state=self.random_state,
                name=name,
            )
        query = self._source_query(effective_source, name)
        # Files and database relations retain the legacy bounded query path;
        # in-memory frames, including RelBench adapters, use representative
        # deterministic sampling before registration.
        limit = f" LIMIT {self.sample_rows}" if sample and not sampled_in_memory else ""
        self.connection.sql(
            f"CREATE OR REPLACE TEMP VIEW {_quote_identifier(view)} AS "
            f"SELECT * FROM ({query}) AS kurversc_source{limit}"
        )
        self.views[name] = view
        description = self.connection.sql(
            f"DESCRIBE SELECT * FROM {_quote_identifier(view)}"
        ).to_df()
        self.columns[name] = description["column_name"].astype(str).tolist()
        return view

    def frame(self, name: str) -> pd.DataFrame:
        return self.connection.sql(
            f"SELECT * FROM {_quote_identifier(self.views[name])}"
        ).to_df()

    def entity_filtered_view(
        self,
        name: str,
        *,
        key_columns: Sequence[str],
        entity_keys: pd.DataFrame,
    ) -> str:
        """Restrict a root source to the entities labeled at one cutoff."""

        ordinal = len(self._registered)
        registration = f"kurversc_entity_keys_{ordinal}"
        view = f"kurversc_{_slug(name, 'root')}_entities_{ordinal}"
        keys = entity_keys.loc[:, list(key_columns)].drop_duplicates().copy()
        # DuckDB 1.2 cannot register pandas' StringDtype, which pandas 3 now
        # uses for ordinary strings returned by ``to_df``.  Normalize only at
        # this interoperability boundary and preserve the key values.
        for column in keys.columns:
            if isinstance(keys[column].dtype, pd.StringDtype):
                keys[column] = keys[column].astype("object")
        self.connection.register(registration, keys)
        self._registered.append(registration)
        predicates = " AND ".join(
            f"source.{_quote_identifier(column)} = "
            f"task_entities.{_quote_identifier(column)}"
            for column in key_columns
        )
        self.connection.sql(
            f"CREATE OR REPLACE TEMP VIEW {_quote_identifier(view)} AS "
            f"SELECT source.* FROM {_quote_identifier(self.views[name])} AS source "
            f"INNER JOIN {_quote_identifier(registration)} AS task_entities "
            f"ON {predicates}"
        )
        self._auxiliary_views.append(view)
        return view

    @contextmanager
    def reachable_sources(
        self,
        tables: Mapping[str, Table],
        root_name: str,
        relationships: Sequence[Relationship],
        root_keys: Sequence[str],
        entity_keys: pd.DataFrame,
    ) -> Iterator[None]:
        """Push a root cohort through a path graph before SQL aggregation.

        A child row that cannot be reached from any requested root cannot
        contribute to that root's relational aggregates. Materializing only
        reachable rows prevents a small prediction batch from first creating
        a full-size temporary table for every child relation.
        """

        original = self.views.copy()
        registered_before = len(self._registered)
        auxiliary_before = len(self._auxiliary_views)
        temporary_tables: list[str] = []
        scoped_children: set[str] = set()
        try:
            self.views[root_name] = self.entity_filtered_view(
                root_name, key_columns=root_keys, entity_keys=entity_keys
            )
            pending = list(relationships)
            while pending:
                remaining = []
                progressed = False
                for edge in pending:
                    if edge.parent not in self.views or self.views[edge.parent] == original.get(edge.parent):
                        remaining.append(edge)
                        continue
                    if edge.child in scoped_children:
                        raise ValueError("Root-reachable materialization requires path-specific child nodes")
                    ordinal = len(temporary_tables)
                    view = f"kurversc_reachable_{_slug(edge.child, 'child')}_{ordinal}_{registered_before}"
                    predicates = " AND ".join(
                        f"parent.{_quote_identifier(parent_key)} = source.{_quote_identifier(child_key)}"
                        for parent_key, child_key in zip(
                            _key_parts(edge.parent_key), _key_parts(edge.child_key), strict=True
                        )
                    )
                    table = tables[edge.child]
                    selected = (
                        list(table.columns)
                        if table.columns is not None
                        else list(self.columns[edge.child])
                    )
                    structural = [
                        *(_key_parts(table.key) if table.key is not None else ()),
                        *((table.date,) if table.date is not None else ()),
                        *table.context_keys,
                    ]
                    for related in relationships:
                        if related.parent == edge.child:
                            structural.extend(_key_parts(related.parent_key))
                        if related.child == edge.child:
                            structural.extend(_key_parts(related.child_key))
                    selected = [
                        column for column in dict.fromkeys((*selected, *structural))
                        if column in self.columns[edge.child]
                    ]
                    projection = ", ".join(
                        f"source.{_quote_identifier(column)}" for column in selected
                    )
                    self.connection.sql(
                        f"CREATE TEMP TABLE {_quote_identifier(view)} AS "
                        f"SELECT {projection} FROM {_quote_identifier(original[edge.child])} AS source "
                        f"WHERE EXISTS (SELECT 1 FROM {_quote_identifier(self.views[edge.parent])} "
                        f"AS parent WHERE {predicates})"
                    )
                    temporary_tables.append(view)
                    scoped_children.add(edge.child)
                    self.views[edge.child] = view
                    progressed = True
                if not progressed:
                    raise ValueError("Root-reachable materialization requires an acyclic path graph")
                pending = remaining
            yield
        finally:
            self.views = original
            for view in reversed(temporary_tables):
                self.connection.sql(f"DROP TABLE IF EXISTS {_quote_identifier(view)}")
            for view in self._auxiliary_views[auxiliary_before:]:
                self.connection.sql(f"DROP VIEW IF EXISTS {_quote_identifier(view)}")
            del self._auxiliary_views[auxiliary_before:]
            for registration in self._registered[registered_before:]:
                self.connection.unregister(registration)
            del self._registered[registered_before:]

    def close(self) -> None:
        for view in self._auxiliary_views:
            try:
                self.connection.sql(f"DROP VIEW IF EXISTS {_quote_identifier(view)}")
            except Exception:
                pass
        for view in self.views.values():
            try:
                self.connection.sql(f"DROP VIEW IF EXISTS {_quote_identifier(view)}")
            except Exception:
                pass
        for registration in self._registered:
            try:
                self.connection.unregister(registration)
            except Exception:
                pass


@contextmanager
def _table_workspace(
    connection: duckdb.DuckDBPyConnection | None,
    tables: Mapping[str, Table],
    *,
    sample_rows: int,
    random_state: int,
    search: bool,
    fidelity: bool = False,
    search_full_data: bool = False,
    duckdb_memory_limit: str | None = "128GB",
    duckdb_max_temp_directory_size: str | None = "128GB",
) -> Iterator[_Workspace]:
    """Open a disposable workspace for one materialized feature frame.

    Internally owned DuckDB connections are intentionally scoped this tightly:
    closing the connection between configurations/cutoffs releases native query
    arenas that Python garbage collection cannot see.
    """

    with _connection_scope(
        connection,
        memory_limit=duckdb_memory_limit,
        max_temp_directory_size=duckdb_max_temp_directory_size,
    ) as con:
        workspace = _Workspace(
            con,
            sample_rows=sample_rows,
            random_state=random_state,
        )
        try:
            for name, table in tables.items():
                # ``search_source`` is the adapter's task-connected relation,
                # not necessarily a row sample.  ``search_full_data`` disables
                # the row cap but must not swap that cohort for unrelated rows
                # from the physical production source: doing so changes
                # automatic annotations before the root-entity join is applied.
                if search and fidelity and table.fidelity_source is not None:
                    source = table.fidelity_source
                elif search and table.search_source is not None:
                    source = table.search_source
                else:
                    source = table.source
                workspace.add(
                    name,
                    source,
                    sample=(search and not search_full_data and table.sample_in_search),
                )
            yield workspace
        finally:
            workspace.close()
            gc.collect()


def _load_frame_source(
    source: Source,
    *,
    connection: duckdb.DuckDBPyConnection | None,
    sample_rows: int,
    random_state: int,
    sample: bool,
    name: str,
    duckdb_memory_limit: str | None = "128GB",
    duckdb_max_temp_directory_size: str | None = "128GB",
) -> pd.DataFrame:
    """Load a label/prediction source in its own short-lived workspace."""

    with _connection_scope(
        connection,
        memory_limit=duckdb_memory_limit,
        max_temp_directory_size=duckdb_max_temp_directory_size,
    ) as con:
        workspace = _Workspace(
            con,
            sample_rows=sample_rows,
            random_state=random_state,
        )
        try:
            workspace.add(name, source, sample=sample)
            return workspace.frame(name)
        finally:
            workspace.close()


def _target_history_frame(
    labels: pd.DataFrame,
    spec: Labels,
) -> pd.DataFrame:
    """Return the generic temporal child relation for observed targets."""

    key_columns = _key_parts(spec.key)
    clock = spec.available_at or spec.timestamp
    selected = [*key_columns, clock, spec.target]
    history = labels.loc[:, selected].dropna(subset=selected).copy()
    history = history.rename(
        columns={
            clock: TARGET_HISTORY_TIME,
            spec.target: TARGET_HISTORY_VALUE,
        }
    )
    history[TARGET_HISTORY_TIME] = pd.to_datetime(history[TARGET_HISTORY_TIME])
    return history.reset_index(drop=True)


def _prepare_target_history(
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    *,
    root_name: str,
    labels_spec: Labels | None,
    task: str,
    validation_fraction: float,
    random_state: int,
    connection: duckdb.DuckDBPyConnection | None,
    duckdb_memory_limit: str | None,
    duckdb_max_temp_directory_size: str | None,
) -> tuple[dict[str, Table], tuple[Relationship, ...], pd.DataFrame | None]:
    """Attach full, unsampled observed labels as a strict temporal relation."""

    normalized_tables = dict(tables)
    normalized_relationships = tuple(relationships)
    if (
        labels_spec is None
        or not labels_spec.target_history
        or labels_spec.timestamp is None
    ):
        return normalized_tables, normalized_relationships, None
    if TARGET_HISTORY_TABLE in normalized_tables:
        raise ValueError(
            f"{TARGET_HISTORY_TABLE!r} is reserved for automatic target history"
        )

    history_source = (
        labels_spec.history_source
        if labels_spec.history_source is not None
        else labels_spec.source
    )
    history_labels = _load_frame_source(
        history_source,
        connection=connection,
        sample_rows=1,
        random_state=random_state,
        sample=False,
        name="target_history_labels",
        duckdb_memory_limit=duckdb_memory_limit,
        duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
    )
    required = {
        *_key_parts(labels_spec.key),
        labels_spec.timestamp,
        labels_spec.target,
        *([labels_spec.available_at] if labels_spec.available_at else []),
        *([labels_spec.split] if labels_spec.split else []),
    }
    missing = required - set(history_labels.columns)
    if missing:
        raise ValueError(f"Target-history source is missing columns: {sorted(missing)}")
    labeled = history_labels.dropna(subset=[labels_spec.target])
    resolved_task = _infer_task(labeled[labels_spec.target], task)
    train_labels, validation_labels = _split_labels(
        labeled,
        labels_spec,
        task=resolved_task,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    training_history = _target_history_frame(train_labels, labels_spec)
    prediction_history = _target_history_frame(
        pd.concat([train_labels, validation_labels], ignore_index=True), labels_spec
    )
    root = normalized_tables[root_name]
    history_table = Table(
        training_history,
        search_source=training_history,
        fidelity_source=training_history,
        name=TARGET_HISTORY_TABLE,
        date=TARGET_HISTORY_TIME,
        prefix="targethist",
        columns=(
            *_key_parts(labels_spec.key),
            TARGET_HISTORY_TIME,
            TARGET_HISTORY_VALUE,
        ),
        sample_in_search=False,
        cutoff_boundary="inclusive" if labels_spec.available_at else "strict",
        available_at=TARGET_HISTORY_TIME if labels_spec.available_at else None,
        auto_base_predicate_max=0,
    )
    normalized_tables[TARGET_HISTORY_TABLE] = history_table
    normalized_relationships = (
        *normalized_relationships,
        Relationship(
            parent=root_name,
            child=TARGET_HISTORY_TABLE,
            parent_key=root.key,
            child_key=labels_spec.key,
        ),
    )
    return normalized_tables, normalized_relationships, prediction_history


def _iteration_schedule(total_iterations: int, frames: int) -> tuple[int, ...]:
    """Distribute a fixed CatBoost tree budget over sequential frame updates."""

    if total_iterations < 1:
        raise ValueError("CatBoost iterations must be positive")
    if frames < 1:
        raise ValueError("At least one incremental training frame is required")
    if total_iterations < frames:
        # CatBoost needs at least one new tree per continuation. This is only a
        # corner case for deliberately tiny test configurations.
        return (1,) * frames
    quotient, remainder = divmod(total_iterations, frames)
    return tuple(quotient + (1 if index < remainder else 0) for index in range(frames))


def _plan_with_schema(
    plan: Mapping[str, Any],
    *,
    columns: Sequence[str],
    categorical_columns: Sequence[str],
    datetime_columns: Sequence[str],
) -> dict[str, Any]:
    frozen = copy.deepcopy(dict(plan))
    frozen["kurversc_feature_schema"] = {
        "columns": tuple(columns),
        "categorical_columns": tuple(categorical_columns),
        "datetime_columns": tuple(datetime_columns),
    }
    return frozen


def _schema_from_plan(
    plan: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    schema = plan.get("kurversc_feature_schema")
    if not isinstance(schema, Mapping):
        return None
    columns = tuple(schema.get("columns", ()))
    if not columns:
        return None
    return (
        columns,
        tuple(schema.get("categorical_columns", ())),
        tuple(schema.get("datetime_columns", ())),
    )


def _score_prediction_chunks(
    labels: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    *,
    task: str,
    target_classes: tuple[Any, ...],
) -> tuple[str, float]:
    if not labels or not predictions:
        raise ValueError("At least one validation frame is required")
    truth = np.concatenate(labels)
    estimate = np.concatenate(predictions)
    if task == "classification":
        from sklearn.metrics import roc_auc_score

        class_map = {value: index for index, value in enumerate(target_classes)}
        encoded = pd.Series(truth).map(class_map)
        if encoded.isna().any():
            raise ValueError("Validation contains a class absent from training")
        if encoded.nunique() != 2:
            raise ValueError("Validation labels must contain both classes for ROC AUC")
        return "roc_auc", float(roc_auc_score(encoded, estimate))
    if task == "multiclass_classification":
        from sklearn.metrics import accuracy_score

        class_map = {value: index for index, value in enumerate(target_classes)}
        encoded = pd.Series(truth).map(class_map)
        if estimate.ndim != 2 or estimate.shape[1] != len(target_classes):
            raise ValueError("Multiclass predictions must align with target classes")
        # At a temporal cutoff, a newly appearing category cannot be predicted by
        # a model fit on preceding frames. It is a valid (necessarily incorrect)
        # holdout observation, not a malformed split.
        encoded_values = encoded.fillna(-1).to_numpy(dtype=int)
        return "accuracy", float(
            accuracy_score(encoded_values, np.argmax(estimate, axis=1))
        )
    from sklearn.metrics import mean_absolute_error

    return "mae", float(mean_absolute_error(pd.to_numeric(pd.Series(truth)), estimate))


def _prediction_standard_error(
    labels: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    *,
    task: str,
    target_classes: tuple[Any, ...],
) -> float:
    """Estimate metric sampling uncertainty without retaining feature frames."""

    if not labels or not predictions:
        return 0.0
    truth = np.concatenate(labels)
    estimate = np.concatenate(predictions)
    if len(truth) < 2:
        return 0.0
    if task == "classification":
        from sklearn.metrics import roc_auc_score

        class_map = {value: index for index, value in enumerate(target_classes)}
        encoded = pd.Series(truth).map(class_map).to_numpy()
        positives = int(np.sum(encoded == 1))
        negatives = int(np.sum(encoded == 0))
        if positives < 1 or negatives < 1:
            return 0.0
        auc = float(roc_auc_score(encoded, estimate))
        q1 = auc / max(1e-12, 2.0 - auc)
        q2 = 2.0 * auc * auc / max(1e-12, 1.0 + auc)
        variance = (
            auc * (1.0 - auc)
            + (positives - 1) * (q1 - auc * auc)
            + (negatives - 1) * (q2 - auc * auc)
        ) / (positives * negatives)
        return float(np.sqrt(max(0.0, variance)))

    if task == "multiclass_classification":
        if estimate.ndim != 2 or estimate.shape[1] != len(target_classes):
            return 0.0
        class_map = {value: index for index, value in enumerate(target_classes)}
        encoded = pd.Series(truth).map(class_map).fillna(-1).to_numpy(dtype=int)
        correct = (np.argmax(estimate, axis=1) == encoded).astype(float)
        return float(np.std(correct, ddof=1) / np.sqrt(len(correct)))
    errors = np.abs(
        pd.to_numeric(pd.Series(truth)).to_numpy(dtype=float)
        - np.asarray(estimate, dtype=float)
    )
    if len(errors) < 2:
        return 0.0
    return float(np.std(errors, ddof=1) / np.sqrt(len(errors)))


def _predict_values(model: Any, features: pd.DataFrame, *, task: str) -> np.ndarray:
    if task == "classification":
        return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    if task == "multiclass_classification":
        return np.asarray(model.predict_proba(features), dtype=float)
    return np.asarray(model.predict(features), dtype=float)


def _training_groups(
    labels: pd.DataFrame,
    label_spec: Labels,
) -> Iterator[tuple[datetime, pd.DataFrame]]:
    if labels.empty:
        return
    if label_spec.timestamp:
        for timestamp, frame in labels.groupby(
            label_spec.timestamp, sort=True, dropna=False
        ):
            if pd.isna(timestamp):
                raise ValueError("Label timestamps must not be missing")
            yield pd.Timestamp(timestamp).to_pydatetime(), frame
    else:
        # A wall-clock cutoff makes identical untimed runs depend on when they
        # execute. Use a stable, pandas-safe far-future boundary so their
        # intended all-history semantics are reproducible.
        yield _UNTIMED_LABEL_CUTOFF, labels


def _ordered_materialization_labels(
    labels: pd.DataFrame,
    label_spec: Labels,
) -> tuple[pd.DataFrame, str]:
    """Attach a canonical row ordinal before a SQL result is merged.

    Relational SQL does not promise output order. CatBoost's ordered boosting
    and positional estimator sampling do depend on row order, so allowing a
    join to choose it makes identical graph trials produce different scores.
    Prediction rows already carry a caller-order column; supervised rows use
    entity, timestamp, split, and target as their canonical order.
    """

    ordered = labels.copy()
    prediction_order = next(
        (
            column
            for column in ordered.columns
            if column.startswith("__kurversc_prediction_order__")
        ),
        None,
    )
    if prediction_order is not None:
        sort_columns = [prediction_order]
    else:
        sort_columns = [
            column
            for column in (
                *_key_parts(label_spec.key),
                label_spec.timestamp,
                label_spec.split,
            )
            if column is not None and column in ordered.columns
        ]
    if sort_columns:
        ordered = ordered.sort_values(sort_columns, kind="stable").reset_index(
            drop=True
        )
    order_column = "__kurversc_materialization_order__"
    while order_column in ordered.columns:
        order_column += "_"
    ordered[order_column] = np.arange(len(ordered), dtype=np.int64)
    return ordered, order_column


def _drop_graph_markers(
    frame: pd.DataFrame,
    *,
    split_marker: str,
) -> pd.DataFrame:
    return frame.drop(columns=[split_marker], errors="ignore")


def _normalize_tables(
    parent: Table,
    tables: Sequence[Table | Source] | Mapping[str, Table | Source],
) -> tuple[str, dict[str, Table]]:
    root_name = parent.name or _source_name(parent.source, "parent")
    root = replace(parent, name=root_name)
    normalized = {root_name: root}
    values: Iterable[tuple[str | None, Table | Source]]
    values = (
        tables.items()
        if isinstance(tables, Mapping)
        else ((None, item) for item in tables)
    )
    for supplied_name, value in values:
        table = value if isinstance(value, Table) else Table(value, name=supplied_name)
        name = (
            supplied_name
            or table.name
            or _source_name(table.source, f"table_{len(normalized)}")
        )
        if name in normalized:
            raise ValueError(f"Duplicate table name: {name}")
        normalized[name] = replace(table, name=name)
    return root_name, normalized


def _rank_feature_funnel(
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    *,
    connection: duckdb.DuckDBPyConnection | None,
    sample_rows: int,
    random_state: int,
    feature_family_max_columns: int | None,
    feature_family_max_column_options: Sequence[int | None],
    feature_family_max_features_per_column: int | None,
    intelligent_column_selection: bool,
    duckdb_memory_limit: str | None,
    duckdb_max_temp_directory_size: str | None,
) -> tuple[dict[str, Table], pd.DataFrame]:
    """Fit one deterministic source-column ordering for the whole search."""

    with _table_workspace(
        connection,
        tables,
        sample_rows=sample_rows,
        random_state=random_state,
        search=True,
        duckdb_memory_limit=duckdb_memory_limit,
        duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
    ) as workspace:
        samples = {name: workspace.frame(name) for name in tables}
    return rank_feature_tables(
        tables,
        relationships,
        samples,
        feature_family_max_columns=feature_family_max_columns,
        feature_family_max_column_options=feature_family_max_column_options,
        feature_family_max_features_per_column=(feature_family_max_features_per_column),
        intelligent_column_selection=intelligent_column_selection,
    )


def _known_source_row_counts(
    tables: Mapping[str, Table],
    *,
    search: bool,
    overrides: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Return free row counts and caller-supplied counts for planning."""

    counts = dict(overrides or {})
    unknown = set(counts) - set(tables)
    if unknown:
        raise ValueError(
            f"source_row_counts contains unknown tables: {sorted(unknown)}"
        )
    if any(rows < 0 for rows in counts.values()):
        raise ValueError("source_row_counts must contain non-negative row counts")
    for name, table in tables.items():
        source = (
            table.search_source
            if search and table.search_source is not None
            else table.source
        )
        if name not in counts and isinstance(source, pd.DataFrame):
            counts[name] = len(source)
    return counts


def _split_labels(
    labels: pd.DataFrame,
    spec: Labels,
    *,
    task: str,
    validation_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if spec.split:
        train = labels.loc[labels[spec.split] == spec.train_value].copy()
        validation = labels.loc[labels[spec.split] == spec.validation_value].copy()
    elif spec.timestamp:
        ordered = labels.sort_values(spec.timestamp, kind="stable")
        boundary = max(
            1, min(len(ordered) - 1, round(len(ordered) * (1 - validation_fraction)))
        )
        train, validation = (
            ordered.iloc[:boundary].copy(),
            ordered.iloc[boundary:].copy(),
        )
    else:
        from sklearn.model_selection import train_test_split

        stratify = labels[spec.target] if _is_classification(task) else None
        train, validation = train_test_split(
            labels,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=stratify,
        )
    if train.empty or validation.empty:
        raise ValueError("The label split produced an empty train or validation set")
    return train, validation


def _infer_task(labels: pd.Series, requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        classes = labels.nunique(dropna=True)
        return (
            "classification"
            if classes == 2
            else "multiclass_classification"
            if classes > 2
            else "regression"
        )
    if requested not in {"classification", "multiclass_classification", "regression"}:
        raise ValueError(
            "task must be 'classification', 'multiclass_classification', "
            "'regression', or 'auto'"
        )
    return requested


def _select_training_cutoffs(
    values: Sequence[Any], limit: int | None
) -> tuple[pd.Timestamp, ...]:
    """Select deterministic, evenly spaced point-in-time training frames."""

    timestamps = tuple(sorted({pd.Timestamp(value) for value in values}))
    if limit is None or len(timestamps) <= limit:
        return timestamps
    if limit < 1:
        raise ValueError("training frame counts must be positive")
    if limit == 1:
        return (timestamps[-1],)
    span = len(timestamps) - 1
    intervals = limit - 1
    indices = tuple(
        (index * span + intervals // 2) // intervals for index in range(limit)
    )
    return tuple(timestamps[index] for index in indices)


def _temporal_frame_row_limit(
    total_limit: int | None,
    *,
    frame_count: int,
    frame_index: int,
) -> int | None:
    """Allocate a global estimator-row budget evenly across time frames."""

    if total_limit is None:
        return None
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if not 0 <= frame_index < frame_count:
        raise ValueError("frame_index must identify an available frame")
    if total_limit < frame_count:
        raise ValueError(
            "estimator_train_rows must be at least the number of temporal "
            f"frames ({frame_count}) so every cutoff is represented"
        )
    base, extra = divmod(total_limit, frame_count)
    return base + (frame_index < extra)


def _sample_temporal_frame(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    total_limit: int | None,
    frame_count: int,
    frame_index: int,
    task: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Take a deterministic, time-balanced share of a global row budget."""

    return sample_training_rows(
        features,
        labels,
        limit=_temporal_frame_row_limit(
            total_limit,
            frame_count=frame_count,
            frame_index=frame_index,
        ),
        task=task,
        random_state=random_state + frame_index,
    )


def _select_label_training_cutoffs(
    labels: pd.DataFrame,
    *,
    timestamp: str,
    target: str,
    task: str,
    limit: int | None,
) -> tuple[pd.Timestamp, ...]:
    """Prefer classification cutoffs that contain at least two classes.

    Sparse temporal heads can contain valid labels overall while a mechanically
    spaced cutoff happens to contain only one class.  When enough class-valid
    cutoffs exist, space the requested frames across those dates so a candidate
    is not mislabeled as failed for reasons unrelated to its graph program.
    """

    values = labels[timestamp].dropna().tolist()
    default = _select_training_cutoffs(values, limit)
    if task not in {"classification", "multiclass_classification"} or limit is None:
        return default
    class_counts = labels.groupby(timestamp, sort=True, dropna=False)[target].nunique(
        dropna=True
    )
    eligible = [
        pd.Timestamp(value)
        for value, count in class_counts.items()
        if not pd.isna(value) and count >= 2
    ]
    if len(eligible) < min(limit, len(default)):
        return default
    return _select_training_cutoffs(eligible, limit)


def _resource_block_for(
    config: GraphConfig,
    blocks: Sequence[tuple[tuple[str, ...], bool, int, str]],
) -> tuple[tuple[str, ...], bool, int, str] | None:
    """Return the prior resource failure that makes a candidate a superset."""

    for block in blocks:
        families, annotations, minimum_depth, _reason = block
        if (
            config.auto_annotate_features == annotations
            and config.depth >= minimum_depth
            and set(families).issubset(config.feature_families)
        ):
            return block
    return None


def _observed_feature_width_estimate(
    config: GraphConfig,
    completed_trials: Sequence[Trial],
    *,
    audit_estimate: int,
) -> int:
    """Refine the audit estimate using already materialized parent shapes."""

    predictions: list[int] = []
    for trial in completed_trials:
        if trial.status != "completed" or trial.feature_count < 1:
            continue
        parent = trial.config
        if (
            parent.auto_annotate_features != config.auto_annotate_features
            or parent.feature_family_max_columns != config.feature_family_max_columns
            or parent.feature_family_max_features_per_column
            != config.feature_family_max_features_per_column
            or parent.feature_propagation_max_functions_per_column
            != config.feature_propagation_max_functions_per_column
        ):
            continue
        if (
            parent.feature_families == config.feature_families
            and parent.depth == config.depth - 1
        ):
            predictions.append(int(np.ceil(trial.feature_count * 1.75)))
        elif (
            parent.depth == config.depth
            and set(parent.feature_families).issubset(config.feature_families)
            and len(config.feature_families) - len(parent.feature_families) == 1
        ):
            predictions.append(int(np.ceil(trial.feature_count * 1.5)))
    # Observations refine a conservative static estimate upward; they must not
    # erase topology costs that are invisible in the already-materialized root
    # frame (the failure mode for deep relational candidates).
    return max((audit_estimate, *predictions))


def _select_trials(
    successful: Sequence[Trial],
    *,
    classification_negligible_gain: float,
    regression_negligible_relative_gain: float,
    complexity_uncertainty_multiplier: float = 1.0,
) -> tuple[Trial, Trial]:
    """Return the raw winner and complexity-aware recommendation."""

    if not successful:
        raise ValueError("At least one successful trial is required")
    best = min(
        successful,
        key=lambda item: (
            -item.objective_score,
            item.feature_count,
            item.config.complexity,
            graph_config_sort_key(item.config),
        ),
    )
    eligible = _trials_indistinguishable_from(
        successful,
        best,
        classification_negligible_gain=classification_negligible_gain,
        regression_negligible_relative_gain=regression_negligible_relative_gain,
        complexity_uncertainty_multiplier=complexity_uncertainty_multiplier,
    )
    recommended = min(
        eligible,
        key=lambda item: (
            item.feature_count,
            item.config.complexity,
            graph_config_sort_key(item.config),
        ),
    )
    return best, recommended


def _trials_indistinguishable_from(
    trials: Sequence[Trial],
    best: Trial,
    *,
    classification_negligible_gain: float,
    regression_negligible_relative_gain: float,
    complexity_uncertainty_multiplier: float,
) -> list[Trial]:
    """Return trials whose loss from ``best`` is inside measured uncertainty."""

    eligible: list[Trial] = []
    for trial in trials:
        uncertainty = (
            complexity_uncertainty_multiplier
            * (best.validation_standard_error**2 + trial.validation_standard_error**2)
            ** 0.5
        )
        if best.metric in {"roc_auc", "accuracy"}:
            loss = best.validation_score - trial.validation_score
            tolerance = max(classification_negligible_gain, uncertainty)
        else:
            loss = trial.validation_score - best.validation_score
            tolerance = max(
                abs(best.validation_score) * regression_negligible_relative_gain,
                uncertainty,
            )
        if loss <= tolerance + 1e-12:
            eligible.append(trial)
    return eligible


def _temporal_complexity_equivalents(
    trials: Sequence[Trial],
    best: Trial,
    *,
    classification_negligible_gain: float,
    regression_negligible_relative_gain: float,
    complexity_uncertainty_multiplier: float,
    classification_loss_cap: float,
    regression_relative_loss_cap: float,
) -> list[Trial]:
    """Return only practical ties eligible for a smaller-graph substitution.

    Temporal finalists are evaluated on the same folds, so their errors are
    paired.  Using the candidates' marginal standard errors as if they were
    independent can make a noisy three-fold panel declare a materially worse
    graph equivalent to the winner.  Estimate uncertainty from paired fold
    losses when possible, and always cap the resulting tolerance by a small
    practical loss.  The recent-fold safety gate remains separately
    uncertainty-aware.
    """

    eligible: list[Trial] = []
    for trial in trials:
        if best.metric in {"roc_auc", "accuracy"}:
            loss = best.validation_score - trial.validation_score
            negligible_loss = classification_negligible_gain
            maximum_loss = classification_loss_cap
        else:
            loss = trial.validation_score - best.validation_score
            scale = abs(best.validation_score)
            negligible_loss = scale * regression_negligible_relative_gain
            maximum_loss = scale * regression_relative_loss_cap

        paired_losses: list[float] = []
        if len(best.cutoff_scores) == len(trial.cutoff_scores):
            for best_score, trial_score in zip(
                best.cutoff_scores,
                trial.cutoff_scores,
                strict=True,
            ):
                if best_score is None or trial_score is None:
                    paired_losses = []
                    break
                paired_losses.append(
                    _objective_score(best.metric, float(best_score))
                    - _objective_score(trial.metric, float(trial_score))
                )
        if len(paired_losses) >= 2:
            uncertainty = complexity_uncertainty_multiplier * float(
                np.std(paired_losses) / np.sqrt(len(paired_losses))
            )
        else:
            uncertainty = (
                complexity_uncertainty_multiplier
                * (
                    best.validation_standard_error**2
                    + trial.validation_standard_error**2
                )
                ** 0.5
            )
        tolerance = min(max(negligible_loss, uncertainty), maximum_loss)
        if loss <= tolerance + 1e-12:
            eligible.append(trial)
    return eligible


def _trial_meaningfully_better(
    candidate: Trial,
    baseline: Trial,
    *,
    classification_negligible_gain: float,
    regression_negligible_relative_gain: float,
    uncertainty_multiplier: float,
    recent: bool = False,
) -> bool:
    """Return whether ``candidate`` clears signal and uncertainty thresholds."""

    candidate_score = candidate.validation_score
    baseline_score = baseline.validation_score
    candidate_error = candidate.validation_standard_error
    baseline_error = baseline.validation_standard_error
    if recent:
        if candidate.cutoff_scores and candidate.cutoff_scores[-1] is not None:
            candidate_score = float(candidate.cutoff_scores[-1])
            candidate_error = candidate.cutoff_standard_errors[-1]
        if baseline.cutoff_scores and baseline.cutoff_scores[-1] is not None:
            baseline_score = float(baseline.cutoff_scores[-1])
            baseline_error = baseline.cutoff_standard_errors[-1]
    uncertainty = (
        uncertainty_multiplier * (candidate_error**2 + baseline_error**2) ** 0.5
    )
    if baseline.metric in {"roc_auc", "accuracy"}:
        improvement = candidate_score - baseline_score
        required = max(classification_negligible_gain, uncertainty)
    else:
        improvement = baseline_score - candidate_score
        required = max(
            abs(baseline_score) * regression_negligible_relative_gain,
            uncertainty,
        )
    return improvement > required


def _select_temporal_rerank_trials(
    successful: Sequence[Trial],
    *,
    classification_negligible_gain: float,
    regression_negligible_relative_gain: float,
    complexity_uncertainty_multiplier: float,
    complexity_selector: bool,
    recent_classification_veto: float = 0.01,
    recent_regression_relative_veto: float = 0.01,
    complexity_classification_loss_cap: float = 0.01,
    complexity_regression_relative_loss_cap: float = 0.01,
    drastic_feature_growth: float = 2.0,
    nested_depth_safety: bool = False,
    practical_recent_ties: bool = False,
) -> tuple[Trial, Trial, Trial]:
    """Select on recent-fold evidence with uncertainty and stability guards.

    The latest fold first defines an uncertainty-equivalent candidate set. A
    graph is excluded only when the recent leader clears both the practical
    veto threshold and their joint standard error. Stability-adjusted
    walk-forward performance breaks ties inside that set. Complexity may then
    replace that winner only with a practical tie at least
    ``drastic_feature_growth`` times smaller. Practical ties use paired-fold
    uncertainty but are hard-capped at the configured metric loss.

    With ``nested_depth_safety``, a deeper selected graph can be replaced by
    its otherwise identical shallower ancestor when the deeper graph has no
    meaningful direct recent-fold gain, the ancestor is a practical
    walk-forward tie, and even the global recent leader is not two standard
    errors better than the ancestor. This protects a nested control from a
    marginal veto by an unrelated third finalist without weakening the
    ordinary recent-fold gate for different feature families.

    With ``practical_recent_ties``, historical stability can break a recent
    uncertainty tie only if its recent-fold loss also fits the practical
    metric cap. A large marginal standard error is not permission to give up
    arbitrarily much current predictive performance. The same recent loss
    bound also applies to subsequent complexity/depth substitutions.
    """

    if not successful:
        raise ValueError("At least one successful rerank trial is required")

    def walk_forward_key(trial: Trial) -> tuple[Any, ...]:
        return (
            -trial.objective_score,
            trial.feature_count,
            trial.config.complexity,
            graph_config_sort_key(trial.config),
        )

    def recent_score(trial: Trial) -> float:
        if trial.cutoff_scores and trial.cutoff_scores[-1] is not None:
            return float(trial.cutoff_scores[-1])
        return trial.validation_score

    walk_forward_best = min(successful, key=walk_forward_key)
    recent_best = min(
        successful,
        key=lambda trial: (
            -_objective_score(trial.metric, recent_score(trial)),
            trial.feature_count,
            trial.config.complexity,
            graph_config_sort_key(trial.config),
        ),
    )

    def practical_recent_tie(trial: Trial) -> bool:
        if not practical_recent_ties:
            return True
        leader_score = recent_score(recent_best)
        candidate_score = recent_score(trial)
        if recent_best.metric in {"roc_auc", "accuracy"}:
            loss = leader_score - candidate_score
            cap = complexity_classification_loss_cap
        else:
            loss = candidate_score - leader_score
            cap = abs(leader_score) * complexity_regression_relative_loss_cap
        return loss <= cap + 1e-12

    recent_safe: list[Trial] = []
    for trial in successful:
        if practical_recent_tie(trial) and not _trial_meaningfully_better(
            recent_best,
            trial,
            classification_negligible_gain=max(
                classification_negligible_gain,
                recent_classification_veto,
            ),
            regression_negligible_relative_gain=max(
                regression_negligible_relative_gain,
                recent_regression_relative_veto,
            ),
            uncertainty_multiplier=complexity_uncertainty_multiplier,
            recent=True,
        ):
            recent_safe.append(trial)

    recent_safe_best = min(recent_safe, key=walk_forward_key)
    recommended = recent_safe_best
    if complexity_selector:
        uncertainty_equivalent = _temporal_complexity_equivalents(
            recent_safe,
            recent_safe_best,
            classification_negligible_gain=classification_negligible_gain,
            regression_negligible_relative_gain=(regression_negligible_relative_gain),
            complexity_uncertainty_multiplier=complexity_uncertainty_multiplier,
            classification_loss_cap=complexity_classification_loss_cap,
            regression_relative_loss_cap=(complexity_regression_relative_loss_cap),
        )
        drastic_alternatives = [
            trial
            for trial in uncertainty_equivalent
            if trial.feature_count * drastic_feature_growth
            <= recent_safe_best.feature_count
        ]
        if drastic_alternatives:
            recommended = min(
                drastic_alternatives,
                key=lambda trial: (
                    trial.feature_count,
                    trial.config.complexity,
                    -trial.objective_score,
                    graph_config_sort_key(trial.config),
                ),
            )
        if nested_depth_safety and recommended.config.depth > 1:
            nested_equivalents = _temporal_complexity_equivalents(
                successful,
                recommended,
                classification_negligible_gain=classification_negligible_gain,
                regression_negligible_relative_gain=(
                    regression_negligible_relative_gain
                ),
                complexity_uncertainty_multiplier=complexity_uncertainty_multiplier,
                classification_loss_cap=complexity_classification_loss_cap,
                regression_relative_loss_cap=complexity_regression_relative_loss_cap,
            )
            shallow_ancestors = [
                trial
                for trial in nested_equivalents
                if practical_recent_tie(trial)
                and trial.config.depth < recommended.config.depth
                and replace(recommended.config, depth=trial.config.depth)
                == trial.config
                and trial.feature_count * drastic_feature_growth
                <= recommended.feature_count
                and not _trial_meaningfully_better(
                    recommended,
                    trial,
                    classification_negligible_gain=max(
                        classification_negligible_gain,
                        recent_classification_veto,
                    ),
                    regression_negligible_relative_gain=max(
                        regression_negligible_relative_gain,
                        recent_regression_relative_veto,
                    ),
                    uncertainty_multiplier=complexity_uncertainty_multiplier,
                    recent=True,
                )
                and not _trial_meaningfully_better(
                    recent_best,
                    trial,
                    classification_negligible_gain=max(
                        classification_negligible_gain,
                        recent_classification_veto,
                    ),
                    regression_negligible_relative_gain=max(
                        regression_negligible_relative_gain,
                        recent_regression_relative_veto,
                    ),
                    uncertainty_multiplier=2.0 * complexity_uncertainty_multiplier,
                    recent=True,
                )
            ]
            if shallow_ancestors:
                recommended = min(
                    shallow_ancestors,
                    key=lambda trial: (
                        trial.feature_count,
                        trial.config.depth,
                        -trial.objective_score,
                        graph_config_sort_key(trial.config),
                    ),
                )
    return walk_forward_best, recent_safe_best, recommended


def _build_graph(
    workspace: _Workspace,
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    root_name: str,
    config: GraphConfig,
    *,
    cut_date: datetime,
    compute_period_days: int,
    excluded_columns: set[str],
    root_entity_keys: pd.DataFrame | None = None,
    graph_labels: GraphLabels | None = None,
    execution_plan: Mapping[str, Any] | None = None,
    train: bool = True,
    infer_ts_periods: bool = True,
):
    from graphreduce.enum import ComputeLayerEnum, PeriodUnit
    from graphreduce.graph_reduce import GraphReduce
    from graphreduce.models import sqlop
    from graphreduce.node import DuckdbNode, GraphReduceNode
    from graphreduce.enum import SQLOpType

    tables, relationships = _topology_for_config(tables, relationships, config)

    if config.published_cutoff_compatibility and any(
        table.cutoff_boundary == "exact" for table in tables.values()
    ):
        raise ValueError(
            "Exact feature snapshots require unshifted cutoffs; set "
            "published_cutoff_compatibility=False for this configuration"
        )

    if not config.include_target_history and TARGET_HISTORY_TABLE in tables:
        tables = {
            name: table
            for name, table in tables.items()
            if name != TARGET_HISTORY_TABLE
        }
        relationships = tuple(
            relationship
            for relationship in relationships
            if relationship.parent != TARGET_HISTORY_TABLE
            and relationship.child != TARGET_HISTORY_TABLE
        )

    # GraphReduce owns cutoff-boundary semantics. Keep KurveRSC runnable with
    # released GraphReduce versions from before that node argument existed by
    # adapting only the generated DuckDB upper-bound predicate. This branch can
    # be removed after KurveRSC's minimum GraphReduce version includes the
    # native API.
    from inspect import signature

    node_parameters = signature(GraphReduceNode.__init__).parameters
    if (
        config.base_distribution_features is True
        and "base_distribution_features" not in node_parameters
    ):
        raise RuntimeError(
            "base_distribution_features=True requires a GraphReduce version "
            "that supports base CDF features"
        )
    if "cutoff_boundary" in node_parameters:
        cutoff_node_class = DuckdbNode
    else:

        class _CutoffAwareDuckdbNode(DuckdbNode):
            def __init__(
                self, *args: Any, cutoff_boundary: str = "strict", **kwargs: Any
            ) -> None:
                self.cutoff_boundary = cutoff_boundary
                super().__init__(*args, **kwargs)

            def prep_for_features(self, allow_null: bool = False):
                operations = super().prep_for_features(allow_null=allow_null)
                if self.cutoff_boundary not in {"inclusive", "exact"} or not isinstance(
                    operations, list
                ):
                    return operations
                for operation in operations:
                    if operation.optype == SQLOpType.where and " < " in operation.opval:
                        operator = " = " if self.cutoff_boundary == "exact" else " <= "
                        operation.opval = operation.opval.replace(" < ", operator, 1)
                        break
                return operations

        # Older GraphReduce dialect dispatch uses the concrete class name.
        _CutoffAwareDuckdbNode.__name__ = "DuckdbNode"
        cutoff_node_class = _CutoffAwareDuckdbNode

    root_key_columns = _key_parts(tables[root_name].key)
    root_source = (
        workspace.entity_filtered_view(
            root_name,
            key_columns=root_key_columns,
            entity_keys=root_entity_keys,
        )
        if root_entity_keys is not None
        else workspace.views[root_name]
    )
    # KurveRSC 0.1.1 represented an inclusive relational cutoff as a strict
    # GraphReduce cutoff shifted forward by one microsecond.  The row set is
    # equivalent to ``date <= cutoff``, but GraphReduce also uses ``cut_date``
    # as the anchor for window thresholds.  Replacing the representation with
    # a native inclusive predicate therefore changed the learned feature plan
    # (and the published benchmark representation) even though the boundary
    # looked mathematically equivalent.  Published anchor configurations keep
    # the original anchor semantics; modern configurations use the exact
    # native cutoff.  Target history is never allowed onto the compatibility
    # path because it requires a strict join against the unshifted cutoff.
    graph_cut_date = pd.Timestamp(cut_date)
    explicit_clock_boundaries = any(table.date is not None and
        table.cutoff_boundary in {"strict","inclusive","exact"} for table in tables.values())
    if (config.published_cutoff_compatibility and graph_labels is None
            and not explicit_clock_boundaries):
        graph_cut_date += pd.Timedelta(microseconds=1)
    graph_cut_date = graph_cut_date.to_pydatetime()
    nodes = {}
    for index, (name, table) in enumerate(tables.items()):
        available = workspace.columns[name]
        selected = list(table.columns) if table.columns is not None else list(available)
        selected = [column for column in selected if column not in excluded_columns
                    and column != table.available_at]
        for key in _key_parts(table.key) if table.key is not None else []:
            if key not in selected:
                selected.insert(0, key)
        if table.date is not None and table.date not in selected:
            selected.append(table.date)
        for context_key in table.context_keys:
            if context_key not in selected:
                selected.append(context_key)
        missing_columns = set(selected) - set(available)
        if missing_columns:
            raise ValueError(
                f"Table {name!r} is missing configured columns: "
                f"{sorted(missing_columns)}"
            )
        prefix = table.prefix or f"{_slug(name, f'n{index}')[:10]}{index}"
        # GraphReduce nodes currently represent this setting as an integer.
        # A very large slice bound preserves KurveRSC's public ``None`` = no
        # cap semantics without changing GraphReduce itself.
        uncapped_budget = 2_147_483_647
        node_source = root_source if name == root_name else workspace.views[name]
        if table.available_at is not None or (table.date and table.cutoff_boundary in {"strict", "inclusive"}):
            if table.available_at and table.available_at not in available:
                raise ValueError(f"Table {name!r} lacks declared availability clock {table.available_at!r}")
            # Availability is independent of event age/windows and is never a
            # model payload. Filter before annotation, not after aggregation.
            original_source = node_source
            node_source = f"{workspace.views[name]}__authorized"
            stamp = pd.Timestamp(cut_date).isoformat(sep=" ").replace("'", "''")
            predicates = []
            if table.available_at:
                predicates.append(f"{_quote_identifier(table.available_at)} <= TIMESTAMP '{stamp}'")
            if table.date and table.cutoff_boundary=="strict":
                # Explicit boundaries are data semantics, not a search knob.
                # A legacy offset candidate must not expose rows at/after
                # an exact timestamp simply by moving GraphReduce's clock.
                predicates.append(f"{_quote_identifier(table.date)} < TIMESTAMP '{stamp}'")
            elif table.date and table.cutoff_boundary=="inclusive":
                predicates.append(f"{_quote_identifier(table.date)} <= TIMESTAMP '{stamp}'")
            workspace.connection.execute(
                f"CREATE OR REPLACE TEMP VIEW {_quote_identifier(node_source)} AS "
                f"SELECT * FROM {_quote_identifier(original_source)} WHERE "
                +" AND ".join(predicates)
            )
            if node_source not in workspace._auxiliary_views:
                workspace._auxiliary_views.append(node_source)
        node = cutoff_node_class(
            fpath=node_source,
            prefix=prefix,
            pk=table.key,
            date_key=table.date,
            columns=selected,
            categorical_cardinality_threshold=20,
            categorical_top_k=5,
            auto_text_features=config.auto_text_features,
            auto_annotate_features=config.auto_annotate_features,
            auto_annotate_max_categorical_columns=10,
            auto_annotate_max_gated_numeric_cols=4,
            auto_annotate_gated_numeric_top_k=3,
            context_keys=table.context_keys,
            auto_base_predicate_max=table.auto_base_predicate_max,
            base_predicate_windows=table.base_predicate_windows,
            base_predicates=table.base_predicates,
            cutoff_boundary=(
                table.cutoff_boundary
                or (
                    "strict"
                    if graph_labels is not None or config.published_cutoff_compatibility
                    else "inclusive"
                )
            ),
            feature_families=config.feature_families,
            feature_family_max_columns=(
                config.feature_family_max_columns
                if config.feature_family_max_columns is not None
                else uncapped_budget
            ),
            feature_family_max_features_per_column=(
                config.feature_family_max_features_per_column
            ),
            feature_propagation_max_functions_per_column=(
                config.feature_propagation_max_functions_per_column
            ),
            # Numeric distributions now belong to GraphReduce's base family,
            # not to another searched family/configuration axis. Keep the
            # published-reference representation unchanged, and retain support
            # for older GraphReduce versions that do not expose this setting.
            **(
                {
                    "base_distribution_features": (
                        config.base_distribution_features
                        if config.base_distribution_features is not None
                        else not config.published_cutoff_compatibility
                    )
                }
                if "base_distribution_features" in node_parameters
                else {}
            ),
        )
        if name == root_name:
            # Filter the source during do_data, before schema inference,
            # annotation, or joins. Merely setting date_key is insufficient:
            # GraphReduce only applies automatic cutoff predicates while
            # reducing relation nodes.
            select_expressions = [
                f"{node.render_identifier(column)} as "
                f"{node.render_identifier(node.colabbr(column))}"
                for column in selected
            ]
            root_filters = [
                sqlop(
                    optype=SQLOpType.where,
                    opval=f"{node.render_identifier(key)} IS NOT NULL",
                )
                for key in _key_parts(table.key)
            ]
            if table.date is not None:
                cutoff = pd.Timestamp(cut_date).isoformat(sep=" ")
                root_filters.append(
                    sqlop(
                        optype=SQLOpType.where,
                        opval=(
                            f"{node.render_identifier(table.date)} "
                            f"<= TIMESTAMP '{cutoff}'"
                        ),
                    )
                )
            node.do_data_ops = [
                sqlop(
                    optype=SQLOpType.select,
                    opval=",".join(select_expressions),
                ),
                *root_filters,
            ]
            # Also push the invariant through GraphReduce's public filtering
            # stage. These expressions reference the aliases produced by
            # do_data, matching custom production GraphReduce nodes.
            node.do_filters_ops = [
                sqlop(
                    optype=SQLOpType.where,
                    opval=(f"{node.render_identifier(node.colabbr(key))} IS NOT NULL"),
                )
                for key in _key_parts(table.key)
            ]
            if table.date is not None:
                node.do_filters_ops.append(
                    sqlop(
                        optype=SQLOpType.where,
                        opval=(
                            f"{node.render_identifier(node.colabbr(table.date))} "
                            f"<= TIMESTAMP '{cutoff}'"
                        ),
                    )
                )
        nodes[name] = node
    if graph_labels is not None:
        if graph_labels.table not in nodes:
            raise ValueError(
                f"GraphLabels references unknown table {graph_labels.table!r}"
            )
        label_node = nodes[graph_labels.table]
        if graph_labels.field not in workspace.columns[graph_labels.table]:
            raise ValueError(
                f"GraphLabels field {graph_labels.field!r} is missing from "
                f"table {graph_labels.table!r}"
            )
    else:
        label_node = None
    graph = GraphReduce(
        name=f"kurversc-{config.depth}-{'-'.join(config.feature_families)}",
        parent_node=nodes[root_name],
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=workspace.connection,
        cut_date=graph_cut_date,
        compute_period_val=compute_period_days,
        compute_period_unit=PeriodUnit.day,
        infer_ts_periods=infer_ts_periods and execution_plan is None,
        date_filters_on_agg=True,
        label_node=label_node,
        label_field=graph_labels.field if graph_labels else None,
        label_operation=graph_labels.operation if graph_labels else None,
        label_period_val=graph_labels.period_days if graph_labels else None,
        label_period_unit=PeriodUnit.day if graph_labels else None,
        train=train,
        **config.graphreduce_kwargs(),
    )
    for node in nodes.values():
        graph.add_node(node)
    for relationship in relationships:
        if relationship.parent not in nodes or relationship.child not in nodes:
            raise ValueError(
                f"Unknown relationship table: {relationship.parent} -> {relationship.child}"
            )
        graph.add_entity_edge(
            nodes[relationship.parent],
            nodes[relationship.child],
            parent_key=relationship.parent_key,
            relation_key=relationship.child_key,
            reduce=relationship.reduce,
        )
    if execution_plan is not None:
        # Date-dependent replay needs hydrated node periods before GraphReduce
        # can rebind cut, lookback, time-series, and future-label literals.
        graph.hydrate_graph_attrs()
        planned_periods = execution_plan.get("kurversc_ts_periods", {})
        for node in graph.nodes():
            if node.prefix in planned_periods:
                node.ts_periods = list(planned_periods[node.prefix])
        from .plan_dates import apply_graph_plan

        apply_graph_plan(
            graph,
            _prepare_execution_plan(
                execution_plan,
                root_prefix=nodes[root_name].prefix,
                cut_date=cut_date,
            )
        )
    return graph


def _topology_for_config(
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    config: GraphConfig,
) -> tuple[dict[str, Table], tuple[Relationship, ...]]:
    """Return the bounded path-unrolled topology requested by ``config``.

    Tables without RelBench path metadata are ordinary user-supplied tables and
    remain available under every topology setting. RelBench adapters build the
    occurrence-two superset once; individual graph configurations can then
    compare legacy acyclic paths with bounded peer/self-relational paths using
    identical source data.
    """

    limit = config.max_base_table_occurrences

    def allowed(table: Table) -> bool:
        path = table.path_base_tables
        return not path or max(path.count(name) for name in set(path)) <= limit

    selected = {name: table for name, table in tables.items() if allowed(table)}
    selected_relationships = tuple(
        relationship
        for relationship in relationships
        if relationship.parent in selected and relationship.child in selected
    )
    return selected, selected_relationships


def _materialize_at_cutoff(
    workspace: _Workspace,
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    root_name: str,
    config: GraphConfig,
    labels: pd.DataFrame,
    label_spec: Labels,
    *,
    cut_date: datetime,
    compute_period_days: int,
    verbose: bool,
    execution_plan: Mapping[str, Any] | None = None,
    frozen_plan_sink: list[dict[str, Any]] | None = None,
    train: bool = True,
    planning_labels: pd.DataFrame | None = None,
    infer_ts_periods: bool = True,
) -> pd.DataFrame:
    batch_rows = config.materialization_batch_rows
    if batch_rows is not None and len(labels) > batch_rows:
        if config.legacy_training_row_order:
            raise ValueError("Batched materialization cannot preserve legacy training row order")
        ordered_labels, order_column = _ordered_materialization_labels(
            labels, label_spec
        )
        ordered_labels = ordered_labels.drop(columns=order_column)
        parts: list[pd.DataFrame] = []
        batch_plan = execution_plan
        plan_sink = frozen_plan_sink if frozen_plan_sink is not None else []
        if batch_plan is None:
            identity_columns = [
                column
                for column in (*_key_parts(label_spec.key), label_spec.timestamp)
                if column is not None and column in ordered_labels.columns
            ]
            if identity_columns:
                identity_hashes = pd.util.hash_pandas_object(
                    ordered_labels[identity_columns], index=False
                ).to_numpy(dtype="uint64")
                plan_positions = np.argpartition(
                    identity_hashes, batch_rows - 1
                )[:batch_rows]
                plan_positions.sort()
            else:
                plan_positions = np.linspace(
                    0, len(ordered_labels) - 1, batch_rows, dtype=int
                )
            plan_labels = ordered_labels.iloc[plan_positions]
            _materialize_at_cutoff(
                workspace, tables, relationships, root_name, config,
                plan_labels, label_spec,
                cut_date=cut_date, compute_period_days=compute_period_days,
                verbose=verbose, execution_plan=None,
                frozen_plan_sink=plan_sink,
                train=train, planning_labels=plan_labels,
                infer_ts_periods=infer_ts_periods,
            )
            batch_plan = plan_sink[-1]
        for start in range(0, len(ordered_labels), batch_rows):
            chunk = ordered_labels.iloc[start : start + batch_rows]
            parts.append(
                _materialize_at_cutoff(
                    workspace, tables, relationships, root_name, config,
                    chunk, label_spec,
                    cut_date=cut_date, compute_period_days=compute_period_days,
                    verbose=verbose, execution_plan=batch_plan,
                    train=train, planning_labels=chunk,
                    infer_ts_periods=infer_ts_periods,
                )
            )
        return pd.concat(parts, ignore_index=True)
    excluded = {label_spec.target}
    if label_spec.split:
        excluded.add(label_spec.split)
    if label_spec.timestamp:
        excluded.add(label_spec.timestamp)
    if label_spec.available_at:
        excluded.add(label_spec.available_at)
    root = tables[root_name]
    root_keys = _key_parts(root.key)
    label_keys = _key_parts(label_spec.key)
    entity_labels = planning_labels if planning_labels is not None else labels
    root_entity_keys = entity_labels.loc[:, label_keys].rename(
        columns=dict(zip(label_keys, root_keys, strict=True))
    )
    if config.root_reachable_materialization and "context" in config.feature_families:
        raise ValueError(
            "Root-reachable materialization changes cross-root context features"
        )
    active_relationships = (
        _topology_for_config(tables, relationships, config)[1]
        if config.root_reachable_materialization
        else ()
    )
    scope = (
        workspace.reachable_sources(
            tables, root_name, active_relationships, root_keys, root_entity_keys
        )
        if config.root_reachable_materialization
        else nullcontext()
    )
    with ExitStack() as stack:
        stack.enter_context(scope)
        if not verbose:
            sink = io.StringIO()
            stack.enter_context(redirect_stdout(sink))
            stack.enter_context(redirect_stderr(sink))
        graph = _build_graph(
            workspace,
            tables,
            relationships,
            root_name,
            config,
            cut_date=cut_date,
            compute_period_days=compute_period_days,
            excluded_columns=excluded,
            root_entity_keys=root_entity_keys,
            execution_plan=execution_plan,
            train=train,
            infer_ts_periods=infer_ts_periods,
        )
        try:
            graph.do_transformations_sql()
            if frozen_plan_sink is not None:
                frozen_plan_sink.append(_freeze_execution_plan(graph, cut_date))
            features = workspace.connection.sql(
                f"SELECT * FROM {_quote_identifier(graph.parent_node._cur_data_ref)}"
            ).to_df()
        finally:
            graph._clean_refs()
    # GraphReduce intentionally keeps primary-key names unprefixed, while
    # custom node implementations may prefix them. Accept either convention.
    feature_keys = [
        key if key in features.columns else f"{graph.parent_node.prefix}_{key}"
        for key in root_keys
    ]
    def with_peer_context(materialized: pd.DataFrame) -> pd.DataFrame:
        provider = label_spec.peer_context
        if provider is None:
            return materialized
        if label_spec.timestamp is None:
            raise ValueError("Peer context requires temporal labels")
        keys = [*_key_parts(label_spec.key), label_spec.timestamp]
        query = materialized[keys].reset_index(drop=True)
        addition = provider.transform(query)
        pd.testing.assert_frame_equal(addition[keys], query)
        columns = [column for column in addition if column not in keys]
        overlap = set(columns) & set(materialized.columns)
        if overlap:
            raise ValueError(f"Peer context feature names collide: {sorted(overlap)}")
        return pd.concat(
            [materialized.reset_index(drop=True), addition[columns].reset_index(drop=True)],
            axis=1,
        )

    if config.legacy_training_row_order and train:
        # The published 0.1.1 representation fed CatBoost rows in the stable
        # GraphReduce feature-merge order.  Reordering labeled frames back to
        # label-source order is mathematically equivalent, but CatBoost's
        # seeded permutation machinery is row-order-sensitive.  Preserve that
        # training/validation contract for release-compatible anchors while
        # retaining explicit source-row order below for inference, where the
        # output must align with the caller's prediction rows.
        return with_peer_context(features.merge(
            labels,
            how="inner",
            left_on=feature_keys,
            right_on=label_keys,
            validate="one_to_many",
        ))
    ordered_labels, order_column = _ordered_materialization_labels(
        labels,
        label_spec,
    )
    materialized = features.merge(
        ordered_labels,
        how="inner",
        left_on=feature_keys,
        right_on=label_keys,
        validate="one_to_many",
    )
    if not train:
        # A strict cutoff can make a requested root's payload unavailable.
        # Keep the request with missing features instead of exposing a future
        # source row or silently dropping its prediction.
        missing = ordered_labels.loc[
            ~ordered_labels[order_column].isin(materialized[order_column])
        ]
        if not missing.empty:
            unavailable = features.iloc[:0].merge(
                missing, how="right", left_on=feature_keys,
                right_on=label_keys, validate="one_to_many",
            )
            payload = set(features.columns) - set(feature_keys) - set(ordered_labels.columns)
            for column in payload:
                if pd.api.types.is_integer_dtype(materialized[column].dtype) or pd.api.types.is_bool_dtype(materialized[column].dtype):
                    materialized[column] = materialized[column].astype(object)
            materialized = pd.concat([materialized, unavailable], ignore_index=True)
    return with_peer_context(
        materialized.sort_values(order_column, kind="stable")
        .drop(columns=order_column)
        .reset_index(drop=True)
    )


def _materialize_split(
    workspace: _Workspace,
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    root_name: str,
    config: GraphConfig,
    labels: pd.DataFrame,
    label_spec: Labels,
    *,
    compute_period_days: int,
    verbose: bool,
    split_marker: str | None = None,
    execution_plan: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frozen_plans: list[dict[str, Any]] = []
    selected_plan = (
        copy.deepcopy(dict(execution_plan)) if execution_plan is not None else None
    )
    if label_spec.timestamp:
        parts = []
        if split_marker is not None:
            train_labels = labels.loc[labels[split_marker] == "train"]
            validation_labels = labels.loc[labels[split_marker] == "validation"]
        else:
            train_labels = labels
            validation_labels = labels.iloc[0:0]
        train_groups = list(
            train_labels.groupby(label_spec.timestamp, sort=True, dropna=False)
        )
        if not train_groups:
            raise ValueError("At least one training cutoff is required")
        # Fit feature operations once on the latest training cutoff, then
        # replay those exact operations everywhere else.
        anchor_timestamp, anchor_labels = train_groups[-1]
        if pd.isna(anchor_timestamp):
            raise ValueError("Label timestamps must not be missing")
        parts.append(
            _materialize_at_cutoff(
                workspace,
                tables,
                relationships,
                root_name,
                config,
                anchor_labels,
                label_spec,
                cut_date=pd.Timestamp(anchor_timestamp).to_pydatetime(),
                compute_period_days=compute_period_days,
                verbose=verbose,
                execution_plan=selected_plan,
                frozen_plan_sink=None if selected_plan is not None else frozen_plans,
                planning_labels=train_labels,
            )
        )
        selected_plan = selected_plan or frozen_plans[0]
        replay_groups = [
            *train_groups[:-1],
            *list(
                validation_labels.groupby(label_spec.timestamp, sort=True, dropna=False)
            ),
        ]
        for timestamp, cutoff_labels in replay_groups:
            if pd.isna(timestamp):
                raise ValueError("Label timestamps must not be missing")
            cutoff = pd.Timestamp(timestamp).to_pydatetime()
            parts.append(
                _materialize_at_cutoff(
                    workspace,
                    tables,
                    relationships,
                    root_name,
                    config,
                    cutoff_labels,
                    label_spec,
                    cut_date=cutoff,
                    compute_period_days=compute_period_days,
                    verbose=verbose,
                    execution_plan=selected_plan,
                )
            )
        return pd.concat(parts, ignore_index=True), selected_plan
    if split_marker is None:
        train_labels = labels
        validation_labels = labels.iloc[0:0]
    else:
        train_labels = labels.loc[labels[split_marker] == "train"]
        validation_labels = labels.loc[labels[split_marker] == "validation"]
    anchor = _materialize_at_cutoff(
        workspace,
        tables,
        relationships,
        root_name,
        config,
        train_labels,
        label_spec,
        cut_date=_UNTIMED_LABEL_CUTOFF,
        compute_period_days=compute_period_days,
        verbose=verbose,
        execution_plan=selected_plan,
        frozen_plan_sink=None if selected_plan is not None else frozen_plans,
    )
    selected_plan = selected_plan or frozen_plans[0]
    if validation_labels.empty:
        return anchor, selected_plan
    validation = _materialize_at_cutoff(
        workspace,
        tables,
        relationships,
        root_name,
        config,
        validation_labels,
        label_spec,
        cut_date=_UNTIMED_LABEL_CUTOFF,
        compute_period_days=compute_period_days,
        verbose=verbose,
        execution_plan=selected_plan,
    )
    return pd.concat([anchor, validation], ignore_index=True), selected_plan


def _materialize_graph_labels_at_cutoff(
    workspace: _Workspace,
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    root_name: str,
    config: GraphConfig,
    graph_labels: GraphLabels,
    *,
    cut_date: datetime,
    split_value: str,
    split_marker: str,
    cutoff_marker: str,
    compute_period_days: int,
    verbose: bool,
    execution_plan: Mapping[str, Any] | None = None,
    frozen_plan_sink: list[dict[str, Any]] | None = None,
    train: bool = True,
    infer_ts_periods: bool = True,
) -> pd.DataFrame:
    """Run one graph whose target is produced by GraphReduce itself."""

    with ExitStack() as stack:
        if not verbose:
            sink = io.StringIO()
            stack.enter_context(redirect_stdout(sink))
            stack.enter_context(redirect_stderr(sink))
        graph = _build_graph(
            workspace,
            tables,
            relationships,
            root_name,
            config,
            cut_date=cut_date,
            compute_period_days=compute_period_days,
            excluded_columns={graph_labels.target},
            graph_labels=graph_labels,
            execution_plan=execution_plan,
            train=train,
            infer_ts_periods=infer_ts_periods,
        )
        label_node = graph.label_node
        generated_target = f"{label_node.colabbr(graph_labels.field)}_label"
        try:
            graph.do_transformations_sql()
            if frozen_plan_sink is not None:
                frozen_plan_sink.append(_freeze_execution_plan(graph, cut_date))
            frame = workspace.connection.sql(
                f"SELECT * FROM {_quote_identifier(graph.parent_node._cur_data_ref)}"
            ).to_df()
        finally:
            graph._clean_refs()
    if generated_target not in frame.columns:
        if not train:
            frame[cutoff_marker] = pd.Timestamp(cut_date)
            frame[split_marker] = split_value
            return frame
        raise ValueError(
            "GraphReduce did not attach its generated target column "
            f"{generated_target!r}; verify that the label table is connected "
            "to the parent and the relationship is reducible"
        )
    frame = frame.rename(columns={generated_target: graph_labels.target})
    if graph_labels.operation.lower() in {"bool", "count", "sum"}:
        frame[graph_labels.target] = frame[graph_labels.target].fillna(0)
    if graph_labels.operation.lower() == "bool":
        frame[graph_labels.target] = (
            pd.to_numeric(frame[graph_labels.target], errors="raise") > 0
        ).astype("int8")
    frame[cutoff_marker] = pd.Timestamp(cut_date)
    frame[split_marker] = split_value
    root = tables[root_name]
    sort_columns = []
    for key in _key_parts(root.key):
        candidate = key if key in frame.columns else f"{graph.parent_node.prefix}_{key}"
        if candidate in frame.columns:
            sort_columns.append(candidate)
    sort_columns.extend((cutoff_marker, split_marker, graph_labels.target))
    return frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def _materialize_graph_label_splits(
    workspace: _Workspace,
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    root_name: str,
    config: GraphConfig,
    graph_labels: GraphLabels,
    *,
    split_marker: str,
    cutoff_marker: str,
    compute_period_days: int,
    verbose: bool,
    train_cutoffs: Sequence[Any] | None = None,
    execution_plan: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    frozen_plans: list[dict[str, Any]] = []
    selected_plan = (
        copy.deepcopy(dict(execution_plan)) if execution_plan is not None else None
    )
    selected_train_cutoffs = tuple(train_cutoffs or graph_labels.train_cutoffs)
    if not selected_train_cutoffs:
        raise ValueError("At least one GraphLabels training frame is required")
    anchor = max(selected_train_cutoffs)
    frames.append(
        _materialize_graph_labels_at_cutoff(
            workspace,
            tables,
            relationships,
            root_name,
            config,
            graph_labels,
            cut_date=pd.Timestamp(anchor).to_pydatetime(),
            split_value="train",
            split_marker=split_marker,
            cutoff_marker=cutoff_marker,
            compute_period_days=compute_period_days,
            verbose=verbose,
            execution_plan=selected_plan,
            frozen_plan_sink=None if selected_plan is not None else frozen_plans,
        )
    )
    selected_plan = selected_plan or frozen_plans[0]
    for split_value, cutoffs in (
        (
            "train",
            tuple(cutoff for cutoff in selected_train_cutoffs if cutoff != anchor),
        ),
        ("validation", graph_labels.validation_cutoffs),
    ):
        for cutoff in cutoffs:
            frames.append(
                _materialize_graph_labels_at_cutoff(
                    workspace,
                    tables,
                    relationships,
                    root_name,
                    config,
                    graph_labels,
                    cut_date=pd.Timestamp(cutoff).to_pydatetime(),
                    split_value=split_value,
                    split_marker=split_marker,
                    cutoff_marker=cutoff_marker,
                    compute_period_days=compute_period_days,
                    verbose=verbose,
                    execution_plan=selected_plan,
                )
            )
    return pd.concat(frames, ignore_index=True), selected_plan


def _materialize_external_test(
    workspace: _Workspace,
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    root_name: str,
    config: GraphConfig,
    labels: pd.DataFrame,
    label_spec: Labels,
    *,
    execution_plan: Mapping[str, Any],
    compute_period_days: int,
    verbose: bool,
) -> pd.DataFrame:
    """Build external-label test features without learning any operations."""

    if labels.empty:
        return pd.DataFrame()
    if label_spec.timestamp:
        parts = []
        groups = labels.groupby(label_spec.timestamp, sort=True, dropna=False)
        for timestamp, cutoff_labels in groups:
            if pd.isna(timestamp):
                raise ValueError("Test label timestamps must not be missing")
            parts.append(
                _materialize_at_cutoff(
                    workspace,
                    tables,
                    relationships,
                    root_name,
                    config,
                    cutoff_labels,
                    label_spec,
                    cut_date=pd.Timestamp(timestamp).to_pydatetime(),
                    compute_period_days=compute_period_days,
                    verbose=verbose,
                    execution_plan=execution_plan,
                    train=False,
                )
            )
        return pd.concat(parts, ignore_index=True)
    return _materialize_at_cutoff(
        workspace,
        tables,
        relationships,
        root_name,
        config,
        labels,
        label_spec,
        cut_date=datetime.now(),
        compute_period_days=compute_period_days,
        verbose=verbose,
        execution_plan=execution_plan,
        train=False,
    )


def _materialize_graph_test(
    workspace: _Workspace,
    tables: Mapping[str, Table],
    relationships: Sequence[Relationship],
    root_name: str,
    config: GraphConfig,
    graph_labels: GraphLabels,
    *,
    execution_plan: Mapping[str, Any],
    split_marker: str,
    cutoff_marker: str,
    compute_period_days: int,
    verbose: bool,
) -> pd.DataFrame:
    frames = [
        _materialize_graph_labels_at_cutoff(
            workspace,
            tables,
            relationships,
            root_name,
            config,
            graph_labels,
            cut_date=pd.Timestamp(cutoff).to_pydatetime(),
            split_value="test",
            split_marker=split_marker,
            cutoff_marker=cutoff_marker,
            compute_period_days=compute_period_days,
            verbose=verbose,
            execution_plan=execution_plan,
            train=False,
        )
        for cutoff in graph_labels.test_cutoffs
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _model_exclusions(
    tables: Mapping[str, Table],
    root_name: str,
    labels_spec: Labels | None,
    graph_labels: GraphLabels | None,
    cutoff_marker: str,
) -> set[str]:
    root = tables[root_name]
    excluded = {*_key_parts(root.key)}
    if labels_spec is not None:
        excluded.update(_key_parts(labels_spec.key))
    root_prefix = root.prefix or f"{_slug(root_name, 'n0')[:10]}0"
    excluded.update(f"{root_prefix}_{key}" for key in _key_parts(root.key))
    for index,(name,table) in enumerate(tables.items()):
        prefix = table.prefix or f"{_slug(name,f'n{index}')[:10]}{index}"
        excluded.update(f"{prefix}_{column}" for column in table.model_exclusions)
    if graph_labels is not None:
        excluded.add(cutoff_marker)
    elif labels_spec is not None:
        if labels_spec.timestamp:
            excluded.add(labels_spec.timestamp)
        if labels_spec.split:
            excluded.add(labels_spec.split)
        if labels_spec.available_at:
            excluded.add(labels_spec.available_at)
    return excluded


def _fit_buffered_legacy(
    parent_node: Table | Source,
    label_node: Labels | GraphLabels | Source,
    *,
    tables: Sequence[Table | Source] | Mapping[str, Table | Source] = (),
    relationships: Sequence[Relationship | Mapping[str, Any]] = (),
    parent_key: Key | Sequence[str] | None = None,
    label_key: Key | Sequence[str] | None = None,
    target: str | None = None,
    parent_date: str | None = None,
    parent_timeless: bool = False,
    label_timestamp: str | None = None,
    split_column: str | None = None,
    task: str = "auto",
    class_labels: Sequence[Any] | None = None,
    max_depth: int = 4,
    feature_family_stages: Sequence[Sequence[str]] = DEFAULT_FAMILY_STAGES,
    auto_annotate_options: Sequence[bool] = (True, False),
    feature_family_max_columns: int | None = 4,
    sample_rows: int = 100_000,
    search_full_data: bool = False,
    search_training_frames: int = 1,
    full_training_frames: int | None = None,
    rerank_top_k: int = 3,
    validation_fraction: float = 0.2,
    compute_period_days: int = 3650,
    random_state: int = 42,
    model_backend: str = "catboost",
    final_model_backend: str | None = None,
    estimator_train_rows: int | None = None,
    model_params: Mapping[str, Any] | None = None,
    connection: duckdb.DuckDBPyConnection | None = None,
    duckdb_memory_limit: str | None = "128GB",
    duckdb_max_temp_directory_size: str | None = "128GB",
    continue_on_error: bool = True,
    verbose: bool = False,
    classification_negligible_gain: float = 0.002,
    regression_negligible_relative_gain: float = 0.005,
    drastic_feature_growth: float = 2.0,
    graph_configs: Sequence[GraphConfig] | None = None,
    infer_ts_periods: bool = False,
    auto_text_features: bool = False,
    auto_annotate_max_text_columns: int | None = None,
    high_cardinality_policy: str = "skip",
    categorical_low_cardinality_policy: str = "top_k",
    text_categorical_fallback: bool = False,
    preselected_config: GraphConfig | None = None,
    preselected_execution_plan: Mapping[str, Any] | None = None,
) -> FitResult:
    """Fit and select a GraphReduce configuration using validation performance.

    All configurations are screened on the latest training frame. By default
    the screening graph is row-sampled; ``search_full_data=True`` evaluates
    every configuration against the complete source tables instead. The
    strongest ``rerank_top_k`` screening candidates (plus the
    complexity-aware screening recommendation) are rescored on the full
    training frames before the production configuration is selected.
    A supplied ``preselected_config`` bypasses both stages; an accompanying
    ``preselected_execution_plan`` replays the already learned feature recipe.
    """

    if sample_rows < 1:
        raise ValueError("sample_rows must be positive")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if search_training_frames < 1:
        raise ValueError("search_training_frames must be positive")
    if full_training_frames is not None and full_training_frames < 1:
        raise ValueError("full_training_frames must be positive or None")
    if rerank_top_k < 1:
        raise ValueError("rerank_top_k must be positive")
    if model_backend not in {
        "catboost",
        "tabpfn_v3",
        "tabpfn_v3_5",
        "tabicl",
        "tabfm",
    }:
        raise ValueError(
            "model_backend must be 'catboost', 'tabpfn_v3', 'tabpfn_v3_5', 'tabicl', or 'tabfm'"
        )
    resolved_final_model_backend = final_model_backend or model_backend
    if resolved_final_model_backend not in {
        "catboost",
        "tabpfn_v3",
        "tabpfn_v3_5",
        "tabicl",
        "tabfm",
    }:
        raise ValueError(
            "final_model_backend must be 'catboost', 'tabpfn_v3', 'tabpfn_v3_5', 'tabicl', or 'tabfm'"
        )
    if estimator_train_rows is None and model_backend in {
        "tabpfn_v3",
        "tabpfn_v3_5",
        "tabicl",
        "tabfm",
    }:
        estimator_train_rows = 10_000
    if estimator_train_rows is not None and estimator_train_rows < 2:
        raise ValueError("estimator_train_rows must be at least 2 or None")
    if duckdb_memory_limit is not None and (
        not isinstance(duckdb_memory_limit, str) or not duckdb_memory_limit.strip()
    ):
        raise ValueError("duckdb_memory_limit must be a non-empty string or None")
    if duckdb_max_temp_directory_size is not None and (
        not isinstance(duckdb_max_temp_directory_size, str)
        or not duckdb_max_temp_directory_size.strip()
    ):
        raise ValueError(
            "duckdb_max_temp_directory_size must be a non-empty string or None"
        )
    if preselected_config is not None and graph_configs is not None:
        raise ValueError("preselected_config and graph_configs are mutually exclusive")
    if preselected_execution_plan is not None and preselected_config is None:
        raise ValueError("preselected_execution_plan requires preselected_config")
    if graph_configs is not None and len(graph_configs) == 0:
        raise ValueError("graph_configs must contain at least one configuration")
    parent = coerce_table(
        parent_node,
        key=parent_key,
        date=parent_date,
        timeless=parent_timeless,
    )
    graph_labels = label_node if isinstance(label_node, GraphLabels) else None
    labels_spec = (
        None
        if graph_labels is not None
        else coerce_labels(
            label_node,
            target=target,
            key=label_key,
            timestamp=label_timestamp,
            split=split_column,
        )
    )
    if parent.key is None or (labels_spec is not None and labels_spec.key is None):
        raise ValueError(
            "parent_key and label_key are required (directly or in Table/Labels)"
        )
    if labels_spec is not None and labels_spec.target is None:
        raise ValueError("target is required (directly or in Labels)")
    if (
        (graph_labels is not None or labels_spec.timestamp is not None)
        and parent.date is None
        and not parent.timeless
    ):
        raise ValueError(
            "parent_date is required when labels have timestamps unless the "
            "root entity is explicitly declared timeless=True; KurveRSC will "
            "not silently run a temporal task without a point-in-time root "
            "cutoff"
        )
    root_name, normalized_tables = _normalize_tables(parent, tables)
    normalized_relationships = tuple(
        coerce_relationship(item) for item in relationships
    )
    if graph_labels is not None and graph_labels.table not in normalized_tables:
        raise ValueError(
            f"GraphLabels table {graph_labels.table!r} must appear in tables"
        )
    configs = (
        (preselected_config,)
        if preselected_config is not None
        else tuple(graph_configs)
        if graph_configs is not None
        else incremental_configs(
            max_depth=max_depth,
            feature_family_stages=feature_family_stages,
            auto_annotate_options=auto_annotate_options,
            feature_family_max_columns=feature_family_max_columns,
        )
    )
    if not configs:
        raise ValueError("graph_configs must contain at least one configuration")
    configs = _with_graph_feature_options(
        configs,
        auto_text_features=auto_text_features,
        auto_annotate_max_text_columns=auto_annotate_max_text_columns,
    )
    with _connection_scope(connection) as con:
        workspace = _Workspace(con, sample_rows=sample_rows, random_state=random_state)
        try:
            for name, table in normalized_tables.items():
                workspace.add(
                    name,
                    table.search_source
                    if table.search_source is not None
                    else table.source,
                    sample=not search_full_data and table.sample_in_search,
                )
            split_marker = "__kurversc_validation_split__"
            cutoff_marker = "__kurversc_cutoff__"
            if graph_labels is not None:
                resolved_task = (
                    "classification"
                    if task == "auto" and graph_labels.operation.lower() == "bool"
                    else "regression"
                    if task == "auto"
                    else _infer_task(pd.Series(), task)
                )
                target_column = graph_labels.target
                search_labels = None
                search_graph_cutoffs = _select_training_cutoffs(
                    graph_labels.train_cutoffs, search_training_frames
                )
            else:
                label_name = "labels"
                while label_name in normalized_tables:
                    label_name += "_"
                workspace.add(
                    label_name,
                    labels_spec.search_source
                    if labels_spec.search_source is not None
                    else labels_spec.source,
                    sample=not search_full_data,
                )
                label_frame = workspace.frame(label_name)
                required = {
                    *_key_parts(labels_spec.key),
                    labels_spec.target,
                    *([labels_spec.timestamp] if labels_spec.timestamp else []),
                    *([labels_spec.split] if labels_spec.split else []),
                }
                missing = required - set(label_frame.columns)
                if missing:
                    raise ValueError(
                        f"Label source is missing columns: {sorted(missing)}"
                    )
                label_frame = label_frame.dropna(subset=[labels_spec.target])
                resolved_task = _infer_task(label_frame[labels_spec.target], task)
                train_labels, validation_labels = _split_labels(
                    label_frame,
                    labels_spec,
                    task=resolved_task,
                    validation_fraction=validation_fraction,
                    random_state=random_state,
                )
                if labels_spec.timestamp:
                    selected_search_cutoffs = _select_label_training_cutoffs(
                        train_labels,
                        timestamp=labels_spec.timestamp,
                        target=labels_spec.target,
                        task=resolved_task,
                        limit=search_training_frames,
                    )
                    train_labels = train_labels.loc[
                        train_labels[labels_spec.timestamp].isin(
                            selected_search_cutoffs
                        )
                    ]
                while split_marker in label_frame.columns:
                    split_marker += "_"
                train_labels = train_labels.copy()
                validation_labels = validation_labels.copy()
                train_labels[split_marker] = "train"
                validation_labels[split_marker] = "validation"
                search_labels = pd.concat(
                    [train_labels, validation_labels], ignore_index=True
                )
                target_column = labels_spec.target

            if preselected_config is None:
                logger.info(
                    "search_started",
                    candidates=len(configs),
                    task=resolved_task,
                    target=target_column,
                    sample_rows=sample_rows,
                    search_full_data=search_full_data,
                    training_frames=(
                        len(search_graph_cutoffs)
                        if graph_labels is not None
                        else (
                            search_labels.loc[
                                search_labels[split_marker] == "train",
                                labels_spec.timestamp,
                            ].nunique()
                            if labels_spec.timestamp
                            else 1
                        )
                    ),
                )
            else:
                logger.info(
                    "search_skipped",
                    reason="preselected_configuration",
                    feature_families=preselected_config.feature_families,
                    depth=preselected_config.depth,
                    auto_annotate_features=(preselected_config.auto_annotate_features),
                    frozen_plan=preselected_execution_plan is not None,
                )
            trials: list[Trial] = []
            if preselected_config is not None:
                trials.append(
                    Trial(
                        config=preselected_config,
                        metric=_metric_for_task(resolved_task),
                        validation_score=float("nan"),
                        objective_score=0.0,
                        feature_count=0,
                        train_rows=0,
                        validation_rows=0,
                        feature_seconds=0.0,
                        model_seconds=0.0,
                        execution_plan=(
                            copy.deepcopy(dict(preselected_execution_plan))
                            if preselected_execution_plan is not None
                            else None
                        ),
                    )
                )
            resource_blocks: list[tuple[tuple[str, ...], bool, int, str]] = []
            search_configs = () if preselected_config is not None else configs
            for trial_number, config in enumerate(search_configs, start=1):
                resource_block = _resource_block_for(config, resource_blocks)
                if resource_block is not None:
                    families, _annotations, minimum_depth, reason = resource_block
                    trials.append(
                        Trial(
                            config=config,
                            metric=_metric_for_task(resolved_task),
                            validation_score=float("nan"),
                            objective_score=float("-inf"),
                            feature_count=0,
                            train_rows=0,
                            validation_rows=0,
                            feature_seconds=0.0,
                            model_seconds=0.0,
                            status="skipped",
                            error=(
                                "Resource superset of failed configuration "
                                f"families={families}, depth={minimum_depth}: {reason}"
                            ),
                        )
                    )
                    logger.warning(
                        "trial_skipped",
                        trial=f"{trial_number}/{len(search_configs)}",
                        feature_families=config.feature_families,
                        depth=config.depth,
                        auto_annotate_features=config.auto_annotate_features,
                        reason="resource_superset",
                        blocked_by_families=families,
                        blocked_by_depth=minimum_depth,
                    )
                    continue
                logger.info(
                    "trial_started",
                    trial=f"{trial_number}/{len(search_configs)}",
                    feature_families=config.feature_families,
                    depth=config.depth,
                    auto_annotate_features=config.auto_annotate_features,
                    include_target_history=config.include_target_history,
                    feature_family_max_columns=config.feature_family_max_columns,
                )
                feature_started = perf_counter()
                try:
                    if graph_labels is not None:
                        materialized, execution_plan = _materialize_graph_label_splits(
                            workspace,
                            normalized_tables,
                            normalized_relationships,
                            root_name,
                            config,
                            graph_labels,
                            split_marker=split_marker,
                            cutoff_marker=cutoff_marker,
                            compute_period_days=compute_period_days,
                            verbose=verbose,
                            train_cutoffs=search_graph_cutoffs,
                        )
                    else:
                        materialized, execution_plan = _materialize_split(
                            workspace,
                            normalized_tables,
                            normalized_relationships,
                            root_name,
                            config,
                            search_labels,
                            labels_spec,
                            compute_period_days=compute_period_days,
                            verbose=verbose,
                            split_marker=split_marker,
                        )
                    train = materialized.loc[
                        materialized[split_marker] == "train"
                    ].drop(columns=split_marker)
                    validation = materialized.loc[
                        materialized[split_marker] == "validation"
                    ].drop(columns=split_marker)
                    feature_seconds = perf_counter() - feature_started
                    if train.empty or validation.empty:
                        raise ValueError(
                            "No labels matched the sampled parent rows; increase "
                            "sample_rows or check parent_key/label_key"
                        )
                    excluded = _model_exclusions(
                        normalized_tables,
                        root_name,
                        labels_spec,
                        graph_labels,
                        cutoff_marker,
                    )
                    train_x, train_y, validation_x, validation_y, categorical = (
                        prepare_features(
                            train,
                            validation,
                            target=target_column,
                            excluded=excluded,
                        )
                    )
                    datetime_columns = tuple(
                        column
                        for column in train_x.columns
                        if pd.api.types.is_datetime64_any_dtype(train[column].dtype)
                    )
                    estimator_train_x, estimator_train_y = sample_training_rows(
                        train_x,
                        train_y,
                        limit=estimator_train_rows,
                        task=resolved_task,
                        random_state=random_state,
                    )
                    model, metric, score, model_seconds = _fit_validation_estimator(
                        estimator_train_x,
                        estimator_train_y,
                        validation_x,
                        validation_y,
                        model_backend=model_backend,
                        task=resolved_task,
                        categorical=categorical,
                        random_state=random_state,
                        model_params=model_params,
                    )
                    completed_trial = Trial(
                        config=config,
                        metric=metric,
                        validation_score=score,
                        objective_score=_objective_score(metric, score),
                        feature_count=train_x.shape[1],
                        train_rows=len(estimator_train_x),
                        validation_rows=len(validation_x),
                        feature_seconds=feature_seconds,
                        model_seconds=model_seconds,
                        model=model,
                        feature_columns=tuple(train_x.columns),
                        categorical_columns=tuple(
                            train_x.columns[index] for index in categorical
                        ),
                        datetime_columns=datetime_columns,
                        execution_plan=execution_plan,
                    )
                    trials.append(completed_trial)
                    logger.info(
                        "trial_completed",
                        trial=f"{trial_number}/{len(search_configs)}",
                        metric=metric,
                        score=round(score, 6),
                        features=completed_trial.feature_count,
                        train_rows=completed_trial.train_rows,
                        validation_rows=completed_trial.validation_rows,
                        feature_seconds=round(completed_trial.feature_seconds, 3),
                        model_seconds=round(completed_trial.model_seconds, 3),
                    )
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    error_text = f"{type(exc).__name__}: {exc}"
                    if (
                        type(exc).__name__ == "OutOfMemoryException"
                        or "max_temp_directory_size" in str(exc)
                    ):
                        resource_blocks.append(
                            (
                                config.feature_families,
                                config.auto_annotate_features,
                                config.depth,
                                error_text,
                            )
                        )
                    trials.append(
                        Trial(
                            config=config,
                            metric=_metric_for_task(resolved_task),
                            validation_score=float("nan"),
                            objective_score=float("-inf"),
                            feature_count=0,
                            train_rows=0,
                            validation_rows=0,
                            feature_seconds=perf_counter() - feature_started,
                            model_seconds=0.0,
                            status="failed",
                            error=error_text,
                        )
                    )
                    logger.warning(
                        "trial_failed",
                        trial=f"{trial_number}/{len(search_configs)}",
                        feature_families=config.feature_families,
                        depth=config.depth,
                        auto_annotate_features=config.auto_annotate_features,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
            annotate_complexity(
                trials,
                classification_gain=classification_negligible_gain,
                regression_relative_gain=regression_negligible_relative_gain,
                feature_growth=drastic_feature_growth,
            )
            successful = [trial for trial in trials if trial.status == "completed"]
            if not successful:
                errors = "; ".join(
                    f"{trial.config}: {trial.error}" for trial in trials[:3]
                )
                raise RuntimeError(f"Every KurveRSC trial failed. {errors}")
            if preselected_config is not None:
                best = recommended = successful[0]
            else:
                best, recommended = _select_trials(
                    successful,
                    classification_negligible_gain=classification_negligible_gain,
                    regression_negligible_relative_gain=(
                        regression_negligible_relative_gain
                    ),
                )
            for trial in successful:
                if trial.note:
                    logger.info(
                        "trial_complexity",
                        feature_families=trial.config.feature_families,
                        depth=trial.config.depth,
                        auto_annotate_features=(trial.config.auto_annotate_features),
                        note=trial.note,
                    )
            if preselected_config is None:
                logger.info(
                    "search_selected",
                    metric=best.metric,
                    score=round(best.validation_score, 6),
                    feature_families=best.config.feature_families,
                    depth=best.config.depth,
                    auto_annotate_features=best.config.auto_annotate_features,
                    features=best.feature_count,
                )
                if recommended is not best:
                    logger.info(
                        "search_recommended",
                        metric=recommended.metric,
                        score=round(recommended.validation_score, 6),
                        feature_families=recommended.config.feature_families,
                        depth=recommended.config.depth,
                        auto_annotate_features=(
                            recommended.config.auto_annotate_features
                        ),
                        features=recommended.feature_count,
                    )

            screening_recommended = recommended
            workspace.close()
            workspace = _Workspace(
                con, sample_rows=sample_rows, random_state=random_state
            )
            for name, table in normalized_tables.items():
                workspace.add(name, table.source, sample=False)

            external_test_labels = pd.DataFrame()
            if graph_labels is not None:
                full_graph_cutoffs = _select_training_cutoffs(
                    graph_labels.train_cutoffs, full_training_frames
                )
                production_training_frames = len(full_graph_cutoffs)
            else:
                workspace.add(label_name, labels_spec.source, sample=False)
                full_label_frame = workspace.frame(label_name)
                required = {
                    *_key_parts(labels_spec.key),
                    labels_spec.target,
                    *([labels_spec.timestamp] if labels_spec.timestamp else []),
                    *([labels_spec.split] if labels_spec.split else []),
                }
                missing = required - set(full_label_frame.columns)
                if missing:
                    raise ValueError(
                        f"Label source is missing columns: {sorted(missing)}"
                    )
                if labels_spec.split:
                    external_test_labels = full_label_frame.loc[
                        full_label_frame[labels_spec.split] == labels_spec.test_value
                    ].copy()
                labeled = full_label_frame.dropna(subset=[labels_spec.target])
                full_train_labels, full_validation_labels = _split_labels(
                    labeled,
                    labels_spec,
                    task=resolved_task,
                    validation_fraction=validation_fraction,
                    random_state=random_state,
                )
                if labels_spec.timestamp:
                    full_cutoffs = _select_training_cutoffs(
                        full_train_labels[labels_spec.timestamp].dropna().tolist(),
                        full_training_frames,
                    )
                    full_train_labels = full_train_labels.loc[
                        full_train_labels[labels_spec.timestamp].isin(full_cutoffs)
                    ]
                    production_training_frames = len(full_cutoffs)
                else:
                    production_training_frames = 1
                full_train_labels = full_train_labels.copy()
                full_validation_labels = full_validation_labels.copy()
                full_train_labels[split_marker] = "train"
                full_validation_labels[split_marker] = "validation"
                full_labels = pd.concat(
                    [full_train_labels, full_validation_labels], ignore_index=True
                )

            def materialize_full_config(
                config: GraphConfig,
                *,
                execution_plan: Mapping[str, Any] | None = None,
            ) -> tuple[pd.DataFrame, dict[str, Any]]:
                if graph_labels is not None:
                    return _materialize_graph_label_splits(
                        workspace,
                        normalized_tables,
                        normalized_relationships,
                        root_name,
                        config,
                        graph_labels,
                        split_marker=split_marker,
                        cutoff_marker=cutoff_marker,
                        compute_period_days=compute_period_days,
                        verbose=verbose,
                        train_cutoffs=full_graph_cutoffs,
                        execution_plan=execution_plan,
                    )
                return _materialize_split(
                    workspace,
                    normalized_tables,
                    normalized_relationships,
                    root_name,
                    config,
                    full_labels,
                    labels_spec,
                    compute_period_days=compute_period_days,
                    verbose=verbose,
                    split_marker=split_marker,
                    execution_plan=execution_plan,
                )

            excluded = _model_exclusions(
                normalized_tables,
                root_name,
                labels_spec,
                graph_labels,
                cutoff_marker,
            )
            ranked_screening = sorted(
                successful,
                key=lambda item: (
                    -item.objective_score,
                    item.feature_count,
                    item.config.complexity,
                    graph_config_sort_key(item.config),
                ),
            )
            finalist_configs = [
                trial.config for trial in ranked_screening[:rerank_top_k]
            ]
            if screening_recommended.config not in finalist_configs:
                finalist_configs.append(screening_recommended.config)

            rerank_trials: list[Trial] = []
            rerank_materializations: dict[GraphConfig, str] = {}
            if production_training_frames > 1 and len(finalist_configs) > 1:
                logger.info(
                    "rerank_started",
                    candidates=len(finalist_configs),
                    training_frames=production_training_frames,
                )
                for finalist_number, config in enumerate(finalist_configs, start=1):
                    logger.info(
                        "rerank_trial_started",
                        trial=f"{finalist_number}/{len(finalist_configs)}",
                        feature_families=config.feature_families,
                        depth=config.depth,
                        auto_annotate_features=config.auto_annotate_features,
                    )
                    feature_started = perf_counter()
                    try:
                        materialized, execution_plan = materialize_full_config(config)
                        rerank_train = materialized.loc[
                            materialized[split_marker] == "train"
                        ].drop(columns=split_marker)
                        rerank_validation = materialized.loc[
                            materialized[split_marker] == "validation"
                        ].drop(columns=split_marker)
                        feature_seconds = perf_counter() - feature_started
                        if rerank_train.empty or rerank_validation.empty:
                            raise ValueError(
                                "Full-history rerank produced an empty model split"
                            )
                        (
                            rerank_train_x,
                            rerank_train_y,
                            rerank_validation_x,
                            rerank_validation_y,
                            rerank_categorical,
                        ) = prepare_features(
                            rerank_train,
                            rerank_validation,
                            target=target_column,
                            excluded=excluded,
                        )
                        rerank_datetime_columns = tuple(
                            column
                            for column in rerank_train_x.columns
                            if pd.api.types.is_datetime64_any_dtype(
                                rerank_train[column].dtype
                            )
                        )
                        rerank_estimator_x, rerank_estimator_y = sample_training_rows(
                            rerank_train_x,
                            rerank_train_y,
                            limit=estimator_train_rows,
                            task=resolved_task,
                            random_state=random_state,
                        )
                        rerank_model, metric, score, model_seconds = (
                            _fit_validation_estimator(
                                rerank_estimator_x,
                                rerank_estimator_y,
                                rerank_validation_x,
                                rerank_validation_y,
                                model_backend=model_backend,
                                task=resolved_task,
                                categorical=rerank_categorical,
                                random_state=random_state,
                                model_params=model_params,
                            )
                        )
                        reranked = Trial(
                            config=config,
                            metric=metric,
                            validation_score=score,
                            objective_score=_objective_score(metric, score),
                            feature_count=rerank_train_x.shape[1],
                            train_rows=len(rerank_estimator_x),
                            validation_rows=len(rerank_validation_x),
                            feature_seconds=feature_seconds,
                            model_seconds=model_seconds,
                            model=rerank_model,
                            feature_columns=tuple(rerank_train_x.columns),
                            categorical_columns=tuple(
                                rerank_train_x.columns[index]
                                for index in rerank_categorical
                            ),
                            datetime_columns=rerank_datetime_columns,
                            execution_plan=execution_plan,
                        )
                        cache_table = (
                            f"__kurversc_rerank_materialized_{finalist_number}"
                        )
                        cache_source = f"{cache_table}_source"
                        con.register(
                            cache_source, _duckdb_compatible_frame(materialized)
                        )
                        try:
                            con.sql(
                                f"CREATE TEMP TABLE {_quote_identifier(cache_table)} "
                                f"AS SELECT * FROM {_quote_identifier(cache_source)}"
                            )
                        finally:
                            con.unregister(cache_source)
                        rerank_materializations[config] = cache_table
                        rerank_trials.append(reranked)
                        logger.info(
                            "rerank_trial_completed",
                            trial=f"{finalist_number}/{len(finalist_configs)}",
                            metric=metric,
                            score=round(score, 6),
                            features=reranked.feature_count,
                            train_rows=reranked.train_rows,
                            validation_rows=reranked.validation_rows,
                            feature_seconds=round(feature_seconds, 3),
                            model_seconds=round(model_seconds, 3),
                        )
                    except Exception as exc:
                        if not continue_on_error:
                            raise
                        error_text = f"{type(exc).__name__}: {exc}"
                        rerank_trials.append(
                            Trial(
                                config=config,
                                metric=_metric_for_task(resolved_task),
                                validation_score=float("nan"),
                                objective_score=float("-inf"),
                                feature_count=0,
                                train_rows=0,
                                validation_rows=0,
                                feature_seconds=perf_counter() - feature_started,
                                model_seconds=0.0,
                                status="failed",
                                error=error_text,
                            )
                        )
                        logger.warning(
                            "rerank_trial_failed",
                            trial=f"{finalist_number}/{len(finalist_configs)}",
                            feature_families=config.feature_families,
                            depth=config.depth,
                            auto_annotate_features=config.auto_annotate_features,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )

                successful_reranks = [
                    trial for trial in rerank_trials if trial.status == "completed"
                ]
                if successful_reranks:
                    best, recommended = _select_trials(
                        successful_reranks,
                        classification_negligible_gain=(classification_negligible_gain),
                        regression_negligible_relative_gain=(
                            regression_negligible_relative_gain
                        ),
                    )
                    logger.info(
                        "rerank_selected",
                        metric=best.metric,
                        score=round(best.validation_score, 6),
                        feature_families=best.config.feature_families,
                        depth=best.config.depth,
                        auto_annotate_features=(best.config.auto_annotate_features),
                        features=best.feature_count,
                    )
                    if recommended is not best:
                        logger.info(
                            "rerank_recommended",
                            metric=recommended.metric,
                            score=round(recommended.validation_score, 6),
                            feature_families=recommended.config.feature_families,
                            depth=recommended.config.depth,
                            auto_annotate_features=(
                                recommended.config.auto_annotate_features
                            ),
                            features=recommended.feature_count,
                        )

            # Production refitting uses the full-history recommendation when
            # reranking succeeded, otherwise the one-frame recommendation.
            selected = recommended
            logger.info(
                "full_refit_started",
                feature_families=selected.config.feature_families,
                depth=selected.config.depth,
                auto_annotate_features=selected.config.auto_annotate_features,
                requested_training_frames=full_training_frames,
                reused_rerank_artifact=(selected.config in rerank_materializations),
                frozen_plan=preselected_execution_plan is not None,
            )
            selected_cache = rerank_materializations.get(selected.config)
            if selected_cache is not None:
                full_materialized = con.sql(
                    f"SELECT * FROM {_quote_identifier(selected_cache)}"
                ).to_df()
                if selected.execution_plan is None:
                    raise RuntimeError(
                        "Selected rerank trial did not retain its execution plan"
                    )
                production_plan = copy.deepcopy(selected.execution_plan)
            else:
                full_materialized, production_plan = materialize_full_config(
                    selected.config,
                    execution_plan=preselected_execution_plan,
                )
            for cache_table in rerank_materializations.values():
                con.sql(f"DROP TABLE IF EXISTS {_quote_identifier(cache_table)}")

            full_train = full_materialized.loc[
                full_materialized[split_marker] == "train"
            ].drop(columns=split_marker)
            full_validation = full_materialized.loc[
                full_materialized[split_marker] == "validation"
            ].drop(columns=split_marker)
            if full_train.empty or full_validation.empty:
                raise ValueError("Full-data refit produced an empty model split")
            plan_fingerprint = _execution_plan_fingerprint(production_plan)
            logger.info(
                "production_plan_frozen",
                training_frames=production_training_frames,
                operations=len(production_plan.get("records", [])),
                fingerprint=plan_fingerprint,
            )
            (
                full_train_x,
                full_train_y,
                full_validation_x,
                full_validation_y,
                full_categorical,
            ) = prepare_features(
                full_train,
                full_validation,
                target=target_column,
                excluded=excluded,
            )
            full_datetime_columns = tuple(
                column
                for column in full_train_x.columns
                if pd.api.types.is_datetime64_any_dtype(full_train[column].dtype)
            )
            if selected_cache is not None:
                if tuple(full_train_x.columns) != selected.feature_columns:
                    raise RuntimeError(
                        "Cached rerank feature schema changed before production fit"
                    )
                validation_model = selected.model
                full_metric = selected.metric
                full_validation_score = selected.validation_score
                validation_estimator_rows = selected.train_rows
            else:
                full_estimator_x, full_estimator_y = sample_training_rows(
                    full_train_x,
                    full_train_y,
                    limit=estimator_train_rows,
                    task=resolved_task,
                    random_state=random_state,
                )
                (
                    validation_model,
                    full_metric,
                    full_validation_score,
                    _,
                ) = _fit_validation_estimator(
                    full_estimator_x,
                    full_estimator_y,
                    full_validation_x,
                    full_validation_y,
                    model_backend=model_backend,
                    task=resolved_task,
                    categorical=full_categorical,
                    random_state=random_state,
                    model_params=model_params,
                )
                validation_estimator_rows = len(full_estimator_x)
            combined_x = pd.concat([full_train_x, full_validation_x], ignore_index=True)
            combined_y = pd.concat([full_train_y, full_validation_y], ignore_index=True)
            final_estimator_x, final_estimator_y = sample_training_rows(
                combined_x,
                combined_y,
                limit=estimator_train_rows,
                task=resolved_task,
                random_state=random_state,
            )
            final_model, final_model_seconds, target_classes = _fit_final_estimator(
                final_estimator_x,
                final_estimator_y,
                model_backend=model_backend,
                task=resolved_task,
                categorical=full_categorical,
                random_state=random_state,
                model_params=model_params,
            )

            test_frame = (
                _materialize_graph_test(
                    workspace,
                    normalized_tables,
                    normalized_relationships,
                    root_name,
                    selected.config,
                    graph_labels,
                    execution_plan=production_plan,
                    split_marker=split_marker,
                    cutoff_marker=cutoff_marker,
                    compute_period_days=compute_period_days,
                    verbose=verbose,
                )
                if graph_labels is not None
                else _materialize_external_test(
                    workspace,
                    normalized_tables,
                    normalized_relationships,
                    root_name,
                    selected.config,
                    external_test_labels,
                    labels_spec,
                    execution_plan=production_plan,
                    compute_period_days=compute_period_days,
                    verbose=verbose,
                )
            )
            test_predictions = None
            test_score = None
            if not test_frame.empty:
                test_x = prepare_prediction_features(
                    test_frame,
                    columns=list(full_train_x.columns),
                    categorical_columns={
                        full_train_x.columns[index] for index in full_categorical
                    },
                    datetime_columns=set(full_datetime_columns),
                )
                predictions = (
                    final_model.predict_proba(test_x)[:, 1]
                    if resolved_task == "classification"
                    else final_model.predict(test_x)
                )
                root = normalized_tables[root_name]
                root_prefix = root.prefix or f"{_slug(root_name, 'n0')[:10]}0"
                test_predictions = pd.DataFrame(index=test_frame.index)
                for key in _key_parts(root.key):
                    source_key = key if key in test_frame else f"{root_prefix}_{key}"
                    if source_key in test_frame:
                        test_predictions[key] = test_frame[source_key]
                for column in (
                    cutoff_marker,
                    *(
                        (labels_spec.timestamp,)
                        if labels_spec and labels_spec.timestamp
                        else ()
                    ),
                ):
                    if column in test_frame:
                        test_predictions[column] = test_frame[column]
                test_predictions["prediction"] = predictions
                if (
                    target_column in test_frame
                    and test_frame[target_column].notna().all()
                ):
                    if resolved_task == "classification":
                        from sklearn.metrics import roc_auc_score

                        class_map = {
                            value: index for index, value in enumerate(target_classes)
                        }
                        encoded_test = test_frame[target_column].map(class_map)
                        if encoded_test.nunique() == 2:
                            test_score = float(roc_auc_score(encoded_test, predictions))
                    else:
                        from sklearn.metrics import mean_absolute_error

                        test_score = float(
                            mean_absolute_error(
                                pd.to_numeric(test_frame[target_column]), predictions
                            )
                        )

            fitted_model = FittedModel(
                config=selected.config,
                execution_plan=production_plan,
                plan_fingerprint=plan_fingerprint,
                estimator=final_model,
                validation_estimator=validation_model,
                feature_columns=tuple(full_train_x.columns),
                categorical_columns=tuple(
                    full_train_x.columns[index] for index in full_categorical
                ),
                datetime_columns=full_datetime_columns,
                target=target_column,
                task=resolved_task,
                metric=full_metric,
                validation_score=full_validation_score,
                train_rows=validation_estimator_rows,
                validation_rows=len(full_validation_x),
                training_frames=production_training_frames,
                model_backend=model_backend,
                target_classes=target_classes,
                test_predictions=test_predictions,
                test_score=test_score,
            )
            if preselected_config is not None:
                best = recommended = Trial(
                    config=selected.config,
                    metric=full_metric,
                    validation_score=full_validation_score,
                    objective_score=_objective_score(
                        full_metric, full_validation_score
                    ),
                    feature_count=full_train_x.shape[1],
                    train_rows=validation_estimator_rows,
                    validation_rows=len(full_validation_x),
                    feature_seconds=0.0,
                    model_seconds=0.0,
                    model=validation_model,
                    feature_columns=tuple(full_train_x.columns),
                    categorical_columns=tuple(
                        full_train_x.columns[index] for index in full_categorical
                    ),
                    datetime_columns=full_datetime_columns,
                    execution_plan=production_plan,
                )
                trials = [best]
            logger.info(
                "full_refit_completed",
                metric=full_metric,
                validation_score=round(full_validation_score, 6),
                model_backend=model_backend,
                training_frames=production_training_frames,
                train_rows=len(full_train_x),
                estimator_train_rows=validation_estimator_rows,
                validation_rows=len(full_validation_x),
                features=len(full_train_x.columns),
                plan_operations=len(production_plan.get("records", [])),
                plan_fingerprint=plan_fingerprint,
                final_model_seconds=round(final_model_seconds, 3),
                final_estimator_rows=len(final_estimator_x),
                reused_rerank_artifact=selected_cache is not None,
                reused_preselected_plan=preselected_execution_plan is not None,
                test_rows=0 if test_predictions is None else len(test_predictions),
                test_score=test_score,
            )
            return FitResult(
                task=resolved_task,
                metric=best.metric,
                best_trial=best,
                recommended_trial=recommended,
                trials=tuple(trials),
                rerank_trials=tuple(rerank_trials),
                fitted_model=fitted_model,
            )
        finally:
            workspace.close()


def estimate_features(
    parent_node: Table | Source,
    *,
    tables: Sequence[Table | Source] | Mapping[str, Table | Source] = (),
    relationships: Sequence[Relationship | Mapping[str, Any]] = (),
    parent_key: Key | Sequence[str] | None = None,
    parent_date: str | None = None,
    parent_timeless: bool = False,
    max_depth: int = 4,
    feature_family_stages: Sequence[Sequence[str]] = DEFAULT_FAMILY_STAGES,
    auto_annotate_options: Sequence[bool] = (True, False),
    feature_family_max_columns: int | None = 4,
    feature_family_max_column_options: Sequence[int | None] | None = None,
    target_history_options: Sequence[bool] | None = None,
    max_base_table_occurrence_options: Sequence[int] | None = None,
    limit_expensive_family_depth: bool = True,
    feature_family_max_features_per_column: int | None = 32,
    feature_propagation_max_functions_per_column: int | None = 1,
    feature_ranking_rows: int = 2_000,
    intelligent_column_selection: bool = True,
    capability_pruning: bool = True,
    graph_configs: Sequence[GraphConfig] | None = None,
    auto_text_features: bool = False,
    auto_annotate_max_text_columns: int | None = None,
    source_row_counts: Mapping[str, int] | None = None,
    connection: duckdb.DuckDBPyConnection | None = None,
    random_state: int = 42,
    duckdb_memory_limit: str | None = "128GB",
    duckdb_max_temp_directory_size: str | None = "128GB",
    base_distribution_features: bool | None = None,
) -> FeaturePreflight:
    """Estimate feature volume without executing relational feature SQL.

    The preflight reads at most ``feature_ranking_rows`` rows from each source,
    profiles its schema, and simulates feature propagation over the configured
    graph topology. It reports final root width, peak intermediate width, and
    total generated columns for every candidate configuration.
    """

    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if feature_ranking_rows < 1:
        raise ValueError("feature_ranking_rows must be positive")
    if (
        feature_family_max_features_per_column is not None
        and feature_family_max_features_per_column < 1
    ):
        raise ValueError(
            "feature_family_max_features_per_column must be positive or None"
        )
    if (
        feature_propagation_max_functions_per_column is not None
        and feature_propagation_max_functions_per_column < 1
    ):
        raise ValueError(
            "feature_propagation_max_functions_per_column must be positive or None"
        )
    if graph_configs is not None and len(graph_configs) == 0:
        raise ValueError("graph_configs must contain at least one configuration")
    if duckdb_memory_limit is not None and (
        not isinstance(duckdb_memory_limit, str) or not duckdb_memory_limit.strip()
    ):
        raise ValueError("duckdb_memory_limit must be a non-empty string or None")
    if duckdb_max_temp_directory_size is not None and (
        not isinstance(duckdb_max_temp_directory_size, str)
        or not duckdb_max_temp_directory_size.strip()
    ):
        raise ValueError(
            "duckdb_max_temp_directory_size must be a non-empty string or None"
        )

    parent = coerce_table(
        parent_node,
        key=parent_key,
        date=parent_date,
        timeless=parent_timeless,
    )
    root_name, normalized_tables = _normalize_tables(parent, tables)
    normalized_relationships = tuple(
        coerce_relationship(item) for item in relationships
    )
    if graph_configs is None:
        column_budgets = resolve_feature_family_column_budgets(
            feature_family_max_columns,
            feature_family_max_column_options,
        )
        configs = incremental_configs(
            max_depth=max_depth,
            feature_family_stages=feature_family_stages,
            auto_annotate_options=auto_annotate_options,
            feature_family_max_columns=feature_family_max_columns,
            feature_family_max_column_options=column_budgets,
            feature_family_max_features_per_column=(
                feature_family_max_features_per_column
            ),
            feature_propagation_max_functions_per_column=(
                feature_propagation_max_functions_per_column
            ),
            target_history_options=target_history_options,
            max_base_table_occurrence_options=(max_base_table_occurrence_options),
            limit_expensive_family_depth=limit_expensive_family_depth,
        )
    else:
        configs = tuple(graph_configs)
        column_budgets = tuple(
            dict.fromkeys(config.feature_family_max_columns for config in configs)
        )
    if base_distribution_features is not None:
        configs = tuple(
            replace(config, base_distribution_features=bool(base_distribution_features))
            for config in configs
        )
    configs = _with_graph_feature_options(
        configs,
        auto_text_features=auto_text_features,
        auto_annotate_max_text_columns=auto_annotate_max_text_columns,
    )

    normalized_tables, feature_audit = _rank_feature_funnel(
        normalized_tables,
        normalized_relationships,
        connection=connection,
        sample_rows=feature_ranking_rows,
        random_state=random_state,
        feature_family_max_columns=column_budgets[0],
        feature_family_max_column_options=column_budgets,
        feature_family_max_features_per_column=(feature_family_max_features_per_column),
        intelligent_column_selection=intelligent_column_selection,
        duckdb_memory_limit=duckdb_memory_limit,
        duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
    )
    if capability_pruning and graph_configs is None:
        supported: list[GraphConfig] = []
        for config in configs:
            config_tables, config_relationships = _topology_for_config(
                normalized_tables, normalized_relationships, config
            )
            available = available_feature_families(
                config_tables,
                config_relationships,
                feature_audit.loc[feature_audit["table"].isin(config_tables)],
                auto_annotate_features=config.auto_annotate_features,
                include_target_history=config.include_target_history,
            )
            if set(config.feature_families).issubset(available):
                supported.append(config)
        configs = tuple(supported)

    known_rows = _known_source_row_counts(
        normalized_tables,
        search=True,
        overrides=source_row_counts,
    )
    estimates_list: list[ConfigFeatureEstimate] = []
    for config in configs:
        config_tables, config_relationships = _topology_for_config(
            normalized_tables, normalized_relationships, config
        )
        estimates_list.append(
            estimate_config_features(
                config,
                feature_audit.loc[feature_audit["table"].isin(config_tables)],
                config_relationships,
                config_tables,
                root_name=root_name,
                source_rows={
                    name: rows
                    for name, rows in known_rows.items()
                    if name in config_tables
                },
            )
        )
    estimates = tuple(estimates_list)
    logger.info(
        "feature_preflight_completed",
        candidates=len(estimates),
        sample_rows=feature_ranking_rows,
        max_output_features=max(
            (estimate.estimated_output_features for estimate in estimates),
            default=0,
        ),
        max_peak_features=max(
            (estimate.estimated_peak_features for estimate in estimates),
            default=0,
        ),
        max_total_generated_features=max(
            (estimate.estimated_total_generated_features for estimate in estimates),
            default=0,
        ),
        max_peak_working_cells=max(
            (estimate.estimated_peak_working_cells or 0 for estimate in estimates),
            default=0,
        ),
    )
    return FeaturePreflight(estimates=estimates, feature_audit=feature_audit)


def fit(
    parent_node: Table | Source,
    label_node: Labels | GraphLabels | Source,
    *,
    tables: Sequence[Table | Source] | Mapping[str, Table | Source] = (),
    relationships: Sequence[Relationship | Mapping[str, Any]] = (),
    parent_key: Key | Sequence[str] | None = None,
    label_key: Key | Sequence[str] | None = None,
    target: str | None = None,
    parent_date: str | None = None,
    parent_timeless: bool = False,
    label_timestamp: str | None = None,
    split_column: str | None = None,
    task: str = "auto",
    class_labels: Sequence[Any] | None = None,
    max_depth: int = 4,
    feature_family_stages: Sequence[Sequence[str]] = DEFAULT_FAMILY_STAGES,
    auto_annotate_options: Sequence[bool] = (True, False),
    feature_family_max_columns: int | None = 4,
    feature_family_max_column_options: Sequence[int | None] | None = None,
    target_history_options: Sequence[bool] | None = None,
    max_base_table_occurrence_options: Sequence[int] | None = None,
    limit_expensive_family_depth: bool = True,
    feature_family_max_features_per_column: int | None = 32,
    feature_propagation_max_functions_per_column: int | None = 1,
    feature_ranking_rows: int = 2_000,
    intelligent_column_selection: bool = True,
    forward_search_beam_width: int = 2,
    sample_rows: int = 100_000,
    screening_rows: int | None = 10_000,
    screening_cost_ratio_threshold: float | None = None,
    confirmation_top_k: int = 8,
    fidelity_promotion_top_k: int = 0,
    fidelity_promotion_training_frames: int | None = None,
    categorical_refinement_top_k: int = 0,
    categorical_refinement_seeds: Sequence[int] = tuple(range(8)),
    stage_candidate_budgets: SearchStageBudgets
    | Mapping[str, int | None]
    | None = None,
    search_parallelism: int = 1,
    search_full_data: bool = False,
    search_training_frames: int | None = None,
    full_training_frames: int | None = 1,
    full_validation_frames: int | None = None,
    rerank_top_k: int = 3,
    rerank_cutoff_frames: int = 3,
    rerank_stability_penalty: float = 0.25,
    validation_fraction: float = 0.2,
    compute_period_days: int = 3650,
    random_state: int = 42,
    model_backend: str = "catboost",
    final_model_backend: str | None = None,
    estimator_train_rows: int | None = None,
    temporal_estimator_mode: str = "pooled",
    model_params: Mapping[str, Any] | None = None,
    connection: duckdb.DuckDBPyConnection | None = None,
    duckdb_memory_limit: str | None = "128GB",
    duckdb_max_temp_directory_size: str | None = "128GB",
    continue_on_error: bool = True,
    verbose: bool = False,
    classification_negligible_gain: float = 0.002,
    regression_negligible_relative_gain: float = 0.005,
    drastic_feature_growth: float = 2.0,
    complexity_uncertainty_multiplier: float = 1.0,
    rerank_complexity_classification_loss_cap: float = 0.01,
    rerank_complexity_regression_relative_loss_cap: float = 0.01,
    rerank_nested_depth_safety: bool = False,
    rerank_practical_recent_ties: bool = False,
    complexity_selector: bool = True,
    structural_promotion: bool = False,
    confidence_frontier_promotion: bool = False,
    shallow_sparse_promotion: int = 0,
    adaptive_depth_promotion: bool = True,
    capability_pruning: bool = True,
    staged_search_top_k: int = 0,
    search_max_features: int | None = 8_000,
    search_max_working_cells: int | None = None,
    graph_configs: Sequence[GraphConfig] | None = None,
    infer_ts_periods: bool = False,
    auto_text_features: bool = False,
    auto_annotate_max_text_columns: int | None = None,
    high_cardinality_policy: str = "skip",
    categorical_low_cardinality_policy: str = "top_k",
    text_categorical_fallback: bool = False,
    preselected_config: GraphConfig | None = None,
    preselected_execution_plan: Mapping[str, Any] | None = None,
    rerank_preselected_config: bool = False,
    planner_collection: bool = False,
    planner_collection_candidates: int | None = 24,
    planner_collection_training_frames: int = 3,
    planner_collection_column_budgets: Sequence[int | None] = (2, 4, 8),
    planner_collection_random_state: int | None = None,
    search_checkpoint_path: str | Path | None = None,
    search_checkpoint_key: str | None = None,
    search_only: bool = False,
    base_distribution_features: bool | None = None,
    root_reachable_materialization: bool = False,
) -> FitResult:
    """Select and fit a GraphReduce plan with one resident feature frame.

    Default search uses three fidelities: a 10K-row forward-beam screen, a
    ``sample_rows`` confirmation of the strongest candidates, and a full-data
    rerank of the top three.  When ``fidelity_promotion_top_k`` is positive, a
    structurally diverse subset is additionally re-evaluated using
    ``fidelity_promotion_training_frames`` historical cutoffs (all available
    cutoffs when it is ``None``) before full-data reranking. Each frame is
    released before the next one is constructed.
    ``screening_cost_ratio_threshold`` optionally pilots
    two representative configurations at both row caps and skips the cheaper
    screen when its measured cost saving is too small. ``structural_promotion``
    reserves base, history, topology, and depth controls before sparse family
    quotas fill promotion slots. ``confidence_frontier_promotion`` retains
    uncertainty-competitive configurations across fidelity stages until the
    explicit stage budget is reached. ``shallow_sparse_promotion`` optionally
    reserves distinct shallow single-family controls ahead of wider depth
    variants. ``rerank_nested_depth_safety`` optionally prevents an
    unjustified deeper finalist from displacing its much smaller, otherwise
    identical shallower control. Optional categorical
    refinement evaluates seeded, deterministic top-k encodings only for
    promoted finalists. ``stage_candidate_budgets`` places hard evaluation
    ceilings on every search stage independently of its promotion quota. The selected plan is then
    replayed over the requested production cutoffs. ``planner_collection=True`` instead evaluates a
    balanced, full-source candidate design over multiple training frames and
    returns its teacher outcomes before production refitting. When
    ``search_checkpoint_path`` is provided, every completed screening,
    confirmation, challenger, and rerank trial is committed atomically and an
    identical retry resumes those trials instead of materializing them again.
    """

    if sample_rows < 1:
        raise ValueError("sample_rows must be positive")
    if screening_rows is not None and screening_rows < 1:
        raise ValueError("screening_rows must be positive or None")
    if (
        screening_cost_ratio_threshold is not None
        and screening_cost_ratio_threshold < 1.0
    ):
        raise ValueError("screening_cost_ratio_threshold must be at least 1 or None")
    if confirmation_top_k < 0:
        raise ValueError("confirmation_top_k must be non-negative")
    if fidelity_promotion_top_k < 0:
        raise ValueError("fidelity_promotion_top_k must be non-negative")
    if categorical_refinement_top_k < 0:
        raise ValueError("categorical_refinement_top_k must be non-negative")
    if shallow_sparse_promotion < 0:
        raise ValueError("shallow_sparse_promotion must be non-negative")
    if shallow_sparse_promotion and not structural_promotion:
        raise ValueError("shallow_sparse_promotion requires structural_promotion")
    if rerank_nested_depth_safety and not complexity_selector:
        raise ValueError("rerank_nested_depth_safety requires complexity_selector")
    categorical_refinement_seeds = tuple(
        dict.fromkeys(int(seed) for seed in categorical_refinement_seeds)
    )
    if categorical_refinement_top_k and not categorical_refinement_seeds:
        raise ValueError(
            "categorical_refinement_seeds must not be empty when "
            "categorical_refinement_top_k is positive"
        )
    if stage_candidate_budgets is None:
        resolved_stage_budgets = SearchStageBudgets()
    elif isinstance(stage_candidate_budgets, SearchStageBudgets):
        resolved_stage_budgets = stage_candidate_budgets
    elif isinstance(stage_candidate_budgets, Mapping):
        try:
            resolved_stage_budgets = SearchStageBudgets(**dict(stage_candidate_budgets))
        except TypeError as exc:
            raise ValueError(f"Invalid stage_candidate_budgets: {exc}") from exc
    else:
        raise TypeError(
            "stage_candidate_budgets must be SearchStageBudgets, a mapping, or None"
        )
    if (
        fidelity_promotion_training_frames is not None
        and fidelity_promotion_training_frames < 1
    ):
        raise ValueError("fidelity_promotion_training_frames must be positive or None")
    if fidelity_promotion_training_frames is not None and not fidelity_promotion_top_k:
        raise ValueError(
            "fidelity_promotion_training_frames requires fidelity_promotion_top_k > 0"
        )
    if search_parallelism < 1:
        raise ValueError("search_parallelism must be positive")
    if search_parallelism > 1 and connection is not None:
        raise ValueError(
            "search_parallelism greater than one requires KurveRSC-owned "
            "DuckDB connections"
        )
    if feature_ranking_rows < 1:
        raise ValueError("feature_ranking_rows must be positive")
    if (
        feature_family_max_features_per_column is not None
        and feature_family_max_features_per_column < 1
    ):
        raise ValueError(
            "feature_family_max_features_per_column must be positive or None"
        )
    if (
        feature_propagation_max_functions_per_column is not None
        and feature_propagation_max_functions_per_column < 1
    ):
        raise ValueError(
            "feature_propagation_max_functions_per_column must be positive or None"
        )
    if forward_search_beam_width < 1:
        raise ValueError("forward_search_beam_width must be positive")
    if target_history_options is not None and not target_history_options:
        raise ValueError("target_history_options must not be empty")
    if max_base_table_occurrence_options is not None and (
        not max_base_table_occurrence_options
        or any(int(value) < 1 for value in max_base_table_occurrence_options)
    ):
        raise ValueError(
            "max_base_table_occurrence_options must contain positive integers"
        )
    if complexity_uncertainty_multiplier < 0:
        raise ValueError("complexity_uncertainty_multiplier must be non-negative")
    if rerank_complexity_classification_loss_cap < 0:
        raise ValueError(
            "rerank_complexity_classification_loss_cap must be non-negative"
        )
    if rerank_complexity_regression_relative_loss_cap < 0:
        raise ValueError(
            "rerank_complexity_regression_relative_loss_cap must be non-negative"
        )
    if staged_search_top_k < 0:
        raise ValueError("staged_search_top_k must be non-negative")
    if search_max_features is not None and search_max_features < 1:
        raise ValueError("search_max_features must be positive or None")
    if search_max_working_cells is not None and search_max_working_cells < 1:
        raise ValueError("search_max_working_cells must be positive or None")
    if high_cardinality_policy not in {"skip", "top_k"}:
        raise ValueError("high_cardinality_policy must be either 'skip' or 'top_k'")
    if categorical_low_cardinality_policy not in {"top_k", "all"}:
        raise ValueError(
            "categorical_low_cardinality_policy must be either 'top_k' or 'all'"
        )
    if search_training_frames is not None and search_training_frames < 1:
        raise ValueError("search_training_frames must be positive or None")
    if full_training_frames is not None and full_training_frames < 1:
        raise ValueError("full_training_frames must be positive or None")
    if full_validation_frames is not None and full_validation_frames < 1:
        raise ValueError("full_validation_frames must be positive or None")
    if rerank_top_k < 0:
        raise ValueError("rerank_top_k must be non-negative")
    if rerank_cutoff_frames < 1:
        raise ValueError("rerank_cutoff_frames must be positive")
    if rerank_stability_penalty < 0:
        raise ValueError("rerank_stability_penalty must be non-negative")
    if planner_collection_candidates is not None and planner_collection_candidates < 1:
        raise ValueError("planner_collection_candidates must be positive or None")
    if planner_collection_training_frames < 1:
        raise ValueError("planner_collection_training_frames must be positive")
    if not planner_collection_column_budgets:
        raise ValueError("planner_collection_column_budgets must not be empty")
    if search_checkpoint_path is not None and not str(search_checkpoint_path).strip():
        raise ValueError("search_checkpoint_path must be a non-empty path or None")
    if model_backend not in {
        "catboost",
        "tabpfn_v3",
        "tabpfn_v3_5",
        "tabicl",
        "tabfm",
    }:
        raise ValueError(
            "model_backend must be 'catboost', 'tabpfn_v3', 'tabpfn_v3_5', 'tabicl', or 'tabfm'"
        )
    # Candidate screening and walk-forward reranking intentionally retain
    # ``model_backend``. A different final backend lets an expensive frozen
    # in-context model consume only the selected GraphReduce representation.
    resolved_final_model_backend = final_model_backend or model_backend
    if resolved_final_model_backend not in {
        "catboost",
        "tabpfn_v3",
        "tabpfn_v3_5",
        "tabicl",
        "tabfm",
    }:
        raise ValueError(
            "final_model_backend must be 'catboost', 'tabpfn_v3', 'tabpfn_v3_5', 'tabicl', or 'tabfm'"
        )
    if estimator_train_rows is None and model_backend in {
        "tabpfn_v3",
        "tabpfn_v3_5",
        "tabicl",
        "tabfm",
    }:
        estimator_train_rows = 10_000
    if estimator_train_rows is not None and estimator_train_rows < 2:
        raise ValueError("estimator_train_rows must be at least 2 or None")
    if temporal_estimator_mode not in {"pooled", "frame_ensemble"}:
        raise ValueError("temporal_estimator_mode must be 'pooled' or 'frame_ensemble'")
    if duckdb_memory_limit is not None and (
        not isinstance(duckdb_memory_limit, str) or not duckdb_memory_limit.strip()
    ):
        raise ValueError("duckdb_memory_limit must be a non-empty string or None")
    if duckdb_max_temp_directory_size is not None and (
        not isinstance(duckdb_max_temp_directory_size, str)
        or not duckdb_max_temp_directory_size.strip()
    ):
        raise ValueError(
            "duckdb_max_temp_directory_size must be a non-empty string or None"
        )
    if preselected_config is not None and graph_configs is not None:
        raise ValueError("preselected_config and graph_configs are mutually exclusive")
    if preselected_execution_plan is not None and preselected_config is None:
        raise ValueError("preselected_execution_plan requires preselected_config")
    if rerank_preselected_config and preselected_execution_plan is None:
        raise ValueError(
            "rerank_preselected_config requires a frozen preselected_execution_plan"
        )
    if rerank_preselected_config and rerank_top_k < 1:
        raise ValueError("rerank_preselected_config requires rerank_top_k >= 1")
    if graph_configs is not None and len(graph_configs) == 0:
        raise ValueError("graph_configs must contain at least one configuration")
    if staged_search_top_k and (
        graph_configs is not None
        or preselected_config is not None
        or planner_collection
    ):
        raise ValueError(
            "staged_search_top_k cannot be combined with graph_configs, "
            "preselected_config, or planner_collection"
        )
    if planner_collection and preselected_config is not None:
        raise ValueError("planner_collection cannot use a preselected configuration")
    if search_training_frames is None:
        # Search and production optimize the same temporal estimator unless a
        # caller explicitly opts into a cheaper approximate search.
        search_training_frames = full_training_frames
    if planner_collection:
        # Teacher collection values coverage over production-search latency.
        # Every selected candidate is evaluated on uncapped sources and the
        # collection result returns before the production refit.
        search_full_data = True
        search_training_frames = planner_collection_training_frames
        confirmation_top_k = 0
        rerank_top_k = 0
        if graph_configs is None and feature_family_max_column_options is None:
            feature_family_max_column_options = tuple(planner_collection_column_budgets)

    parent = coerce_table(
        parent_node,
        key=parent_key,
        date=parent_date,
        timeless=parent_timeless,
    )
    graph_labels = label_node if isinstance(label_node, GraphLabels) else None
    labels_spec = (
        None
        if graph_labels is not None
        else coerce_labels(
            label_node,
            target=target,
            key=label_key,
            timestamp=label_timestamp,
            split=split_column,
        )
    )
    if parent.key is None or (labels_spec is not None and labels_spec.key is None):
        raise ValueError(
            "parent_key and label_key are required (directly or in Table/Labels)"
        )
    if labels_spec is not None and labels_spec.target is None:
        raise ValueError("target is required (directly or in Labels)")
    if (
        (graph_labels is not None or labels_spec.timestamp is not None)
        and parent.date is None
        and not parent.timeless
    ):
        raise ValueError(
            "parent_date is required when labels have timestamps unless the "
            "root entity is explicitly declared timeless=True; KurveRSC will "
            "not silently run a temporal task without a point-in-time root cutoff"
        )

    root_name, normalized_tables = _normalize_tables(parent, tables)
    normalized_relationships = tuple(
        coerce_relationship(item) for item in relationships
    )
    if graph_labels is not None and graph_labels.table not in normalized_tables:
        raise ValueError(
            f"GraphLabels table {graph_labels.table!r} must appear in tables"
        )
    (
        normalized_tables,
        normalized_relationships,
        prediction_target_history,
    ) = _prepare_target_history(
        normalized_tables,
        normalized_relationships,
        root_name=root_name,
        labels_spec=labels_spec,
        task=task,
        validation_fraction=validation_fraction,
        random_state=random_state,
        connection=connection,
        duckdb_memory_limit=duckdb_memory_limit,
        duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
    )
    confirmation_trials: list[Trial] = []
    if preselected_config is not None:
        column_budgets = (preselected_config.feature_family_max_columns,)
        audit_features_per_column = (
            preselected_config.feature_family_max_features_per_column
        )
    elif graph_configs is not None:
        column_budgets = tuple(
            dict.fromkeys(config.feature_family_max_columns for config in graph_configs)
        )
        audit_features_per_column = feature_family_max_features_per_column
    else:
        column_budgets = resolve_feature_family_column_budgets(
            feature_family_max_columns,
            feature_family_max_column_options,
        )
        audit_features_per_column = feature_family_max_features_per_column
    normalized_tables, feature_audit = _rank_feature_funnel(
        normalized_tables,
        normalized_relationships,
        connection=connection,
        sample_rows=min(sample_rows, feature_ranking_rows),
        random_state=random_state,
        feature_family_max_columns=column_budgets[0],
        feature_family_max_column_options=column_budgets,
        feature_family_max_features_per_column=audit_features_per_column,
        intelligent_column_selection=intelligent_column_selection,
        duckdb_memory_limit=duckdb_memory_limit,
        duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
    )
    logger.info(
        "feature_funnel_ranked",
        tables=len(normalized_tables),
        columns=len(feature_audit),
        eligible=int(feature_audit["eligible"].sum()),
        within_family_budget=int(feature_audit["within_family_budget"].sum()),
        within_expanded_family_budget=int(
            feature_audit["within_expanded_family_budget"].sum()
        ),
        source_column_budgets=column_budgets,
        max_features_per_source_column=audit_features_per_column,
        intelligent_column_selection=intelligent_column_selection,
        column_selection_strategy=(
            "metadata_utility"
            if intelligent_column_selection
            else "configured_source_order"
        ),
        ranking_rows=min(sample_rows, feature_ranking_rows),
    )
    configs = (
        (preselected_config,)
        if preselected_config is not None
        else tuple(graph_configs)
        if graph_configs is not None
        else published_compatible_configs(
            auto_annotate_options=auto_annotate_options,
            feature_family_max_columns=feature_family_max_columns,
            feature_family_max_column_options=column_budgets,
            feature_family_max_features_per_column=(
                feature_family_max_features_per_column
            ),
            feature_propagation_max_functions_per_column=(
                feature_propagation_max_functions_per_column
            ),
            target_history_options=target_history_options,
            max_base_table_occurrence_options=(max_base_table_occurrence_options),
            limit_expensive_family_depth=limit_expensive_family_depth,
            legacy_plan_compatibility=False,
        )
        if staged_search_top_k
        else incremental_configs(
            max_depth=max_depth,
            feature_family_stages=feature_family_stages,
            auto_annotate_options=auto_annotate_options,
            feature_family_max_columns=feature_family_max_columns,
            feature_family_max_column_options=column_budgets,
            feature_family_max_features_per_column=(
                feature_family_max_features_per_column
            ),
            feature_propagation_max_functions_per_column=(
                feature_propagation_max_functions_per_column
            ),
        )
    )
    if not configs:
        raise ValueError("graph_configs must contain at least one configuration")
    if base_distribution_features is not None:
        configs = tuple(
            replace(config, base_distribution_features=bool(base_distribution_features))
            for config in configs
        )
    configs = _with_graph_feature_options(
        configs,
        auto_text_features=auto_text_features,
        auto_annotate_max_text_columns=auto_annotate_max_text_columns,
        high_cardinality_policy=high_cardinality_policy,
        categorical_low_cardinality_policy=categorical_low_cardinality_policy,
        text_categorical_fallback=text_categorical_fallback,
    )
    if root_reachable_materialization:
        # Path-specific, non-context graphs only need rows reachable from the
        # roots being scored. Cross-root context features deliberately retain
        # their full source tables.
        configs = tuple(
            replace(config, root_reachable_materialization=True)
            if "context" not in config.feature_families
            else config
            for config in configs
        )

    # This block belongs to the public streaming fit implementation below the
    # feature-funnel audit. Explicit configurations remain exact caller intent.
    if capability_pruning and graph_configs is None and preselected_config is None:
        supported_configs: list[GraphConfig] = []
        unavailable_by_annotation: dict[bool, set[str]] = {}
        for config in configs:
            config_tables, config_relationships = _topology_for_config(
                normalized_tables, normalized_relationships, config
            )
            available = available_feature_families(
                config_tables,
                config_relationships,
                feature_audit.loc[feature_audit["table"].isin(config_tables)],
                auto_annotate_features=config.auto_annotate_features,
                include_target_history=config.include_target_history,
            )
            unavailable = set(config.feature_families) - set(available)
            unavailable_by_annotation.setdefault(
                config.auto_annotate_features, set()
            ).update(unavailable)
            if not unavailable:
                supported_configs.append(config)
        logger.info(
            "search_capabilities_resolved",
            candidates_before=len(configs),
            candidates_after=len(supported_configs),
            unavailable_with_annotations=sorted(
                unavailable_by_annotation.get(True, set())
            ),
            unavailable_without_annotations=sorted(
                unavailable_by_annotation.get(False, set())
            ),
        )
        configs = tuple(supported_configs)
    if resolved_stage_budgets.screening is not None:
        candidates_before_budget = len(configs)
        configs = diverse_collection_configs(
            configs,
            max_candidates=resolved_stage_budgets.screening,
            random_state=random_state,
        )
        logger.info(
            "stage_candidate_budget_applied",
            stage="screening",
            candidates_before=candidates_before_budget,
            candidates_after=len(configs),
            budget=resolved_stage_budgets.screening,
        )
    if planner_collection and graph_configs is None:
        configs = diverse_collection_configs(
            configs,
            max_candidates=planner_collection_candidates,
            random_state=(
                random_state
                if planner_collection_random_state is None
                else planner_collection_random_state
            ),
        )
    planning_source_rows = _known_source_row_counts(
        normalized_tables,
        search=True,
    )
    config_feature_estimates: dict[GraphConfig, ConfigFeatureEstimate] = {}
    for config in configs:
        config_tables, config_relationships = _topology_for_config(
            normalized_tables, normalized_relationships, config
        )
        config_feature_estimates[config] = estimate_config_features(
            config,
            feature_audit.loc[feature_audit["table"].isin(config_tables)],
            config_relationships,
            config_tables,
            root_name=root_name,
            source_rows={
                name: rows
                for name, rows in planning_source_rows.items()
                if name in config_tables
            },
        )
    logger.info(
        "feature_preflight_completed",
        candidates=len(config_feature_estimates),
        sample_rows=min(sample_rows, feature_ranking_rows),
        max_output_features=max(
            (
                estimate.estimated_output_features
                for estimate in config_feature_estimates.values()
            ),
            default=0,
        ),
        max_peak_features=max(
            (
                estimate.estimated_peak_features
                for estimate in config_feature_estimates.values()
            ),
            default=0,
        ),
        max_total_generated_features=max(
            (
                estimate.estimated_total_generated_features
                for estimate in config_feature_estimates.values()
            ),
            default=0,
        ),
        max_peak_working_cells=max(
            (
                estimate.estimated_peak_working_cells or 0
                for estimate in config_feature_estimates.values()
            ),
            default=0,
        ),
    )
    effective_screening_rows = min(
        sample_rows,
        sample_rows if screening_rows is None else screening_rows,
    )

    split_marker = "__kurversc_validation_split__"
    cutoff_marker = "__kurversc_cutoff__"
    target_column: str
    target_class_universe: tuple[Any, ...] = ()
    external_test_labels = pd.DataFrame()
    rerank_source_train_labels: pd.DataFrame | None = None

    if graph_labels is not None:
        resolved_task = (
            "classification"
            if task == "auto" and graph_labels.operation.lower() == "bool"
            else "regression"
            if task == "auto"
            else _infer_task(pd.Series(), task)
        )
        target_column = graph_labels.target
        search_graph_cutoffs = _select_training_cutoffs(
            graph_labels.train_cutoffs, search_training_frames
        )
        fidelity_graph_cutoffs = _select_training_cutoffs(
            graph_labels.train_cutoffs,
            fidelity_promotion_training_frames,
        )
        search_train_labels = search_validation_labels = None
        fidelity_train_labels = None
    else:
        search_label_source = (
            # ``search_source`` defines the adapter's task-connected label
            # cohort. Full-data search removes its row cap; it must not replace
            # that cohort with production/test label rows.
            labels_spec.search_source
            if labels_spec.search_source is not None
            else labels_spec.source
        )
        search_label_frame = _load_frame_source(
            search_label_source,
            connection=connection,
            sample_rows=sample_rows,
            random_state=random_state,
            sample=not search_full_data,
            name="search_labels",
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
        )
        required = {
            *_key_parts(labels_spec.key),
            labels_spec.target,
            *([labels_spec.timestamp] if labels_spec.timestamp else []),
            *([labels_spec.split] if labels_spec.split else []),
        }
        missing = required - set(search_label_frame.columns)
        if missing:
            raise ValueError(f"Label source is missing columns: {sorted(missing)}")
        search_label_frame = search_label_frame.dropna(subset=[labels_spec.target])
        resolved_task = _infer_task(search_label_frame[labels_spec.target], task)
        if class_labels is not None and resolved_task != "multiclass_classification":
            raise ValueError(
                "class_labels is only supported for multiclass classification"
            )
        if resolved_task == "multiclass_classification":
            target_class_universe = tuple(
                class_labels
                if class_labels is not None
                else _binary_classes(search_label_frame[labels_spec.target])
            )
            if len(target_class_universe) < 2:
                raise ValueError(
                    "multiclass classification requires at least two classes"
                )
            unseen = set(pd.unique(search_label_frame[labels_spec.target])) - set(
                target_class_universe
            )
            if unseen:
                raise ValueError("class_labels does not cover every observed target")
        search_train_label_pool, search_validation_labels = _split_labels(
            search_label_frame,
            labels_spec,
            task=resolved_task,
            validation_fraction=validation_fraction,
            random_state=random_state,
        )
        fidelity_label_source = (
            labels_spec.fidelity_source
            if labels_spec.fidelity_source is not None
            else search_label_source
        )
        if fidelity_label_source is search_label_source:
            fidelity_label_frame = search_label_frame
        else:
            fidelity_label_frame = _load_frame_source(
                fidelity_label_source,
                connection=connection,
                sample_rows=sample_rows,
                random_state=random_state,
                sample=not search_full_data,
                name="fidelity_labels",
                duckdb_memory_limit=duckdb_memory_limit,
                duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
            )
            fidelity_missing = required - set(fidelity_label_frame.columns)
            if fidelity_missing:
                raise ValueError(
                    "Fidelity label source is missing columns: "
                    f"{sorted(fidelity_missing)}"
                )
            fidelity_label_frame = fidelity_label_frame.dropna(
                subset=[labels_spec.target]
            )
        fidelity_train_label_pool, _fidelity_validation_labels = _split_labels(
            fidelity_label_frame,
            labels_spec,
            task=resolved_task,
            validation_fraction=validation_fraction,
            random_state=random_state,
        )
        if labels_spec.timestamp:
            selected = _select_label_training_cutoffs(
                search_train_label_pool,
                timestamp=labels_spec.timestamp,
                target=labels_spec.target,
                task=resolved_task,
                limit=search_training_frames,
            )
            search_train_labels = search_train_label_pool.loc[
                search_train_label_pool[labels_spec.timestamp].isin(selected)
            ]
            fidelity_selected = _select_label_training_cutoffs(
                fidelity_train_label_pool,
                timestamp=labels_spec.timestamp,
                target=labels_spec.target,
                task=resolved_task,
                limit=fidelity_promotion_training_frames,
            )
            fidelity_train_labels = fidelity_train_label_pool.loc[
                fidelity_train_label_pool[labels_spec.timestamp].isin(fidelity_selected)
            ]
        else:
            search_train_labels = search_train_label_pool
            fidelity_train_labels = fidelity_train_label_pool
        target_column = labels_spec.target
        del (
            search_label_frame,
            search_train_label_pool,
            fidelity_label_frame,
            fidelity_train_label_pool,
            _fidelity_validation_labels,
        )
        gc.collect()

    search_training_frame_count = (
        len(search_graph_cutoffs)
        if graph_labels is not None
        else sum(1 for _ in _training_groups(search_train_labels, labels_spec))
    )
    fidelity_training_frame_count = (
        len(fidelity_graph_cutoffs)
        if graph_labels is not None
        else sum(1 for _ in _training_groups(fidelity_train_labels, labels_spec))
    )

    search_checkpoint: SearchCheckpoint | None = None
    checkpoint_contract: dict[str, Any] | None = None
    if search_checkpoint_path is not None and preselected_config is None:

        def source_identity(source: Source | None) -> dict[str, Any] | None:
            if source is None:
                return None
            if isinstance(source, pd.DataFrame):
                return {
                    "kind": "dataframe",
                    "rows": len(source),
                    "columns": tuple(source.columns),
                    "dtypes": tuple(str(dtype) for dtype in source.dtypes),
                }
            return {"kind": "reference", "value": str(source)}

        table_contract = {
            name: {
                "source": source_identity(table.source),
                "search_source": source_identity(table.search_source),
                "fidelity_source": source_identity(table.fidelity_source),
                "key": table.key,
                "date": table.date,
                "timeless": table.timeless,
                "prefix": table.prefix,
                "columns": table.columns,
                "context_keys": table.context_keys,
                "sample_in_search": table.sample_in_search,
                "cutoff_boundary": table.cutoff_boundary,
                **({"available_at":table.available_at} if table.available_at else {}),
                **({"model_exclusions":table.model_exclusions} if table.model_exclusions else {}),
                "auto_base_predicate_max": table.auto_base_predicate_max,
                "base_predicate_windows": table.base_predicate_windows,
                "base_predicates": tuple(
                    repr(predicate) for predicate in (table.base_predicates or ())
                ),
                "path_base_tables": table.path_base_tables,
            }
            for name, table in normalized_tables.items()
        }
        if graph_labels is not None:
            label_contract: dict[str, Any] = {
                "kind": "graph",
                "table": graph_labels.table,
                "field": graph_labels.field,
                "operation": graph_labels.operation,
                "period_days": graph_labels.period_days,
                "train_cutoffs": tuple(str(value) for value in search_graph_cutoffs),
                "fidelity_train_cutoffs": tuple(
                    str(value) for value in fidelity_graph_cutoffs
                ),
                "validation_cutoffs": tuple(
                    str(value) for value in graph_labels.validation_cutoffs
                ),
                "target": graph_labels.target,
            }
        else:
            timestamp = labels_spec.timestamp
            label_contract = {
                "kind": "table",
                "source": source_identity(labels_spec.source),
                "search_source": source_identity(labels_spec.search_source),
                "fidelity_source": source_identity(labels_spec.fidelity_source),
                "target": labels_spec.target,
                "key": labels_spec.key,
                "timestamp": timestamp,
                **({"available_at":labels_spec.available_at} if labels_spec.available_at else {}),
                "split": labels_spec.split,
                "train_rows": len(search_train_labels),
                "fidelity_train_rows": len(fidelity_train_labels),
                "validation_rows": len(search_validation_labels),
                "train_cutoffs": (
                    tuple(
                        str(value)
                        for value in sorted(
                            pd.unique(search_train_labels[timestamp].dropna())
                        )
                    )
                    if timestamp
                    else ()
                ),
                "validation_cutoffs": (
                    tuple(
                        str(value)
                        for value in sorted(
                            pd.unique(search_validation_labels[timestamp].dropna())
                        )
                    )
                    if timestamp
                    else ()
                ),
                "fidelity_train_cutoffs": (
                    tuple(
                        str(value)
                        for value in sorted(
                            pd.unique(fidelity_train_labels[timestamp].dropna())
                        )
                    )
                    if timestamp
                    else ()
                ),
                "peer_context": (
                    labels_spec.peer_context.contract()
                    if labels_spec.peer_context is not None
                    else None
                ),
            }
        feature_audit_fingerprint = hashlib.sha256(
            feature_audit.to_json(
                orient="split",
                date_format="iso",
                default_handler=str,
            ).encode("utf-8")
        ).hexdigest()
        # Rerank selection-only flags are intentionally absent: within a
        # fixed source revision they do not change trial materialization or
        # scores, so alternative selectors can reuse completed trials.
        checkpoint_contract = {
            "checkpoint_key": search_checkpoint_key,
            "root_name": root_name,
            "task": resolved_task,
            "target": target_column,
            "tables": table_contract,
            "relationships": tuple(
                asdict(relationship) for relationship in normalized_relationships
            ),
            "labels": label_contract,
            "feature_audit_fingerprint": feature_audit_fingerprint,
            "sample_rows": sample_rows,
            "screening_rows": effective_screening_rows,
            "screening_cost_ratio_threshold": screening_cost_ratio_threshold,
            "structural_promotion": structural_promotion,
            "shallow_sparse_promotion": shallow_sparse_promotion,
            "search_full_data": search_full_data,
            "search_training_frames": search_training_frames,
            "fidelity_promotion_training_frames": (fidelity_promotion_training_frames),
            "categorical_refinement_top_k": categorical_refinement_top_k,
            "categorical_refinement_seeds": categorical_refinement_seeds,
            "stage_candidate_budgets": asdict(resolved_stage_budgets),
            "full_training_frames": full_training_frames,
            "full_validation_frames": full_validation_frames,
            "rerank_cutoff_frames": rerank_cutoff_frames,
            "rerank_stability_penalty": rerank_stability_penalty,
            "rerank_complexity_classification_loss_cap": (
                rerank_complexity_classification_loss_cap
            ),
            "rerank_complexity_regression_relative_loss_cap": (
                rerank_complexity_regression_relative_loss_cap
            ),
            "rerank_practical_recent_ties": rerank_practical_recent_ties,
            "estimator_train_rows": estimator_train_rows,
            "model_backend": model_backend,
            "temporal_estimator_mode": temporal_estimator_mode,
            "model_params": dict(model_params or {}),
            "random_state": random_state,
            "compute_period_days": compute_period_days,
            "infer_ts_periods": infer_ts_periods,
            "intelligent_column_selection": intelligent_column_selection,
            "confirmation_top_k": confirmation_top_k,
            "fidelity_promotion_top_k": fidelity_promotion_top_k,
            "staged_search_top_k": staged_search_top_k,
            "rerank_top_k": rerank_top_k,
        }
        if screening_cost_ratio_threshold is None:
            try:
                search_checkpoint = SearchCheckpoint(
                    search_checkpoint_path,
                    contract=checkpoint_contract,
                )
                logger.info(
                    "search_checkpoint_ready",
                    path=str(search_checkpoint.run_directory),
                    fingerprint=search_checkpoint.contract_fingerprint,
                )
            except Exception as exc:
                logger.warning(
                    "search_checkpoint_unavailable",
                    path=str(search_checkpoint_path),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

    def load_checkpointed_trial(
        config: GraphConfig,
        *,
        stage: str,
        sample_rows: int | None,
        trial_label: str,
    ) -> Trial | None:
        if search_checkpoint is None:
            return None
        try:
            trial = search_checkpoint.load(
                config,
                stage=stage,
                sample_rows=sample_rows,
            )
        except Exception as exc:
            logger.warning(
                "search_checkpoint_read_failed",
                stage=stage,
                trial=trial_label,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return None
        if trial is not None:
            logger.info(
                f"{'trial' if stage == 'screening' else f'{stage}_trial'}_resumed",
                trial=trial_label,
                metric=trial.metric,
                score=round(trial.validation_score, 6),
                features=trial.feature_count,
                feature_families=config.feature_families,
                depth=config.depth,
                auto_annotate_features=config.auto_annotate_features,
                include_target_history=config.include_target_history,
                sample_rows=sample_rows,
            )
        return trial

    def save_checkpointed_trial(trial: Trial, *, trial_label: str) -> None:
        if search_checkpoint is None or trial.status != "completed":
            return
        try:
            path = search_checkpoint.save(trial)
            logger.info(
                "search_checkpoint_saved",
                stage=trial.stage,
                trial=trial_label,
                path=None if path is None else str(path),
            )
        except Exception as exc:
            logger.warning(
                "search_checkpoint_write_failed",
                stage=trial.stage,
                trial=trial_label,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    excluded = _model_exclusions(
        normalized_tables,
        root_name,
        labels_spec,
        graph_labels,
        cutoff_marker,
    )

    trials: list[Trial] = []
    resource_blocks: list[tuple[tuple[str, ...], bool, int, str]] = []
    if preselected_config is None:
        logger.info(
            "search_started",
            candidates=len(configs),
            task=resolved_task,
            target=target_column,
            screening_rows=(None if search_full_data else effective_screening_rows),
            confirmation_rows=(
                None
                if search_full_data or effective_screening_rows >= sample_rows
                else sample_rows
            ),
            confirmation_candidates=confirmation_top_k,
            categorical_refinement_parents=categorical_refinement_top_k,
            categorical_refinement_seeds=categorical_refinement_seeds,
            stage_candidate_budgets=asdict(resolved_stage_budgets),
            search_parallelism=search_parallelism,
            staged_search_top_k=staged_search_top_k,
            search_full_data=search_full_data,
            planner_collection=planner_collection,
            training_frames=(
                len(search_graph_cutoffs)
                if graph_labels is not None
                else sum(1 for _ in _training_groups(search_train_labels, labels_spec))
            ),
            buffered_training_frames=True,
            rerank_candidates=rerank_top_k,
            forward_search_beam_width=forward_search_beam_width,
            source_column_budgets=column_budgets,
            max_features_per_source_column=(feature_family_max_features_per_column),
            max_propagation_functions_per_column=(
                feature_propagation_max_functions_per_column
            ),
            adaptive_depth_promotion=adaptive_depth_promotion,
            capability_pruning=capability_pruning,
            search_max_features=search_max_features,
            search_max_working_cells=search_max_working_cells,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
        )
    else:
        logger.info(
            "search_skipped",
            reason="preselected_configuration",
            feature_families=preselected_config.feature_families,
            depth=preselected_config.depth,
            auto_annotate_features=preselected_config.auto_annotate_features,
            frozen_plan=preselected_execution_plan is not None,
        )

    def evaluate_search_trial(
        config: GraphConfig,
        *,
        rows: int,
        full_data: bool,
        stage: str,
        trial_label: str,
        estimated_feature_count: int | None,
        use_fidelity_training_frames: bool = False,
        frozen_candidate: Trial | None = None,
    ) -> Trial:
        if frozen_candidate is not None:
            if frozen_candidate.config != config:
                raise ValueError("A frozen candidate must use the same graph config")
            if frozen_candidate.execution_plan is None:
                raise ValueError("A frozen candidate must retain its execution plan")
            if not frozen_candidate.feature_columns:
                raise ValueError("A frozen candidate must retain its feature schema")
        if graph_labels is not None:
            trial_training_descriptors: tuple[Any, ...] = tuple(
                fidelity_graph_cutoffs
                if use_fidelity_training_frames
                else search_graph_cutoffs
            )
            trial_planning_labels = None
        else:
            trial_planning_labels = (
                fidelity_train_labels
                if use_fidelity_training_frames
                else search_train_labels
            )
            trial_training_descriptors = tuple(
                _training_groups(trial_planning_labels, labels_spec)
            )
        event_prefix = "trial" if stage == "screening" else f"{stage}_trial"
        checkpoint_sample_rows = None if full_data else rows
        resumed = load_checkpointed_trial(
            config,
            stage=stage,
            sample_rows=checkpoint_sample_rows,
            trial_label=trial_label,
        )
        if resumed is not None:
            return resumed
        planning_estimate = config_feature_estimates.get(config)
        logger.info(
            f"{event_prefix}_started",
            trial=trial_label,
            feature_families=config.feature_families,
            depth=config.depth,
            auto_annotate_features=config.auto_annotate_features,
            include_target_history=config.include_target_history,
            feature_family_max_columns=config.feature_family_max_columns,
            feature_family_max_features_per_column=(
                config.feature_family_max_features_per_column
            ),
            feature_propagation_max_functions_per_column=(
                config.feature_propagation_max_functions_per_column
            ),
            sample_rows=None if full_data else rows,
            estimated_feature_count=estimated_feature_count,
            estimated_output_features=(
                planning_estimate.estimated_output_features
                if planning_estimate is not None
                else None
            ),
            estimated_peak_features=(
                planning_estimate.estimated_peak_features
                if planning_estimate is not None
                else None
            ),
            estimated_total_generated_features=(
                planning_estimate.estimated_total_generated_features
                if planning_estimate is not None
                else None
            ),
            estimated_peak_working_cells=(
                planning_estimate.estimated_peak_working_cells
                if planning_estimate is not None
                else None
            ),
            training_frames=len(trial_training_descriptors),
            buffered_training_frames=True,
            frozen_candidate_plan=frozen_candidate is not None,
        )
        started = perf_counter()
        model_seconds = 0.0
        model = None
        trial_tables, trial_relationships = _topology_for_config(
            normalized_tables, normalized_relationships, config
        )
        try:
            with _table_workspace(
                connection,
                trial_tables,
                sample_rows=rows,
                random_state=random_state,
                search=True,
                fidelity=use_fidelity_training_frames,
                search_full_data=full_data,
                duckdb_memory_limit=duckdb_memory_limit,
                duckdb_max_temp_directory_size=(duckdb_max_temp_directory_size),
            ) as workspace:
                frozen_plans: list[dict[str, Any]] = []
                execution_plan = (
                    copy.deepcopy(frozen_candidate.execution_plan)
                    if frozen_candidate is not None
                    else None
                )
                search_training_descriptors = trial_training_descriptors
                if graph_labels is not None:
                    stage_validation_descriptors = tuple(
                        graph_labels.validation_cutoffs
                    )
                else:
                    stage_validation_descriptors = tuple(
                        _training_groups(search_validation_labels, labels_spec)
                    )
                if not search_training_descriptors:
                    raise ValueError("Configuration search requires training cutoffs")

                # Learn the candidate's uncertain feature operations on the latest
                # search cutoff, then replay that exact plan on the earlier cutoffs.
                # This experimental path deliberately retains every cutoff frame
                # until they are concatenated for one downstream fit.
                latest_descriptor = search_training_descriptors[-1]
                if graph_labels is not None:
                    latest_train_frame = _materialize_graph_labels_at_cutoff(
                        workspace,
                        trial_tables,
                        trial_relationships,
                        root_name,
                        config,
                        graph_labels,
                        cut_date=pd.Timestamp(latest_descriptor).to_pydatetime(),
                        split_value="train",
                        split_marker=split_marker,
                        cutoff_marker=cutoff_marker,
                        compute_period_days=compute_period_days,
                        verbose=verbose,
                        execution_plan=execution_plan,
                        frozen_plan_sink=(
                            frozen_plans if execution_plan is None else None
                        ),
                        infer_ts_periods=infer_ts_periods,
                    )
                else:
                    latest_cutoff, latest_labels = latest_descriptor
                    latest_train_frame = _materialize_at_cutoff(
                        workspace,
                        trial_tables,
                        trial_relationships,
                        root_name,
                        config,
                        latest_labels,
                        labels_spec,
                        cut_date=latest_cutoff,
                        compute_period_days=compute_period_days,
                        verbose=verbose,
                        execution_plan=execution_plan,
                        frozen_plan_sink=(
                            frozen_plans if execution_plan is None else None
                        ),
                        planning_labels=trial_planning_labels,
                        infer_ts_periods=infer_ts_periods,
                    )
                if execution_plan is None:
                    execution_plan = frozen_plans[0]
                latest_train_frame = _drop_graph_markers(
                    latest_train_frame, split_marker=split_marker
                )
                buffered_frames: dict[int, pd.DataFrame] = {
                    len(search_training_descriptors) - 1: latest_train_frame
                }
                expected_columns = tuple(latest_train_frame.columns)
                for frame_index, descriptor in enumerate(
                    search_training_descriptors[:-1]
                ):
                    if graph_labels is not None:
                        cutoff_frame = _materialize_graph_labels_at_cutoff(
                            workspace,
                            trial_tables,
                            trial_relationships,
                            root_name,
                            config,
                            graph_labels,
                            cut_date=pd.Timestamp(descriptor).to_pydatetime(),
                            split_value="train",
                            split_marker=split_marker,
                            cutoff_marker=cutoff_marker,
                            compute_period_days=compute_period_days,
                            verbose=verbose,
                            execution_plan=execution_plan,
                            infer_ts_periods=False,
                        )
                    else:
                        cutoff, cutoff_labels = descriptor
                        cutoff_frame = _materialize_at_cutoff(
                            workspace,
                            trial_tables,
                            trial_relationships,
                            root_name,
                            config,
                            cutoff_labels,
                            labels_spec,
                            cut_date=cutoff,
                            compute_period_days=compute_period_days,
                            verbose=verbose,
                            execution_plan=execution_plan,
                            infer_ts_periods=False,
                        )
                    cutoff_frame = _drop_graph_markers(
                        cutoff_frame, split_marker=split_marker
                    )
                    if set(cutoff_frame.columns) != set(expected_columns):
                        missing = set(expected_columns) - set(cutoff_frame.columns)
                        extra = set(cutoff_frame.columns) - set(expected_columns)
                        raise ValueError(
                            "Frozen GraphReduce search plan changed columns across "
                            f"cutoffs; missing={sorted(missing)[:10]}, "
                            f"extra={sorted(extra)[:10]}"
                        )
                    buffered_frames[frame_index] = cutoff_frame.loc[:, expected_columns]

                ordered_frames = [
                    buffered_frames[index]
                    for index in range(len(search_training_descriptors))
                ]
                cutoff_frame_rows = tuple(len(frame) for frame in ordered_frames)
                sampled_frames: list[pd.DataFrame] = []
                for frame_index, frame in enumerate(ordered_frames):
                    sampled_frame, _ = _sample_temporal_frame(
                        frame,
                        frame[target_column],
                        total_limit=estimator_train_rows,
                        frame_count=len(ordered_frames),
                        frame_index=frame_index,
                        task=resolved_task,
                        random_state=random_state,
                    )
                    sampled_frames.append(sampled_frame)
                sampled_cutoff_frame_rows = tuple(
                    len(frame) for frame in sampled_frames
                )
                train_frame = pd.concat(
                    sampled_frames,
                    ignore_index=True,
                    sort=False,
                )
                del sampled_frames, ordered_frames, buffered_frames, latest_train_frame
                (
                    feature_columns,
                    categorical_columns,
                    datetime_columns,
                ) = (
                    (
                        frozen_candidate.feature_columns,
                        frozen_candidate.categorical_columns,
                        frozen_candidate.datetime_columns,
                    )
                    if frozen_candidate is not None
                    else ((), (), ())
                )
                if frozen_candidate is not None:
                    missing_features = set(feature_columns) - set(train_frame.columns)
                    if missing_features:
                        raise ValueError(
                            "Frozen candidate plan did not reproduce training "
                            f"features: {sorted(missing_features)[:10]}"
                        )
                    train_x = prepare_prediction_features(
                        train_frame,
                        columns=list(feature_columns),
                        categorical_columns=set(categorical_columns),
                        datetime_columns=set(datetime_columns),
                    )
                    train_y = train_frame[target_column]
                    categorical = [
                        index
                        for index, column in enumerate(feature_columns)
                        if column in set(categorical_columns)
                    ]
                else:
                    (
                        train_x,
                        train_y,
                        categorical,
                        datetime_columns,
                    ) = prepare_training_features(
                        train_frame,
                        target=target_column,
                        excluded=excluded,
                    )
                    feature_columns = tuple(train_x.columns)
                    categorical_columns = tuple(
                        train_x.columns[index] for index in categorical
                    )
                    execution_plan = _plan_with_schema(
                        execution_plan,
                        columns=feature_columns,
                        categorical_columns=categorical_columns,
                        datetime_columns=datetime_columns,
                    )
                estimator_x, estimator_y = sample_training_rows(
                    train_x,
                    train_y,
                    limit=estimator_train_rows,
                    task=resolved_task,
                    random_state=random_state,
                )
                model, model_seconds, target_classes = _fit_final_estimator(
                    estimator_x,
                    estimator_y,
                    model_backend=model_backend,
                    task=resolved_task,
                    categorical=categorical,
                    random_state=random_state,
                    model_params=model_params,
                )
                train_rows = len(estimator_x)
                feature_count = len(feature_columns)
                del train_frame, train_x, train_y, estimator_x, estimator_y
                gc.collect()

                validation_truth: list[np.ndarray] = []
                validation_predictions: list[np.ndarray] = []
                cutoff_labels: list[str] = []
                cutoff_scores: list[float | None] = []
                validation_rows = 0
                for validation_index, descriptor in enumerate(
                    stage_validation_descriptors
                ):
                    if graph_labels is not None:
                        validation_frame = _materialize_graph_labels_at_cutoff(
                            workspace,
                            trial_tables,
                            trial_relationships,
                            root_name,
                            config,
                            graph_labels,
                            cut_date=pd.Timestamp(descriptor).to_pydatetime(),
                            split_value="validation",
                            split_marker=split_marker,
                            cutoff_marker=cutoff_marker,
                            compute_period_days=compute_period_days,
                            verbose=verbose,
                            execution_plan=execution_plan,
                            infer_ts_periods=False,
                        )
                    else:
                        validation_cutoff, validation_labels = descriptor
                        validation_frame = _materialize_at_cutoff(
                            workspace,
                            trial_tables,
                            trial_relationships,
                            root_name,
                            config,
                            validation_labels,
                            labels_spec,
                            cut_date=validation_cutoff,
                            compute_period_days=compute_period_days,
                            verbose=verbose,
                            execution_plan=execution_plan,
                            infer_ts_periods=False,
                        )
                    validation_frame = _drop_graph_markers(
                        validation_frame, split_marker=split_marker
                    )
                    missing_features = set(feature_columns) - set(
                        validation_frame.columns
                    )
                    if missing_features:
                        raise ValueError(
                            "Frozen GraphReduce plan did not reproduce training "
                            f"features: {sorted(missing_features)[:10]}"
                        )
                    validation_x = prepare_prediction_features(
                        validation_frame,
                        columns=list(feature_columns),
                        categorical_columns=set(categorical_columns),
                        datetime_columns=set(datetime_columns),
                    )
                    cutoff_truth = validation_frame[target_column].to_numpy(copy=True)
                    cutoff_prediction = _predict_values(
                        model, validation_x, task=resolved_task
                    )
                    validation_truth.append(cutoff_truth)
                    validation_predictions.append(cutoff_prediction)
                    if planner_collection:
                        cutoff_descriptor = (
                            descriptor if graph_labels is not None else descriptor[0]
                        )
                        cutoff_labels.append(
                            pd.Timestamp(cutoff_descriptor).isoformat()
                            if cutoff_descriptor is not None
                            else f"validation-{validation_index}"
                        )
                        try:
                            _cutoff_metric, cutoff_score = _score_prediction_chunks(
                                [cutoff_truth],
                                [cutoff_prediction],
                                task=resolved_task,
                                target_classes=target_classes,
                            )
                        except ValueError:
                            # A binary cutoff containing one class cannot define
                            # AUC, although the aggregate score can.
                            cutoff_score = None
                        cutoff_scores.append(cutoff_score)
                    validation_rows += len(validation_frame)
                    del validation_frame, validation_x
                    gc.collect()
                metric, score = _score_prediction_chunks(
                    validation_truth,
                    validation_predictions,
                    task=resolved_task,
                    target_classes=target_classes,
                )
                validation_standard_error = _prediction_standard_error(
                    validation_truth,
                    validation_predictions,
                    task=resolved_task,
                    target_classes=target_classes,
                )
                feature_seconds = perf_counter() - started - model_seconds
                result = Trial(
                    config=config,
                    metric=metric,
                    validation_score=score,
                    objective_score=_objective_score(metric, score),
                    feature_count=feature_count,
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                    feature_seconds=feature_seconds,
                    model_seconds=model_seconds,
                    validation_standard_error=validation_standard_error,
                    model=None,
                    feature_columns=feature_columns,
                    categorical_columns=categorical_columns,
                    datetime_columns=datetime_columns,
                    execution_plan=execution_plan,
                    stage=stage,
                    sample_rows=None if full_data else rows,
                    estimated_feature_count=estimated_feature_count,
                    cutoff_labels=tuple(cutoff_labels),
                    cutoff_scores=tuple(cutoff_scores),
                )
                save_checkpointed_trial(result, trial_label=trial_label)
                logger.info(
                    f"{event_prefix}_completed",
                    trial=trial_label,
                    metric=metric,
                    score=round(score, 6),
                    include_target_history=config.include_target_history,
                    score_standard_error=round(validation_standard_error, 6),
                    features=feature_count,
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                    feature_seconds=round(feature_seconds, 3),
                    model_seconds=round(model_seconds, 3),
                    sample_rows=None if full_data else rows,
                    training_frames=len(search_training_descriptors),
                    cutoff_frame_rows=cutoff_frame_rows,
                    sampled_cutoff_frame_rows=sampled_cutoff_frame_rows,
                    buffered_training_frames=True,
                    frozen_candidate_plan=frozen_candidate is not None,
                )
                return result
        except Exception as exc:
            if not continue_on_error:
                raise
            error_text = f"{type(exc).__name__}: {exc}"
            if type(
                exc
            ).__name__ == "OutOfMemoryException" or "max_temp_directory_size" in str(
                exc
            ):
                resource_blocks.append(
                    (
                        config.feature_families,
                        config.auto_annotate_features,
                        config.depth,
                        error_text,
                    )
                )
            logger.warning(
                f"{event_prefix}_failed",
                trial=trial_label,
                feature_families=config.feature_families,
                depth=config.depth,
                auto_annotate_features=config.auto_annotate_features,
                error_type=type(exc).__name__,
                error=str(exc),
                sample_rows=None if full_data else rows,
            )
            return Trial(
                config=config,
                metric="roc_auc" if resolved_task == "classification" else "mae",
                validation_score=float("nan"),
                objective_score=float("-inf"),
                feature_count=0,
                train_rows=0,
                validation_rows=0,
                feature_seconds=perf_counter() - started,
                model_seconds=model_seconds,
                status="failed",
                error=error_text,
                stage=stage,
                sample_rows=None if full_data else rows,
                estimated_feature_count=estimated_feature_count,
            )
        finally:
            if model is not None:
                del model
            gc.collect()

    staged_anchor_config: GraphConfig | None = None
    search_configs = () if preselected_config is not None else configs
    calibration_cache: dict[tuple[GraphConfig, int], Trial] = {}
    if (
        screening_cost_ratio_threshold is not None
        and search_configs
        and not search_full_data
        and not planner_collection
        and effective_screening_rows < sample_rows
    ):
        eligible_pilots = [
            config
            for config in search_configs
            if config in config_feature_estimates
            and (
                search_max_features is None
                or config_feature_estimates[config].estimated_output_features
                <= search_max_features
            )
        ]
        if eligible_pilots:
            simple_pilot = min(
                eligible_pilots,
                key=lambda config: (
                    config.feature_families != ("base",),
                    config.depth,
                    config.auto_annotate_features,
                    graph_config_sort_key(config),
                ),
            )
            width_order = sorted(
                eligible_pilots,
                key=lambda config: (
                    config_feature_estimates[config].estimated_output_features,
                    graph_config_sort_key(config),
                ),
            )
            pilot_configs = tuple(
                dict.fromkeys((simple_pilot, width_order[len(width_order) // 2]))
            )
            calibration_ratios: list[float] = []
            for pilot_number, config in enumerate(pilot_configs, start=1):
                pilot_rows = (effective_screening_rows, sample_rows)

                def run_calibration(rows: int) -> Trial:
                    return evaluate_search_trial(
                        config,
                        rows=rows,
                        full_data=False,
                        stage="screening_calibration",
                        trial_label=f"{pilot_number}/{len(pilot_configs)}@{rows}",
                        estimated_feature_count=(
                            config_feature_estimates[config].estimated_output_features
                        ),
                    )

                # A CatBoost calibration in the parent can initialize native
                # threads before the later fork-based graph search. Keep the
                # parent thread-clean when parallel search is requested.
                pilot_trials = (
                    _fork_process_map(run_calibration, pilot_rows, max_workers=1)
                    if search_parallelism > 1
                    else [run_calibration(rows) for rows in pilot_rows]
                )
                for rows, pilot in zip(pilot_rows, pilot_trials):
                    calibration_cache[(config, rows)] = pilot
                low, high = pilot_trials
                if low.status == "completed" and high.status == "completed":
                    low_seconds = low.feature_seconds + low.model_seconds
                    high_seconds = high.feature_seconds + high.model_seconds
                    calibration_ratios.append(high_seconds / max(low_seconds, 0.001))
            if (
                len(calibration_ratios) == len(pilot_configs)
                and max(calibration_ratios) <= screening_cost_ratio_threshold
            ):
                effective_screening_rows = sample_rows
            logger.info(
                "screening_cost_calibrated",
                pilot_configs=len(pilot_configs),
                successful_pairs=len(calibration_ratios),
                cost_ratios=tuple(round(value, 3) for value in calibration_ratios),
                threshold=screening_cost_ratio_threshold,
                selected_rows=effective_screening_rows,
            )

    # The actual row fidelity is part of the checkpoint identity. A calibration
    # decision can vary with available resources, and trials planned at two row
    # caps must never share a frozen-plan or rerank checkpoint namespace.
    if checkpoint_contract is not None and search_checkpoint is None:
        checkpoint_contract["screening_rows"] = effective_screening_rows
        try:
            search_checkpoint = SearchCheckpoint(
                search_checkpoint_path,
                contract=checkpoint_contract,
            )
            logger.info(
                "search_checkpoint_ready",
                path=str(search_checkpoint.run_directory),
                fingerprint=search_checkpoint.contract_fingerprint,
            )
        except Exception as exc:
            logger.warning(
                "search_checkpoint_unavailable",
                path=str(search_checkpoint_path),
                error_type=type(exc).__name__,
                error=str(exc),
            )
    beam_search_enabled = (
        graph_configs is None
        and preselected_config is None
        and not planner_collection
        and not staged_search_top_k
    )
    original_config_order = {
        config: index for index, config in enumerate(search_configs)
    }
    scheduled_configs = list(enumerate(search_configs, start=1))
    if search_parallelism > 1 and beam_search_enabled:
        budget_order = {budget: index for index, budget in enumerate(column_budgets)}
        scheduled_configs.sort(
            key=lambda item: (
                budget_order[item[1].feature_family_max_columns],
                len(item[1].feature_families),
                item[1].depth,
                item[0],
            )
        )

    pending_rank: tuple[int, int, int] | None = None
    pending_trials: list[tuple[GraphConfig, str, int]] = []

    def flush_pending_trials() -> None:
        nonlocal pending_trials
        if not pending_trials:
            return

        def run_pending(item: tuple[GraphConfig, str, int]) -> Trial:
            candidate, label, estimated_width = item
            calibrated = calibration_cache.get((candidate, effective_screening_rows))
            if calibrated is not None and calibrated.status == "completed":
                reused = replace(
                    calibrated,
                    stage="screening",
                    sample_rows=effective_screening_rows,
                )
                save_checkpointed_trial(reused, trial_label=label)
                return reused
            return evaluate_search_trial(
                candidate,
                rows=effective_screening_rows,
                full_data=search_full_data,
                stage="collection" if planner_collection else "screening",
                trial_label=label,
                estimated_feature_count=estimated_width,
            )

        if search_parallelism == 1 or len(pending_trials) == 1:
            completed = [run_pending(item) for item in pending_trials]
        else:
            completed = _fork_process_map(
                run_pending,
                pending_trials,
                max_workers=search_parallelism,
            )
        for completed_trial in completed:
            if completed_trial.status != "failed" or not completed_trial.error:
                continue
            if (
                "OutOfMemoryException" in completed_trial.error
                or "max_temp_directory_size" in completed_trial.error
            ):
                resource_blocks.append(
                    (
                        completed_trial.config.feature_families,
                        completed_trial.config.auto_annotate_features,
                        completed_trial.config.depth,
                        completed_trial.error,
                    )
                )
        trials.extend(completed)
        pending_trials = []

    budget_promotions: dict[int | None, tuple[GraphConfig, ...]] = {}
    for trial_number, config in scheduled_configs:
        if beam_search_enabled:
            config_rank = (
                column_budgets.index(config.feature_family_max_columns),
                len(config.feature_families),
                config.depth,
            )
        else:
            config_rank = (0, 0, 0)
        if pending_rank is not None and config_rank != pending_rank:
            flush_pending_trials()
        pending_rank = config_rank
        if (
            beam_search_enabled
            and config.feature_family_max_columns in column_budgets
            and column_budgets.index(config.feature_family_max_columns) > 0
            and config.feature_family_max_columns not in budget_promotions
        ):
            budget_index = column_budgets.index(config.feature_family_max_columns)
            previous_budget = column_budgets[budget_index - 1]
            previous_trials = [
                trial
                for trial in trials
                if trial.status == "completed"
                and trial.config.feature_family_max_columns == previous_budget
            ]
            promotions: list[Trial] = []
            if previous_trials:
                raw, complexity_recommendation = _select_trials(
                    previous_trials,
                    classification_negligible_gain=(classification_negligible_gain),
                    regression_negligible_relative_gain=(
                        regression_negligible_relative_gain
                    ),
                    complexity_uncertainty_multiplier=(
                        complexity_uncertainty_multiplier
                    ),
                )
                promotions.extend((raw, complexity_recommendation))
                ranked_previous = sorted(
                    previous_trials,
                    key=lambda trial: (
                        -trial.objective_score,
                        trial.feature_count,
                        trial.config.complexity,
                        graph_config_sort_key(trial.config),
                    ),
                )
                promotions.extend(ranked_previous)
            budget_promotions[config.feature_family_max_columns] = tuple(
                dict.fromkeys(trial.config for trial in promotions)
            )[:forward_search_beam_width]
            logger.info(
                "column_budget_promoted",
                from_budget=previous_budget,
                to_budget=config.feature_family_max_columns,
                configurations=[
                    {
                        "feature_families": promoted.feature_families,
                        "depth": promoted.depth,
                        "auto_annotate_features": (promoted.auto_annotate_features),
                    }
                    for promoted in budget_promotions[config.feature_family_max_columns]
                ],
            )
        if beam_search_enabled and not forward_candidate_allowed(
            config,
            trials,
            beam_width=forward_search_beam_width,
            feature_family_max_column_options=column_budgets,
            column_budget_promotions=budget_promotions.get(
                config.feature_family_max_columns
            ),
        ):
            trials.append(
                Trial(
                    config=config,
                    metric="roc_auc" if resolved_task == "classification" else "mae",
                    validation_score=float("nan"),
                    objective_score=float("-inf"),
                    feature_count=0,
                    train_rows=0,
                    validation_rows=0,
                    feature_seconds=0.0,
                    model_seconds=0.0,
                    status="pruned",
                    error=(
                        "Candidate is not a one-family descendant of the "
                        f"top-{forward_search_beam_width} prior-level configurations"
                    ),
                )
            )
            logger.info(
                "trial_pruned",
                trial=f"{trial_number}/{len(search_configs)}",
                feature_families=config.feature_families,
                depth=config.depth,
                auto_annotate_features=config.auto_annotate_features,
                reason="outside_forward_beam",
            )
            continue
        if (
            beam_search_enabled
            and adaptive_depth_promotion
            and not adaptive_depth_candidate_allowed(
                config,
                trials,
                classification_gain=classification_negligible_gain,
                regression_relative_gain=regression_negligible_relative_gain,
                uncertainty_multiplier=complexity_uncertainty_multiplier,
            )
        ):
            trials.append(
                Trial(
                    config=config,
                    metric=("roc_auc" if resolved_task == "classification" else "mae"),
                    validation_score=float("nan"),
                    objective_score=float("-inf"),
                    feature_count=0,
                    train_rows=0,
                    validation_rows=0,
                    feature_seconds=0.0,
                    model_seconds=0.0,
                    status="pruned",
                    error=(
                        "Candidate was not promoted by the adaptive depth "
                        "beam or lacked a meaningful deeper-hop gain"
                    ),
                    stage="screening",
                    sample_rows=(
                        None if search_full_data else effective_screening_rows
                    ),
                )
            )
            logger.info(
                "trial_pruned",
                trial=f"{trial_number}/{len(search_configs)}",
                feature_families=config.feature_families,
                depth=config.depth,
                auto_annotate_features=config.auto_annotate_features,
                reason="adaptive_depth_not_promoted",
            )
            continue
        resource_block = _resource_block_for(config, resource_blocks)
        if resource_block is not None:
            families, _annotations, minimum_depth, reason = resource_block
            trials.append(
                Trial(
                    config=config,
                    metric="roc_auc" if resolved_task == "classification" else "mae",
                    validation_score=float("nan"),
                    objective_score=float("-inf"),
                    feature_count=0,
                    train_rows=0,
                    validation_rows=0,
                    feature_seconds=0.0,
                    model_seconds=0.0,
                    status="skipped",
                    error=(
                        "Resource superset of failed configuration "
                        f"families={families}, depth={minimum_depth}: {reason}"
                    ),
                )
            )
            continue
        planning_estimate = config_feature_estimates[config]
        audit_width = max(
            planning_estimate.estimated_output_features,
            planning_estimate.estimated_peak_features,
        )
        estimated_width = _observed_feature_width_estimate(
            config,
            trials,
            audit_estimate=audit_width,
        )
        exceeds_feature_limit = (
            search_max_features is not None and estimated_width > search_max_features
        )
        exceeds_working_cell_limit = (
            search_max_working_cells is not None
            and planning_estimate.estimated_peak_working_cells is not None
            and planning_estimate.estimated_peak_working_cells
            > search_max_working_cells
        )
        if not (config.depth == 1 and config.feature_families == ("base",)) and (
            exceeds_feature_limit or exceeds_working_cell_limit
        ):
            reasons = []
            if exceeds_feature_limit:
                reasons.append(
                    "estimated root/peak feature width "
                    f"{estimated_width} exceeds "
                    f"search_max_features={search_max_features}"
                )
            if exceeds_working_cell_limit:
                reasons.append(
                    "estimated peak working cells "
                    f"{planning_estimate.estimated_peak_working_cells} exceeds "
                    "search_max_working_cells="
                    f"{search_max_working_cells}"
                )
            trials.append(
                Trial(
                    config=config,
                    metric=("roc_auc" if resolved_task == "classification" else "mae"),
                    validation_score=float("nan"),
                    objective_score=float("-inf"),
                    feature_count=0,
                    train_rows=0,
                    validation_rows=0,
                    feature_seconds=0.0,
                    model_seconds=0.0,
                    status="pruned",
                    error="; ".join(reasons),
                    stage="screening",
                    sample_rows=(
                        None if search_full_data else effective_screening_rows
                    ),
                    estimated_feature_count=estimated_width,
                )
            )
            logger.info(
                "trial_pruned",
                trial=f"{trial_number}/{len(search_configs)}",
                feature_families=config.feature_families,
                depth=config.depth,
                auto_annotate_features=config.auto_annotate_features,
                reason=(
                    "estimated_feature_width"
                    if exceeds_feature_limit
                    else "estimated_working_cells"
                ),
                estimated_features=estimated_width,
                estimated_output_features=(planning_estimate.estimated_output_features),
                estimated_peak_features=planning_estimate.estimated_peak_features,
                estimated_total_generated_features=(
                    planning_estimate.estimated_total_generated_features
                ),
                estimated_peak_working_cells=(
                    planning_estimate.estimated_peak_working_cells
                ),
                search_max_features=search_max_features,
                search_max_working_cells=search_max_working_cells,
            )
            continue
        pending_trials.append(
            (
                config,
                f"{trial_number}/{len(search_configs)}",
                estimated_width,
            )
        )

    flush_pending_trials()

    if staged_search_top_k:
        anchor_successful = [trial for trial in trials if trial.status == "completed"]
        if not anchor_successful:
            errors = "; ".join(f"{trial.config}: {trial.error}" for trial in trials[:3])
            raise RuntimeError(
                "Every published-anchor KurveRSC trial failed. " + errors
            )
        annotate_complexity(
            anchor_successful,
            classification_gain=classification_negligible_gain,
            regression_relative_gain=regression_negligible_relative_gain,
            feature_growth=drastic_feature_growth,
            uncertainty_multiplier=complexity_uncertainty_multiplier,
        )
        _anchor_best, anchor_recommended = _select_trials(
            anchor_successful,
            classification_negligible_gain=classification_negligible_gain,
            regression_negligible_relative_gain=regression_negligible_relative_gain,
            complexity_uncertainty_multiplier=complexity_uncertainty_multiplier,
        )
        if not complexity_selector:
            anchor_recommended = _anchor_best
        staged_anchor_config = anchor_recommended.config
        ranked_anchor = sorted(
            anchor_successful,
            key=lambda trial: (
                -trial.objective_score,
                trial.feature_count,
                trial.config.complexity,
                graph_config_sort_key(trial.config),
            ),
        )
        promoted_anchor = ranked_anchor[:staged_search_top_k]
        if anchor_recommended not in promoted_anchor:
            if len(promoted_anchor) >= staged_search_top_k:
                promoted_anchor[-1] = anchor_recommended
            else:
                promoted_anchor.append(anchor_recommended)
        promoted_by_config: dict[GraphConfig, Trial] = {}
        for trial in promoted_anchor:
            promoted_by_config.setdefault(trial.config, trial)
        promoted_anchor = list(promoted_by_config.values())
        logger.info(
            "staged_anchor_selected",
            candidates=len(anchor_successful),
            promoted=len(promoted_anchor),
            anchor_feature_families=staged_anchor_config.feature_families,
            anchor_depth=staged_anchor_config.depth,
            anchor_auto_annotate=staged_anchor_config.auto_annotate_features,
            anchor_target_history=staged_anchor_config.include_target_history,
            anchor_score=round(anchor_recommended.validation_score, 6),
        )

        available_by_annotation = {
            (config.auto_annotate_features, config.include_target_history): set(
                available_feature_families(
                    normalized_tables,
                    normalized_relationships,
                    feature_audit,
                    auto_annotate_features=config.auto_annotate_features,
                    include_target_history=config.include_target_history,
                )
            )
            for config in {trial.config for trial in promoted_anchor}
        }

        def evaluate_staged_batch(
            candidates: Sequence[GraphConfig], *, stage: str
        ) -> list[Trial]:
            if base_distribution_features is not None:
                candidates = tuple(
                    replace(
                        candidate,
                        base_distribution_features=bool(base_distribution_features),
                    )
                    for candidate in candidates
                )
            unique_candidates = [
                candidate
                for candidate in dict.fromkeys(candidates)
                if candidate not in original_config_order
            ]
            stage_budget = getattr(resolved_stage_budgets, stage)
            candidates_before_budget = len(unique_candidates)
            if stage_budget == 0:
                unique_candidates = []
            elif stage_budget is not None:
                unique_candidates = list(
                    diverse_collection_configs(
                        unique_candidates,
                        max_candidates=stage_budget,
                        random_state=random_state + (1 if stage == "challenger" else 2),
                    )
                )
            if stage_budget is not None:
                logger.info(
                    "stage_candidate_budget_applied",
                    stage=stage,
                    candidates_before=candidates_before_budget,
                    candidates_after=len(unique_candidates),
                    budget=stage_budget,
                )
            runnable: list[tuple[GraphConfig, str, int]] = []
            completed: list[Trial] = []
            for candidate in unique_candidates:
                original_config_order[candidate] = len(original_config_order)
                availability_key = (
                    candidate.auto_annotate_features,
                    candidate.include_target_history,
                )
                available = available_by_annotation.get(availability_key)
                if available is None:
                    available = set(
                        available_feature_families(
                            normalized_tables,
                            normalized_relationships,
                            feature_audit,
                            auto_annotate_features=(candidate.auto_annotate_features),
                            include_target_history=(candidate.include_target_history),
                        )
                    )
                    available_by_annotation[availability_key] = available
                if capability_pruning and not set(candidate.feature_families).issubset(
                    available
                ):
                    completed.append(
                        Trial(
                            config=candidate,
                            metric=(
                                "roc_auc"
                                if resolved_task == "classification"
                                else "mae"
                            ),
                            validation_score=float("nan"),
                            objective_score=float("-inf"),
                            feature_count=0,
                            train_rows=0,
                            validation_rows=0,
                            feature_seconds=0.0,
                            model_seconds=0.0,
                            status="pruned",
                            error="Modern challenger family is unavailable",
                            stage=stage,
                        )
                    )
                    continue
                estimate = estimate_config_features(
                    candidate,
                    feature_audit,
                    normalized_relationships,
                    normalized_tables,
                    root_name=root_name,
                    source_rows=planning_source_rows,
                )
                config_feature_estimates[candidate] = estimate
                audit_width = max(
                    estimate.estimated_output_features,
                    estimate.estimated_peak_features,
                )
                estimated_width = _observed_feature_width_estimate(
                    candidate,
                    trials,
                    audit_estimate=audit_width,
                )
                if (
                    search_max_features is not None
                    and estimated_width > search_max_features
                ):
                    completed.append(
                        Trial(
                            config=candidate,
                            metric=(
                                "roc_auc"
                                if resolved_task == "classification"
                                else "mae"
                            ),
                            validation_score=float("nan"),
                            objective_score=float("-inf"),
                            feature_count=0,
                            train_rows=0,
                            validation_rows=0,
                            feature_seconds=0.0,
                            model_seconds=0.0,
                            status="pruned",
                            error=(
                                f"Estimated feature width {estimated_width} "
                                "exceeds search_max_features="
                                f"{search_max_features}"
                            ),
                            stage=stage,
                            estimated_feature_count=estimated_width,
                        )
                    )
                    continue
                runnable.append(
                    (
                        candidate,
                        f"{len(runnable) + 1}/{len(unique_candidates)}",
                        estimated_width,
                    )
                )

            def run_challenger(item: tuple[GraphConfig, str, int]) -> Trial:
                candidate, label, estimated_width = item
                return evaluate_search_trial(
                    candidate,
                    rows=effective_screening_rows,
                    full_data=search_full_data,
                    stage=stage,
                    trial_label=label,
                    estimated_feature_count=estimated_width,
                )

            if search_parallelism == 1 or len(runnable) <= 1:
                completed.extend(run_challenger(item) for item in runnable)
            else:
                completed.extend(
                    _fork_process_map(
                        run_challenger,
                        runnable,
                        max_workers=search_parallelism,
                    )
                )
            trials.extend(completed)
            logger.info(
                "staged_challengers_completed",
                stage=stage,
                requested=len(unique_candidates),
                completed=sum(trial.status == "completed" for trial in completed),
                pruned=sum(trial.status == "pruned" for trial in completed),
                failed=sum(trial.status == "failed" for trial in completed),
            )
            return completed

        parents = [trial.config for trial in promoted_anchor]
        staged_depths = tuple(range(1, max_depth + 1))
        staged_expensive_depths = tuple(
            range(
                1, min(max_depth, 2 if limit_expensive_family_depth else max_depth) + 1
            )
        )
        staged_topology_options = (
            (1,)
            if max_base_table_occurrence_options is None
            else tuple(
                dict.fromkeys(int(value) for value in max_base_table_occurrence_options)
            )
        )

        def topology_variants(
            candidates: Sequence[GraphConfig],
        ) -> tuple[GraphConfig, ...]:
            return tuple(
                dict.fromkeys(
                    replace(
                        candidate,
                        max_base_table_occurrences=topology_limit,
                    )
                    for candidate in candidates
                    for topology_limit in staged_topology_options
                )
            )

        # A forward-only expansion of promoted legacy graphs cannot discover a
        # sparse modern winner such as base+state when plain base was not itself
        # promoted.  Always evaluate a small matched native-semantics seed panel
        # for every legacy annotation/budget control before layering modern
        # families onto the strongest legacy parents.
        legacy_base_controls = tuple(
            config
            for config in configs
            if config.feature_families == ("base",)
            and config.depth == 1
            and not config.include_target_history
        )
        modern_seed_candidates = topology_variants(
            staged_modern_seed_configs(
                legacy_base_controls,
                depths=staged_depths,
            )
        )
        individual_by_parent: dict[
            tuple[GraphConfig, str], tuple[GraphConfig, ...]
        ] = {}
        individual_candidates: list[GraphConfig] = list(modern_seed_candidates)
        for parent in parents:
            depth_topology_candidates = topology_variants(
                tuple(replace(parent, depth=depth) for depth in staged_depths)
            )
            individual_candidates.extend(depth_topology_candidates)
            history_candidates = topology_variants(
                (
                    replace(
                        parent,
                        include_target_history=True,
                        published_cutoff_compatibility=False,
                        categorical_tie_break="value",
                        categorical_tie_break_seed=0,
                        legacy_training_row_order=False,
                    ),
                )
            )
            individual_by_parent[(parent, "target_history")] = history_candidates
            individual_candidates.extend(history_candidates)
            for family in ("state", "context"):
                candidates = topology_variants(
                    staged_challenger_configs(
                        (parent,),
                        additions=((family,),),
                        depths=(
                            staged_depths
                            if family == "state"
                            else staged_expensive_depths
                        ),
                        include_target_history=False,
                    )
                )
                individual_by_parent[(parent, family)] = candidates
                individual_candidates.extend(candidates)
        evaluate_staged_batch(
            individual_candidates,
            stage="challenger",
        )

        trial_by_config = {
            trial.config: trial for trial in trials if trial.status == "completed"
        }
        anchor_trial_by_config = {trial.config: trial for trial in anchor_successful}
        combined_candidates: list[GraphConfig] = []
        for parent in parents:
            parent_trial = anchor_trial_by_config[parent]
            best_by_family: dict[str, Trial] = {}
            for family in ("target_history", "state", "context"):
                family_trials = [
                    trial_by_config[candidate]
                    for candidate in individual_by_parent[(parent, family)]
                    if candidate in trial_by_config
                ]
                if family_trials:
                    best_by_family[family] = min(
                        family_trials,
                        key=lambda trial: (
                            -trial.objective_score,
                            trial.feature_count,
                            graph_config_sort_key(trial.config),
                        ),
                    )
            improved = {
                family
                for family, trial in best_by_family.items()
                if _trial_meaningfully_better(
                    trial,
                    parent_trial,
                    classification_negligible_gain=(classification_negligible_gain),
                    regression_negligible_relative_gain=(
                        regression_negligible_relative_gain
                    ),
                    uncertainty_multiplier=(complexity_uncertainty_multiplier),
                )
            }
            if {"state", "context"}.issubset(improved):
                combined_candidates.extend(
                    topology_variants(
                        staged_challenger_configs(
                            (parent,),
                            additions=(("state", "context"),),
                            depths=staged_expensive_depths,
                            include_target_history=False,
                        )
                    )
                )
            if {"target_history", "state"}.issubset(improved):
                combined_candidates.extend(
                    topology_variants(
                        staged_challenger_configs(
                            (parent,),
                            additions=(("state",),),
                            depths=staged_depths,
                            include_target_history=True,
                        )
                    )
                )
            if {"target_history", "context"}.issubset(improved):
                combined_candidates.extend(
                    topology_variants(
                        staged_challenger_configs(
                            (parent,),
                            additions=(("context",),),
                            depths=staged_expensive_depths,
                            include_target_history=True,
                        )
                    )
                )
            if {"target_history", "state", "context"}.issubset(improved):
                combined_candidates.extend(
                    topology_variants(
                        staged_challenger_configs(
                            (parent,),
                            additions=(("state", "context"),),
                            depths=staged_expensive_depths,
                            include_target_history=True,
                        )
                    )
                )
        if combined_candidates:
            evaluate_staged_batch(
                combined_candidates,
                stage="combined_challenger",
            )
        else:
            logger.info(
                "staged_combined_challengers_skipped",
                reason="no_pair_of_modern_capabilities_improved_the_same_parent",
            )

    trials.sort(key=lambda trial: original_config_order.get(trial.config, len(configs)))

    if preselected_config is not None:
        selected_plan = (
            copy.deepcopy(dict(preselected_execution_plan))
            if preselected_execution_plan is not None
            else None
        )
        preselected_schema = _schema_from_plan(selected_plan or {})
        if rerank_preselected_config and preselected_schema is None:
            raise ValueError(
                "rerank_preselected_config requires a frozen feature schema"
            )
        feature_columns, categorical_columns, datetime_columns = (
            preselected_schema if preselected_schema is not None else ((), (), ())
        )
        placeholder = Trial(
            config=preselected_config,
            metric="roc_auc" if resolved_task == "classification" else "mae",
            validation_score=float("nan"),
            objective_score=0.0,
            feature_count=len(feature_columns),
            train_rows=0,
            validation_rows=0,
            feature_seconds=0.0,
            model_seconds=0.0,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            execution_plan=selected_plan,
        )
        trials = [placeholder]
        best = recommended = placeholder
        successful = [placeholder]
        categorical_refinement_parent_lineages = ()
    else:
        annotate_complexity(
            trials,
            classification_gain=classification_negligible_gain,
            regression_relative_gain=regression_negligible_relative_gain,
            feature_growth=drastic_feature_growth,
            uncertainty_multiplier=complexity_uncertainty_multiplier,
        )
        screening_successful = [
            trial for trial in trials if trial.status == "completed"
        ]
        if not screening_successful:
            errors = "; ".join(f"{trial.config}: {trial.error}" for trial in trials[:3])
            raise RuntimeError(f"Every KurveRSC trial failed. {errors}")
        _screening_best, screening_recommended = _select_trials(
            screening_successful,
            classification_negligible_gain=classification_negligible_gain,
            regression_negligible_relative_gain=regression_negligible_relative_gain,
            complexity_uncertainty_multiplier=complexity_uncertainty_multiplier,
        )
        if not complexity_selector:
            screening_recommended = _screening_best
        successful = screening_successful
        successful_stage = "screening"
        effective_confirmation_top_k = confirmation_top_k
        if resolved_stage_budgets.confirmation is not None:
            effective_confirmation_top_k = min(
                effective_confirmation_top_k,
                resolved_stage_budgets.confirmation,
            )
        if (
            not search_full_data
            and effective_confirmation_top_k > 0
            and effective_screening_rows < sample_rows
        ):
            if confidence_frontier_promotion:
                confirmation_finalists = list(
                    confidence_frontier_trials(
                        screening_successful,
                        top_k=effective_confirmation_top_k,
                        uncertainty_multiplier=complexity_uncertainty_multiplier,
                        extra_controls=4,
                    )
                )
                frontier_count = len(
                    confidence_frontier_trials(
                        screening_successful,
                        top_k=len(screening_successful),
                        uncertainty_multiplier=complexity_uncertainty_multiplier,
                        fill_budget=False,
                    )
                )
                logger.info(
                    "confirmation_confidence_frontier",
                    plausible_candidates=frontier_count,
                    promoted_candidates=len(confirmation_finalists),
                    budget=effective_confirmation_top_k,
                    unresolved=frontier_count > effective_confirmation_top_k,
                )
            elif fidelity_promotion_top_k > 0:
                confirmation_finalists = list(
                    diverse_fidelity_trials(
                        screening_successful,
                        top_k=effective_confirmation_top_k,
                        complexity_recommendation=screening_recommended,
                        structural_first=structural_promotion,
                        shallow_sparse_controls=shallow_sparse_promotion,
                    )
                )
            else:
                confirmation_finalists = list(
                    diverse_confirmation_trials(
                        screening_successful,
                        top_k=effective_confirmation_top_k,
                        complexity_recommendation=screening_recommended,
                    )
                )
            logger.info(
                "confirmation_started",
                candidates=len(confirmation_finalists),
                candidate_budget=effective_confirmation_top_k,
                screening_rows=effective_screening_rows,
                confirmation_rows=sample_rows,
                streaming=True,
            )
            confirmation_items: list[tuple[int, Trial]] = []
            for confirmation_number, screening_trial in enumerate(
                confirmation_finalists, start=1
            ):
                if (
                    search_max_features is not None
                    and screening_trial.feature_count > search_max_features
                ):
                    confirmation_trials.append(
                        Trial(
                            config=screening_trial.config,
                            metric=screening_trial.metric,
                            validation_score=float("nan"),
                            objective_score=float("-inf"),
                            feature_count=screening_trial.feature_count,
                            train_rows=0,
                            validation_rows=0,
                            feature_seconds=0.0,
                            model_seconds=0.0,
                            status="pruned",
                            error=(
                                f"Screening width {screening_trial.feature_count} "
                                "exceeds search_max_features="
                                f"{search_max_features}"
                            ),
                            stage="confirmation",
                            sample_rows=sample_rows,
                            estimated_feature_count=(screening_trial.feature_count),
                        )
                    )
                    continue
                confirmation_items.append((confirmation_number, screening_trial))

            def run_confirmation(item: tuple[int, Trial]) -> Trial:
                confirmation_number, screening_trial = item
                return evaluate_search_trial(
                    screening_trial.config,
                    rows=sample_rows,
                    full_data=False,
                    stage="confirmation",
                    trial_label=(
                        f"{confirmation_number}/{len(confirmation_finalists)}"
                    ),
                    estimated_feature_count=(screening_trial.feature_count),
                    frozen_candidate=screening_trial,
                )

            if search_parallelism == 1 or len(confirmation_items) <= 1:
                confirmation_trials.extend(
                    run_confirmation(item) for item in confirmation_items
                )
            else:
                confirmation_trials.extend(
                    _fork_process_map(
                        run_confirmation,
                        confirmation_items,
                        max_workers=search_parallelism,
                    )
                )
            confirmation_successful = [
                trial for trial in confirmation_trials if trial.status == "completed"
            ]
            if confirmation_successful:
                annotate_complexity(
                    confirmation_trials,
                    classification_gain=classification_negligible_gain,
                    regression_relative_gain=regression_negligible_relative_gain,
                    feature_growth=drastic_feature_growth,
                    uncertainty_multiplier=complexity_uncertainty_multiplier,
                )
                successful = confirmation_successful
                successful_stage = "confirmation"
            else:
                logger.warning(
                    "confirmation_failed",
                    candidates=len(confirmation_trials),
                    fallback="screening",
                )

        effective_fidelity_promotion_top_k = fidelity_promotion_top_k
        if resolved_stage_budgets.fidelity_promotion is not None:
            effective_fidelity_promotion_top_k = min(
                effective_fidelity_promotion_top_k,
                resolved_stage_budgets.fidelity_promotion,
            )
        if (
            effective_fidelity_promotion_top_k > 0
            and fidelity_training_frame_count > search_training_frame_count
        ):
            _promotion_best, promotion_recommendation = _select_trials(
                successful,
                classification_negligible_gain=classification_negligible_gain,
                regression_negligible_relative_gain=(
                    regression_negligible_relative_gain
                ),
                complexity_uncertainty_multiplier=(complexity_uncertainty_multiplier),
            )
            if not complexity_selector:
                promotion_recommendation = _promotion_best
            promotion_candidates = list(
                diverse_fidelity_trials(
                    successful,
                    top_k=effective_fidelity_promotion_top_k,
                    complexity_recommendation=promotion_recommendation,
                    structural_first=structural_promotion,
                    shallow_sparse_controls=shallow_sparse_promotion,
                )
            )
            if staged_anchor_config is not None:
                anchor_promotion = next(
                    (
                        trial
                        for trial in screening_successful
                        if trial.config == staged_anchor_config
                    ),
                    None,
                )
                if anchor_promotion is not None and anchor_promotion.config not in {
                    trial.config for trial in promotion_candidates
                }:
                    if len(promotion_candidates) >= effective_fidelity_promotion_top_k:
                        promotion_candidates[-1] = anchor_promotion
                    else:
                        promotion_candidates.append(anchor_promotion)

            logger.info(
                "fidelity_promotion_started",
                candidates=len(promotion_candidates),
                candidate_budget=effective_fidelity_promotion_top_k,
                source_training_frames=search_training_frame_count,
                promoted_training_frames=fidelity_training_frame_count,
                sample_rows=None if search_full_data else sample_rows,
                structurally_diverse=True,
            )

            def run_fidelity_promotion(
                item: tuple[int, Trial],
            ) -> Trial:
                promotion_number, source_trial = item
                return evaluate_search_trial(
                    source_trial.config,
                    rows=sample_rows,
                    full_data=search_full_data,
                    stage="fidelity_promotion",
                    trial_label=(f"{promotion_number}/{len(promotion_candidates)}"),
                    estimated_feature_count=source_trial.feature_count,
                    use_fidelity_training_frames=True,
                    frozen_candidate=source_trial,
                )

            promotion_items = list(enumerate(promotion_candidates, start=1))
            if search_parallelism == 1 or len(promotion_items) <= 1:
                fidelity_trials = [
                    run_fidelity_promotion(item) for item in promotion_items
                ]
            else:
                fidelity_trials = _fork_process_map(
                    run_fidelity_promotion,
                    promotion_items,
                    max_workers=search_parallelism,
                )
            confirmation_trials.extend(fidelity_trials)
            fidelity_successful = [
                trial for trial in fidelity_trials if trial.status == "completed"
            ]
            if fidelity_successful:
                annotate_complexity(
                    fidelity_successful,
                    classification_gain=classification_negligible_gain,
                    regression_relative_gain=regression_negligible_relative_gain,
                    feature_growth=drastic_feature_growth,
                    uncertainty_multiplier=complexity_uncertainty_multiplier,
                )
                successful = fidelity_successful
                successful_stage = "fidelity_promotion"
            else:
                logger.warning(
                    "fidelity_promotion_failed",
                    candidates=len(fidelity_trials),
                    fallback="previous_fidelity",
                )
        elif effective_fidelity_promotion_top_k > 0:
            logger.info(
                "fidelity_promotion_skipped",
                reason="no_additional_training_frames",
                source_training_frames=search_training_frame_count,
                promoted_training_frames=fidelity_training_frame_count,
            )

        categorical_refinement_parent_lineages: tuple[GraphConfig, ...] = ()
        effective_categorical_refinement_budget = (
            resolved_stage_budgets.categorical_refinement
        )
        if (
            categorical_refinement_top_k > 0
            and effective_categorical_refinement_budget != 0
        ):
            # This is a local representation refinement, not another graph
            # promotion decision. Refine the score-leading parents directly;
            # the complexity recommendation remains eligible in its original
            # form and is reconsidered after reranking. Reserving a parent
            # slot for it here can displace the second-best signal graph and
            # waste every seed assigned to that lineage.
            refinement_parents = sorted(
                successful,
                key=lambda trial: (
                    -trial.objective_score,
                    trial.feature_count,
                    trial.config.complexity,
                    graph_config_sort_key(trial.config),
                ),
            )[:categorical_refinement_top_k]
            categorical_refinement_parent_lineages = tuple(
                dict.fromkeys(
                    _categorical_refinement_lineage(trial.config)
                    for trial in refinement_parents
                )
            )
            parent_by_candidate: dict[GraphConfig, Trial] = {}
            refinement_candidates: list[GraphConfig] = []
            existing_configs = {trial.config for trial in successful}
            # Seed-major ordering gives every promoted shape the same first
            # hash policy before a second seed is allocated to any shape.
            for seed in categorical_refinement_seeds:
                for parent_trial in refinement_parents:
                    candidate = replace(
                        parent_trial.config,
                        categorical_tie_break="hash",
                        categorical_tie_break_seed=seed,
                        legacy_training_row_order=False,
                    )
                    if (
                        candidate in existing_configs
                        or candidate in parent_by_candidate
                    ):
                        continue
                    parent_by_candidate[candidate] = parent_trial
                    refinement_candidates.append(candidate)
            candidates_before_budget = len(refinement_candidates)
            if effective_categorical_refinement_budget is not None:
                refinement_candidates = refinement_candidates[
                    :effective_categorical_refinement_budget
                ]
            logger.info(
                "categorical_refinement_started",
                source_stage=successful_stage,
                parents=len(refinement_parents),
                seeds=categorical_refinement_seeds,
                candidates_before_budget=candidates_before_budget,
                candidates=len(refinement_candidates),
                candidate_budget=effective_categorical_refinement_budget,
            )

            refinement_full_data = search_full_data
            refinement_rows = (
                effective_screening_rows
                if successful_stage == "screening"
                else sample_rows
            )
            refinement_uses_fidelity_frames = successful_stage == "fidelity_promotion"

            def run_categorical_refinement(
                item: tuple[int, GraphConfig],
            ) -> Trial:
                refinement_number, candidate = item
                parent_trial = parent_by_candidate[candidate]
                return evaluate_search_trial(
                    candidate,
                    rows=refinement_rows,
                    full_data=refinement_full_data,
                    stage="categorical_refinement",
                    trial_label=(f"{refinement_number}/{len(refinement_candidates)}"),
                    estimated_feature_count=parent_trial.feature_count,
                    use_fidelity_training_frames=(refinement_uses_fidelity_frames),
                )

            refinement_items = list(enumerate(refinement_candidates, start=1))
            if search_parallelism == 1 or len(refinement_items) <= 1:
                refinement_trials = [
                    run_categorical_refinement(item) for item in refinement_items
                ]
            else:
                refinement_trials = _fork_process_map(
                    run_categorical_refinement,
                    refinement_items,
                    max_workers=search_parallelism,
                )
            confirmation_trials.extend(refinement_trials)
            refinement_successful = [
                trial for trial in refinement_trials if trial.status == "completed"
            ]
            if refinement_successful:
                annotate_complexity(
                    refinement_successful,
                    classification_gain=classification_negligible_gain,
                    regression_relative_gain=regression_negligible_relative_gain,
                    feature_growth=drastic_feature_growth,
                    uncertainty_multiplier=complexity_uncertainty_multiplier,
                )
                # Retain every unrefined parent as a no-regression control.
                successful = [*successful, *refinement_successful]
            logger.info(
                "categorical_refinement_completed",
                source_stage=successful_stage,
                requested=len(refinement_candidates),
                completed=len(refinement_successful),
                failed=sum(trial.status == "failed" for trial in refinement_trials),
            )
        best, recommended = _select_trials(
            successful,
            classification_negligible_gain=classification_negligible_gain,
            regression_negligible_relative_gain=regression_negligible_relative_gain,
            complexity_uncertainty_multiplier=complexity_uncertainty_multiplier,
        )
        if not complexity_selector:
            recommended = best
        if (
            staged_anchor_config is not None
            and recommended.config != staged_anchor_config
        ):
            anchor_trial = next(
                (trial for trial in successful if trial.config == staged_anchor_config),
                None,
            )
            if anchor_trial is not None and not _trial_meaningfully_better(
                recommended,
                anchor_trial,
                classification_negligible_gain=classification_negligible_gain,
                regression_negligible_relative_gain=(
                    regression_negligible_relative_gain
                ),
                uncertainty_multiplier=complexity_uncertainty_multiplier,
            ):
                logger.info(
                    "search_anchor_guard",
                    rejected_feature_families=recommended.config.feature_families,
                    rejected_depth=recommended.config.depth,
                    anchor_feature_families=anchor_trial.config.feature_families,
                    anchor_depth=anchor_trial.config.depth,
                    reason="challenger_did_not_clear_validation_uncertainty",
                )
                recommended = anchor_trial
        selected_plan = copy.deepcopy(recommended.execution_plan)
        logger.info(
            "search_selected",
            stage=best.stage,
            metric=best.metric,
            score=round(best.validation_score, 6),
            feature_families=best.config.feature_families,
            depth=best.config.depth,
            auto_annotate_features=best.config.auto_annotate_features,
            include_target_history=best.config.include_target_history,
            features=best.feature_count,
        )
        if recommended is not best:
            logger.info(
                "search_recommended",
                metric=recommended.metric,
                score=round(recommended.validation_score, 6),
                feature_families=recommended.config.feature_families,
                depth=recommended.config.depth,
                auto_annotate_features=recommended.config.auto_annotate_features,
                include_target_history=recommended.config.include_target_history,
                features=recommended.feature_count,
            )

    effective_rerank_top_k = rerank_top_k
    if resolved_stage_budgets.rerank is not None:
        effective_rerank_top_k = min(
            effective_rerank_top_k,
            resolved_stage_budgets.rerank,
        )
    if planner_collection or (search_only and effective_rerank_top_k == 0):
        logger.info(
            "planner_collection_completed"
            if planner_collection
            else "search_only_completed",
            candidates=len(trials),
            completed=sum(trial.status == "completed" for trial in trials),
            failed=sum(trial.status == "failed" for trial in trials),
            metric=best.metric,
            best_score=round(best.validation_score, 6),
            training_frames=search_training_frame_count,
        )
        return FitResult(
            task=resolved_task,
            metric=best.metric,
            best_trial=best,
            recommended_trial=recommended,
            trials=tuple(trials),
            confirmation_trials=tuple(confirmation_trials),
            feature_audit=feature_audit,
        )

    # Resolve the uncapped production label frames without joining their
    # feature matrices together.
    if graph_labels is not None:
        full_train_cutoffs = _select_training_cutoffs(
            graph_labels.train_cutoffs, full_training_frames
        )
        validation_descriptors = _select_training_cutoffs(
            graph_labels.validation_cutoffs, full_validation_frames
        )
        full_train_labels = full_validation_labels = None
    else:
        full_label_frame = _load_frame_source(
            labels_spec.source,
            connection=connection,
            sample_rows=sample_rows,
            random_state=random_state,
            sample=False,
            name="full_labels",
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
        )
        required = {
            *_key_parts(labels_spec.key),
            labels_spec.target,
            *([labels_spec.timestamp] if labels_spec.timestamp else []),
            *([labels_spec.split] if labels_spec.split else []),
        }
        missing = required - set(full_label_frame.columns)
        if missing:
            raise ValueError(f"Label source is missing columns: {sorted(missing)}")
        if labels_spec.split:
            external_test_labels = full_label_frame.loc[
                full_label_frame[labels_spec.split] == labels_spec.test_value
            ].copy()
        labeled = full_label_frame.dropna(subset=[labels_spec.target])
        full_train_labels, full_validation_labels = _split_labels(
            labeled,
            labels_spec,
            task=resolved_task,
            validation_fraction=validation_fraction,
            random_state=random_state,
        )
        rerank_source_train_labels = full_train_labels
        if labels_spec.timestamp:
            full_train_cutoffs = _select_training_cutoffs(
                full_train_labels[labels_spec.timestamp].dropna().tolist(),
                full_training_frames,
            )
            full_train_labels = full_train_labels.loc[
                full_train_labels[labels_spec.timestamp].isin(full_train_cutoffs)
            ]
        else:
            full_train_cutoffs = (pd.Timestamp(datetime.now()),)
        all_validation_descriptors = tuple(
            _training_groups(full_validation_labels, labels_spec)
        )
        selected_validation_cutoffs = set(
            _select_training_cutoffs(
                [descriptor[0] for descriptor in all_validation_descriptors],
                full_validation_frames,
            )
        )
        validation_descriptors = tuple(
            descriptor
            for descriptor in all_validation_descriptors
            if pd.Timestamp(descriptor[0]) in selected_validation_cutoffs
        )
        del full_label_frame, labeled
        gc.collect()

    rerank_trials: list[Trial] = []
    if (
        (preselected_config is None or rerank_preselected_config)
        and effective_rerank_top_k > 0
    ):
        if graph_labels is not None:
            chronological_train = tuple(sorted(set(graph_labels.train_cutoffs)))
            consecutive_folds: list[tuple[tuple[Any, ...], tuple[Any, ...]]] = [
                (
                    chronological_train[: index + 1],
                    (chronological_train[index + 1],),
                )
                for index in range(max(0, len(chronological_train) - 1))
            ]
            final_fold = (
                chronological_train,
                tuple(graph_labels.validation_cutoffs),
            )
        elif labels_spec.timestamp and rerank_source_train_labels is not None:
            chronological_train = tuple(
                sorted(
                    _training_groups(rerank_source_train_labels, labels_spec),
                    key=lambda item: pd.Timestamp(item[0]),
                )
            )
            consecutive_folds = [
                (
                    chronological_train[: index + 1],
                    (chronological_train[index + 1],),
                )
                for index in range(max(0, len(chronological_train) - 1))
            ]
            final_fold = (chronological_train, validation_descriptors)
        else:
            chronological_train = ()
            consecutive_folds = []
            final_fold = None

        prior_fold_count = max(0, rerank_cutoff_frames - 1)
        selected_prior_folds = (
            consecutive_folds[-prior_fold_count:] if prior_fold_count else []
        )
        rerank_folds = [
            *selected_prior_folds,
            *([final_fold] if final_fold is not None else []),
        ]
        rerank_folds = [
            (tuple(train_descriptors), tuple(fold_validation))
            for train_descriptors, fold_validation in rerank_folds
            if train_descriptors and fold_validation
        ]
        if graph_labels is None and labels_spec.available_at:
            mature_folds = []
            for train_descriptors,fold_validation in rerank_folds:
                boundary = min(pd.Timestamp(date) for date,_ in fold_validation)
                visible = tuple((date,frame.loc[pd.to_datetime(frame[labels_spec.available_at]).le(boundary)].copy())
                    for date,frame in train_descriptors)
                visible = tuple((date,frame) for date,frame in visible if not frame.empty)
                if visible:
                    mature_folds.append((visible,fold_validation))
            rerank_folds = mature_folds

        if not rerank_folds:
            if rerank_preselected_config:
                raise ValueError(
                    "No chronological folds are available to rerank the frozen plan"
                )
            logger.info(
                "rerank_skipped",
                reason="no_temporal_folds",
                available_folds=len(rerank_folds),
                requested_folds=rerank_cutoff_frames,
            )
        else:
            ranked_with_categorical_variants = sorted(
                successful,
                key=lambda trial: (
                    -trial.objective_score,
                    trial.feature_count,
                    trial.config.complexity,
                    graph_config_sort_key(trial.config),
                ),
            )
            best_by_categorical_lineage: dict[GraphConfig, Trial] = {}
            for trial in ranked_with_categorical_variants:
                best_by_categorical_lineage.setdefault(
                    _categorical_refinement_lineage(trial.config), trial
                )
            ranked_screening = list(best_by_categorical_lineage.values())
            screening_best = ranked_screening[0]

            finalists: list[Trial] = []
            finalist_lineages: set[GraphConfig] = set()

            def add_finalist(trial: Trial | None) -> None:
                if trial is None or len(finalists) >= effective_rerank_top_k:
                    return
                lineage = _categorical_refinement_lineage(trial.config)
                if lineage in finalist_lineages:
                    return
                finalists.append(trial)
                finalist_lineages.add(lineage)

            recommendation_lineage = _categorical_refinement_lineage(recommended.config)
            recommendation_trial = best_by_categorical_lineage.get(
                recommendation_lineage
            )
            if confidence_frontier_promotion:
                frontier = confidence_frontier_trials(
                    ranked_screening,
                    top_k=len(ranked_screening),
                    uncertainty_multiplier=complexity_uncertainty_multiplier,
                    fill_budget=False,
                )
                selected_frontier = confidence_frontier_trials(
                    ranked_screening,
                    top_k=effective_rerank_top_k,
                    uncertainty_multiplier=complexity_uncertainty_multiplier,
                    extra_controls=4,
                )
                for trial in selected_frontier:
                    add_finalist(trial)
                logger.info(
                    "rerank_confidence_frontier",
                    plausible_candidates=len(frontier),
                    promoted_candidates=len(finalists),
                    budget=effective_rerank_top_k,
                    unresolved=len(frontier) > effective_rerank_top_k,
                )
            else:
                include_complexity_challenger = (
                    complexity_selector
                    and recommendation_trial is not None
                    and recommendation_trial.feature_count * drastic_feature_growth
                    <= screening_best.feature_count
                )
                if include_complexity_challenger and not shallow_sparse_promotion:
                    add_finalist(recommendation_trial)
                add_finalist(screening_best)
                if structural_promotion:
                    for trial in diverse_fidelity_trials(
                        ranked_screening,
                        top_k=effective_rerank_top_k,
                        complexity_recommendation=(
                            recommendation_trial if complexity_selector else None
                        ),
                        structural_first=True,
                        shallow_sparse_controls=shallow_sparse_promotion,
                    ):
                        add_finalist(trial)
                for lineage in categorical_refinement_parent_lineages:
                    add_finalist(best_by_categorical_lineage.get(lineage))
                if complexity_selector:
                    add_finalist(recommendation_trial)
                if staged_anchor_config is not None:
                    add_finalist(
                        best_by_categorical_lineage.get(
                            _categorical_refinement_lineage(staged_anchor_config)
                        )
                    )
                if not structural_promotion:
                    for trial in diverse_fidelity_trials(
                        ranked_screening,
                        top_k=effective_rerank_top_k,
                        complexity_recommendation=(
                            recommendation_trial if complexity_selector else None
                        ),
                    ):
                        add_finalist(trial)
                for trial in ranked_screening:
                    add_finalist(trial)
            logger.info(
                "rerank_started",
                candidates=len(finalists),
                raw_top_k=effective_rerank_top_k,
                categorical_variants=len(ranked_with_categorical_variants),
                structural_lineages=len(ranked_screening),
                cutoff_folds=len(rerank_folds),
                stability_penalty=rerank_stability_penalty,
                temporal_estimator_mode=temporal_estimator_mode,
                production_training_frames=full_training_frames,
                full_data=True,
                streaming=True,
            )

            for finalist_number, screening_trial in enumerate(finalists, start=1):
                config = screening_trial.config
                trial_label = f"{finalist_number}/{len(finalists)}"
                resumed = load_checkpointed_trial(
                    config,
                    stage="rerank",
                    sample_rows=None,
                    trial_label=trial_label,
                )
                if resumed is not None:
                    rerank_trials.append(resumed)
                    continue
                rerank_started = perf_counter()
                rerank_model_seconds = 0.0
                fold_scores: list[float] = []
                fold_standard_errors: list[float] = []
                fold_feature_counts: list[int] = []
                rerank_train_rows = 0
                rerank_validation_rows = 0
                fold_model = None
                rerank_tables, rerank_relationships = _topology_for_config(
                    normalized_tables, normalized_relationships, config
                )
                logger.info(
                    "rerank_trial_started",
                    trial=trial_label,
                    feature_families=config.feature_families,
                    depth=config.depth,
                    auto_annotate_features=config.auto_annotate_features,
                    include_target_history=config.include_target_history,
                    cutoff_folds=len(rerank_folds),
                    temporal_estimator_mode=temporal_estimator_mode,
                    production_training_frames=full_training_frames,
                )
                try:
                    rerank_metric: str | None = None
                    for (
                        fold_train_descriptors,
                        fold_validation_descriptors,
                    ) in rerank_folds:
                        with _table_workspace(
                            connection,
                            rerank_tables,
                            sample_rows=sample_rows,
                            random_state=random_state,
                            search=False,
                            duckdb_memory_limit=duckdb_memory_limit,
                            duckdb_max_temp_directory_size=(
                                duckdb_max_temp_directory_size
                            ),
                        ) as workspace:
                            fold_plan = copy.deepcopy(
                                screening_trial.execution_plan or {}
                            )
                            fold_feature_columns = screening_trial.feature_columns
                            fold_categorical_columns = (
                                screening_trial.categorical_columns
                            )
                            fold_datetime_columns = screening_trial.datetime_columns
                            if not fold_feature_columns:
                                raise ValueError(
                                    "Rerank finalist is missing its frozen feature schema"
                                )
                            fold_categorical = [
                                index
                                for index, column in enumerate(fold_feature_columns)
                                if column in set(fold_categorical_columns)
                            ]
                            effective_fold_train_descriptors = tuple(
                                fold_train_descriptors
                            )
                            if full_training_frames is not None:
                                effective_fold_train_descriptors = (
                                    effective_fold_train_descriptors[
                                        -full_training_frames:
                                    ]
                                )
                            fold_train_parts: list[pd.DataFrame] = []
                            fold_label_parts: list[pd.Series] = []
                            fold_models: list[Any] = []
                            fold_target_classes = target_class_universe
                            for train_number, fold_train_descriptor in enumerate(
                                effective_fold_train_descriptors,
                                start=1,
                            ):
                                if graph_labels is not None:
                                    fold_train_frame = (
                                        _materialize_graph_labels_at_cutoff(
                                            workspace,
                                            rerank_tables,
                                            rerank_relationships,
                                            root_name,
                                            config,
                                            graph_labels,
                                            cut_date=pd.Timestamp(
                                                fold_train_descriptor
                                            ).to_pydatetime(),
                                            split_value="train",
                                            split_marker=split_marker,
                                            cutoff_marker=cutoff_marker,
                                            compute_period_days=compute_period_days,
                                            verbose=verbose,
                                            execution_plan=fold_plan,
                                            infer_ts_periods=False,
                                        )
                                    )
                                else:
                                    fold_train_cutoff, fold_train_labels = (
                                        fold_train_descriptor
                                    )
                                    fold_train_frame = _materialize_at_cutoff(
                                        workspace,
                                        rerank_tables,
                                        rerank_relationships,
                                        root_name,
                                        config,
                                        fold_train_labels,
                                        labels_spec,
                                        cut_date=fold_train_cutoff,
                                        compute_period_days=compute_period_days,
                                        verbose=verbose,
                                        execution_plan=fold_plan,
                                        infer_ts_periods=False,
                                    )
                                fold_train_frame = _drop_graph_markers(
                                    fold_train_frame,
                                    split_marker=split_marker,
                                )
                                missing_features = set(fold_feature_columns) - set(
                                    fold_train_frame.columns
                                )
                                if missing_features:
                                    raise ValueError(
                                        "Frozen rerank plan did not reproduce "
                                        "training features: "
                                        f"{sorted(missing_features)[:10]}"
                                    )
                                fold_train_x = prepare_prediction_features(
                                    fold_train_frame,
                                    columns=list(fold_feature_columns),
                                    categorical_columns=set(fold_categorical_columns),
                                    datetime_columns=set(fold_datetime_columns),
                                )
                                fold_train_y = fold_train_frame[target_column]
                                if temporal_estimator_mode == "frame_ensemble":
                                    (
                                        fold_estimator_x,
                                        fold_estimator_y,
                                    ) = sample_training_rows(
                                        fold_train_x,
                                        fold_train_y,
                                        limit=estimator_train_rows,
                                        task=resolved_task,
                                        random_state=random_state + train_number,
                                    )
                                    (
                                        frame_model,
                                        fold_seconds,
                                        frame_target_classes,
                                    ) = _fit_temporal_frame_estimator(
                                        fold_estimator_x,
                                        fold_estimator_y,
                                        model_backend=model_backend,
                                        task=resolved_task,
                                        categorical=fold_categorical,
                                        random_state=random_state,
                                        model_params=model_params,
                                        target_classes=fold_target_classes,
                                    )
                                    if (
                                        fold_target_classes
                                        and frame_target_classes != fold_target_classes
                                    ):
                                        raise ValueError(
                                            "target classes changed across rerank frames"
                                        )
                                    fold_target_classes = (
                                        frame_target_classes or fold_target_classes
                                    )
                                    fold_models.append(frame_model)
                                    rerank_model_seconds += fold_seconds
                                    rerank_train_rows += len(fold_estimator_x)
                                    del fold_estimator_x, fold_estimator_y
                                else:
                                    (
                                        fold_train_x,
                                        fold_train_y,
                                    ) = _sample_temporal_frame(
                                        fold_train_x,
                                        fold_train_y,
                                        total_limit=estimator_train_rows,
                                        frame_count=len(
                                            effective_fold_train_descriptors
                                        ),
                                        frame_index=train_number - 1,
                                        task=resolved_task,
                                        random_state=random_state,
                                    )
                                    fold_train_parts.append(fold_train_x)
                                    fold_label_parts.append(fold_train_y)
                                del fold_train_frame, fold_train_x, fold_train_y

                            if temporal_estimator_mode == "frame_ensemble":
                                fold_model = FrameEnsemble(
                                    fold_models,
                                    task=resolved_task,
                                    target_classes=fold_target_classes,
                                )
                            else:
                                fold_estimator_x = pd.concat(
                                    fold_train_parts,
                                    ignore_index=True,
                                )
                                fold_estimator_y = pd.concat(
                                    fold_label_parts,
                                    ignore_index=True,
                                )
                                (
                                    fold_estimator_x,
                                    fold_estimator_y,
                                ) = sample_training_rows(
                                    fold_estimator_x,
                                    fold_estimator_y,
                                    limit=estimator_train_rows,
                                    task=resolved_task,
                                    random_state=random_state,
                                )
                                (
                                    fold_model,
                                    fold_seconds,
                                    fold_target_classes,
                                ) = _fit_final_estimator(
                                    fold_estimator_x,
                                    fold_estimator_y,
                                    model_backend=model_backend,
                                    task=resolved_task,
                                    categorical=fold_categorical,
                                    random_state=random_state,
                                    model_params=model_params,
                                )
                                rerank_model_seconds += fold_seconds
                                rerank_train_rows += len(fold_estimator_x)
                                del fold_estimator_x, fold_estimator_y
                            fold_feature_counts.append(len(fold_feature_columns))
                            del fold_train_parts, fold_label_parts, fold_models
                            gc.collect()

                            fold_truth: list[np.ndarray] = []
                            fold_predictions: list[np.ndarray] = []
                            for (
                                fold_validation_descriptor
                            ) in fold_validation_descriptors:
                                if graph_labels is not None:
                                    fold_validation_frame = (
                                        _materialize_graph_labels_at_cutoff(
                                            workspace,
                                            rerank_tables,
                                            rerank_relationships,
                                            root_name,
                                            config,
                                            graph_labels,
                                            cut_date=pd.Timestamp(
                                                fold_validation_descriptor
                                            ).to_pydatetime(),
                                            split_value="validation",
                                            split_marker=split_marker,
                                            cutoff_marker=cutoff_marker,
                                            compute_period_days=compute_period_days,
                                            verbose=verbose,
                                            execution_plan=fold_plan,
                                            infer_ts_periods=False,
                                        )
                                    )
                                else:
                                    (
                                        fold_validation_cutoff,
                                        fold_validation_labels,
                                    ) = fold_validation_descriptor
                                    fold_validation_frame = _materialize_at_cutoff(
                                        workspace,
                                        rerank_tables,
                                        rerank_relationships,
                                        root_name,
                                        config,
                                        fold_validation_labels,
                                        labels_spec,
                                        cut_date=fold_validation_cutoff,
                                        compute_period_days=compute_period_days,
                                        verbose=verbose,
                                        execution_plan=fold_plan,
                                        infer_ts_periods=False,
                                    )
                                fold_validation_frame = _drop_graph_markers(
                                    fold_validation_frame,
                                    split_marker=split_marker,
                                )
                                missing_features = set(fold_feature_columns) - set(
                                    fold_validation_frame.columns
                                )
                                if missing_features:
                                    raise ValueError(
                                        "Frozen rerank plan did not reproduce "
                                        "training features: "
                                        f"{sorted(missing_features)[:10]}"
                                    )
                                fold_validation_x = prepare_prediction_features(
                                    fold_validation_frame,
                                    columns=list(fold_feature_columns),
                                    categorical_columns=set(fold_categorical_columns),
                                    datetime_columns=set(fold_datetime_columns),
                                )
                                fold_truth.append(
                                    fold_validation_frame[target_column].to_numpy(
                                        copy=True
                                    )
                                )
                                fold_predictions.append(
                                    _predict_values(
                                        fold_model,
                                        fold_validation_x,
                                        task=resolved_task,
                                    )
                                )
                                rerank_validation_rows += len(fold_validation_frame)
                                del fold_validation_frame, fold_validation_x
                                gc.collect()
                            fold_metric, fold_score = _score_prediction_chunks(
                                fold_truth,
                                fold_predictions,
                                task=resolved_task,
                                target_classes=fold_target_classes,
                            )
                            fold_standard_error = _prediction_standard_error(
                                fold_truth,
                                fold_predictions,
                                task=resolved_task,
                                target_classes=fold_target_classes,
                            )
                            if (
                                rerank_metric is not None
                                and rerank_metric != fold_metric
                            ):
                                raise ValueError(
                                    "Rerank folds produced inconsistent metrics"
                                )
                            rerank_metric = fold_metric
                            fold_scores.append(fold_score)
                            fold_standard_errors.append(fold_standard_error)
                            del fold_model
                            fold_model = None
                        gc.collect()

                    mean_score = float(np.mean(fold_scores))
                    score_std = float(np.std(fold_scores))
                    objective_score, score_standard_error = _rerank_objective(
                        rerank_metric or screening_trial.metric,
                        mean_score,
                        score_std,
                        len(fold_scores),
                        rerank_stability_penalty,
                    )
                    recent_score = fold_scores[-1]
                    recent_standard_error = fold_standard_errors[-1]
                    rerank_trial = Trial(
                        config=config,
                        metric=rerank_metric or screening_trial.metric,
                        validation_score=mean_score,
                        objective_score=objective_score,
                        feature_count=max(fold_feature_counts),
                        train_rows=rerank_train_rows,
                        validation_rows=rerank_validation_rows,
                        feature_seconds=(
                            perf_counter() - rerank_started - rerank_model_seconds
                        ),
                        model_seconds=rerank_model_seconds,
                        validation_standard_error=score_standard_error,
                        feature_columns=screening_trial.feature_columns,
                        categorical_columns=screening_trial.categorical_columns,
                        datetime_columns=screening_trial.datetime_columns,
                        execution_plan=copy.deepcopy(screening_trial.execution_plan),
                        note=(
                            f"recent={recent_score:.6f}, "
                            f"recent_se={recent_standard_error:.6f}, "
                            f"historical_mean="
                            f"{np.mean(fold_scores[:-1]) if len(fold_scores) > 1 else recent_score:.6f}, "
                            f"walk_forward_mean={mean_score:.6f}, "
                            f"walk_forward_std={score_std:.6f}, "
                            f"walk_forward_se={score_standard_error:.6f}, "
                            f"estimator_mode={temporal_estimator_mode}, "
                            f"production_frames={full_training_frames}, "
                            f"folds={len(fold_scores)}"
                        ),
                        stage="rerank",
                        sample_rows=None,
                        cutoff_labels=tuple(
                            "recent"
                            if index == len(fold_scores) - 1
                            else f"historical-{index + 1}"
                            for index in range(len(fold_scores))
                        ),
                        cutoff_scores=tuple(fold_scores),
                        cutoff_standard_errors=tuple(fold_standard_errors),
                    )
                    rerank_trials.append(rerank_trial)
                    save_checkpointed_trial(
                        rerank_trial,
                        trial_label=trial_label,
                    )
                    logger.info(
                        "rerank_trial_completed",
                        trial=trial_label,
                        metric=rerank_trial.metric,
                        score=round(mean_score, 6),
                        score_standard_error=round(score_standard_error, 6),
                        recent_score=round(recent_score, 6),
                        recent_standard_error=round(recent_standard_error, 6),
                        historical_score=round(
                            float(np.mean(fold_scores[:-1]))
                            if len(fold_scores) > 1
                            else recent_score,
                            6,
                        ),
                        walk_forward_mean=round(mean_score, 6),
                        walk_forward_std=round(score_std, 6),
                        walk_forward_standard_error=round(score_standard_error, 6),
                        objective_score=round(rerank_trial.objective_score, 6),
                        cutoff_folds=len(fold_scores),
                        temporal_estimator_mode=temporal_estimator_mode,
                        production_training_frames=full_training_frames,
                        features=rerank_trial.feature_count,
                        train_rows=rerank_trial.train_rows,
                        validation_rows=rerank_trial.validation_rows,
                    )
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    rerank_trials.append(
                        Trial(
                            config=config,
                            metric=screening_trial.metric,
                            validation_score=float("nan"),
                            objective_score=float("-inf"),
                            feature_count=0,
                            train_rows=rerank_train_rows,
                            validation_rows=rerank_validation_rows,
                            feature_seconds=perf_counter() - rerank_started,
                            model_seconds=rerank_model_seconds,
                            status="failed",
                            error=f"{type(exc).__name__}: {exc}",
                            stage="rerank",
                            sample_rows=None,
                        )
                    )
                    logger.warning(
                        "rerank_trial_failed",
                        trial=f"{finalist_number}/{len(finalists)}",
                        feature_families=config.feature_families,
                        depth=config.depth,
                        auto_annotate_features=config.auto_annotate_features,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                finally:
                    if fold_model is not None:
                        del fold_model
                    gc.collect()

            successful_reranks = [
                trial for trial in rerank_trials if trial.status == "completed"
            ]
            if successful_reranks:
                annotate_complexity(
                    successful_reranks,
                    classification_gain=classification_negligible_gain,
                    regression_relative_gain=regression_negligible_relative_gain,
                    feature_growth=drastic_feature_growth,
                    uncertainty_multiplier=complexity_uncertainty_multiplier,
                )
                (
                    best,
                    recent_safe_best,
                    recommended,
                ) = _select_temporal_rerank_trials(
                    successful_reranks,
                    classification_negligible_gain=classification_negligible_gain,
                    regression_negligible_relative_gain=(
                        regression_negligible_relative_gain
                    ),
                    complexity_uncertainty_multiplier=(
                        complexity_uncertainty_multiplier
                    ),
                    complexity_selector=complexity_selector,
                    complexity_classification_loss_cap=(
                        rerank_complexity_classification_loss_cap
                    ),
                    complexity_regression_relative_loss_cap=(
                        rerank_complexity_regression_relative_loss_cap
                    ),
                    drastic_feature_growth=drastic_feature_growth,
                    nested_depth_safety=rerank_nested_depth_safety,
                    practical_recent_ties=rerank_practical_recent_ties,
                )
                # The release-lineage anchor is a full-fidelity control, not
                # an incumbent with an asymmetric significance advantage.
                # Once every finalist has identical walk-forward evidence,
                # the temporal selector (including its recency and complexity
                # guards) is authoritative.
                selected_plan = copy.deepcopy(recommended.execution_plan)
                if recent_safe_best is not best:
                    logger.info(
                        "rerank_recent_veto",
                        metric=best.metric,
                        score=round(best.validation_score, 6),
                        recent_score=round(float(best.cutoff_scores[-1]), 6),
                        feature_families=best.config.feature_families,
                        depth=best.config.depth,
                        auto_annotate_features=(best.config.auto_annotate_features),
                        features=best.feature_count,
                        replacement_feature_families=(
                            recent_safe_best.config.feature_families
                        ),
                        replacement_depth=recent_safe_best.config.depth,
                        replacement_recent_score=round(
                            float(recent_safe_best.cutoff_scores[-1]), 6
                        ),
                    )
                if recommended is not recent_safe_best:
                    logger.info(
                        "rerank_complexity_guard",
                        raw_features=recent_safe_best.feature_count,
                        selected_features=recommended.feature_count,
                        raw_score=round(recent_safe_best.validation_score, 6),
                        selected_score=round(recommended.validation_score, 6),
                        drastic_feature_growth=drastic_feature_growth,
                        classification_loss_cap=(
                            rerank_complexity_classification_loss_cap
                        ),
                        regression_relative_loss_cap=(
                            rerank_complexity_regression_relative_loss_cap
                        ),
                        nested_depth_safety=rerank_nested_depth_safety,
                    )
                logger.info(
                    "rerank_selected",
                    metric=recommended.metric,
                    score=round(recommended.validation_score, 6),
                    score_standard_error=round(
                        recommended.validation_standard_error, 6
                    ),
                    feature_families=recommended.config.feature_families,
                    depth=recommended.config.depth,
                    auto_annotate_features=(recommended.config.auto_annotate_features),
                    include_target_history=(recommended.config.include_target_history),
                    nested_depth_safety=rerank_nested_depth_safety,
                    features=recommended.feature_count,
                    complexity_selector=complexity_selector,
                )

    if search_only:
        logger.info(
            "search_only_completed",
            candidates=len(trials),
            rerank_candidates=len(rerank_trials),
            completed=sum(trial.status == "completed" for trial in trials),
            rerank_completed=sum(
                trial.status == "completed" for trial in rerank_trials
            ),
            metric=recommended.metric,
            best_score=round(recommended.validation_score, 6),
        )
        return FitResult(
            task=resolved_task,
            metric=recommended.metric,
            best_trial=best,
            recommended_trial=recommended,
            trials=tuple(trials),
            confirmation_trials=tuple(confirmation_trials),
            rerank_trials=tuple(rerank_trials),
            feature_audit=feature_audit,
        )

    production_training_frames = len(full_train_cutoffs)
    validation_frame_count = len(validation_descriptors)
    total_fit_frames = production_training_frames + validation_frame_count
    iterations_per_model = (
        int((model_params or {}).get("iterations", 300))
        if resolved_final_model_backend == "catboost"
        else None
    )
    # TabFM's published ICL context is explicitly bounded. Sampling before its
    # context builder means a CatBoost search can remain full-fidelity while
    # the selected production frame stays inexpensive and reproducible.
    per_frame_limit = (
        100
        if resolved_final_model_backend == "tabfm" and estimator_train_rows is None
        else estimator_train_rows
    )
    logger.info(
        "temporal_refit_started",
        feature_families=recommended.config.feature_families,
        depth=recommended.config.depth,
        auto_annotate_features=recommended.config.auto_annotate_features,
        include_target_history=recommended.config.include_target_history,
        training_frames=production_training_frames,
        validation_frames=validation_frame_count,
        iterations_per_model=iterations_per_model,
        temporal_estimator_mode=temporal_estimator_mode,
        estimator_models=(
            1 if temporal_estimator_mode == "pooled" else total_fit_frames
        ),
        source_frames=total_fit_frames,
        frozen_plan=selected_plan is not None,
    )

    schema = _schema_from_plan(selected_plan or {})
    feature_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    datetime_columns: tuple[str, ...] = ()
    categorical: list[int] = []
    final_model = None
    validation_model = None
    training_models: list[Any] = []
    validation_frame_models: list[Any] = []
    pooled_training_features: list[pd.DataFrame] = []
    pooled_training_labels: list[pd.Series] = []
    pooled_validation_features: list[pd.DataFrame] = []
    pooled_validation_labels: list[pd.Series] = []
    target_classes: tuple[Any, ...] = target_class_universe
    final_model_seconds = 0.0
    train_rows = 0
    source_train_rows = 0
    validation_rows = 0
    final_estimator_rows = 0
    validation_truth: list[np.ndarray] = []
    validation_predictions: list[np.ndarray] = []
    production_tables, production_relationships = _topology_for_config(
        normalized_tables, normalized_relationships, recommended.config
    )

    def fit_production_estimator(
        features: pd.DataFrame,
        labels: pd.Series,
        *,
        target_classes: tuple[Any, ...],
    ) -> tuple[Any, float, tuple[Any, ...]]:
        return _fit_temporal_frame_estimator(
            features,
            labels,
            model_backend=resolved_final_model_backend,
            task=resolved_task,
            categorical=categorical,
            random_state=random_state,
            model_params=model_params,
            target_classes=target_classes,
            iterations=iterations_per_model,
        )

    def materialize_production_frame(
        workspace: _Workspace,
        *,
        descriptor: Any,
        split_value: str,
        allow_plan_learning: bool,
    ) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        nonlocal selected_plan
        frozen: list[dict[str, Any]] = []
        if graph_labels is not None:
            frame = _materialize_graph_labels_at_cutoff(
                workspace,
                production_tables,
                production_relationships,
                root_name,
                recommended.config,
                graph_labels,
                cut_date=pd.Timestamp(descriptor).to_pydatetime(),
                split_value=split_value,
                split_marker=split_marker,
                cutoff_marker=cutoff_marker,
                compute_period_days=compute_period_days,
                verbose=verbose,
                execution_plan=selected_plan,
                frozen_plan_sink=frozen if allow_plan_learning else None,
                infer_ts_periods=infer_ts_periods and selected_plan is None,
            )
        else:
            cutoff, labels = descriptor
            frame = _materialize_at_cutoff(
                workspace,
                production_tables,
                production_relationships,
                root_name,
                recommended.config,
                labels,
                labels_spec,
                cut_date=cutoff,
                compute_period_days=compute_period_days,
                verbose=verbose,
                execution_plan=selected_plan,
                frozen_plan_sink=frozen if allow_plan_learning else None,
                planning_labels=full_train_labels if allow_plan_learning else None,
                infer_ts_periods=infer_ts_periods and selected_plan is None,
            )
        if selected_plan is None and frozen:
            selected_plan = frozen[0]
        return _drop_graph_markers(frame, split_marker=split_marker), selected_plan

    if graph_labels is not None:
        train_descriptors: Iterable[Any] = full_train_cutoffs
    else:
        train_descriptors = tuple(_training_groups(full_train_labels, labels_spec))

    for frame_number, descriptor in enumerate(train_descriptors, start=1):
        with _table_workspace(
            connection,
            production_tables,
            sample_rows=sample_rows,
            random_state=random_state,
            search=False,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
        ) as workspace:
            frame, _ = materialize_production_frame(
                workspace,
                descriptor=descriptor,
                split_value="train",
                allow_plan_learning=selected_plan is None,
            )
            source_train_rows += len(frame)
            if schema is None:
                x, y, categorical, datetime_columns = prepare_training_features(
                    frame,
                    target=target_column,
                    excluded=excluded,
                )
                feature_columns = tuple(x.columns)
                categorical_columns = tuple(x.columns[index] for index in categorical)
                selected_plan = _plan_with_schema(
                    selected_plan,
                    columns=feature_columns,
                    categorical_columns=categorical_columns,
                    datetime_columns=datetime_columns,
                )
                schema = (
                    feature_columns,
                    categorical_columns,
                    datetime_columns,
                )
            else:
                feature_columns, categorical_columns, datetime_columns = schema
                missing_features = set(feature_columns) - set(frame.columns)
                if missing_features:
                    raise ValueError(
                        "Frozen GraphReduce plan did not reproduce production "
                        f"features: {sorted(missing_features)[:10]}"
                    )
                x = prepare_prediction_features(
                    frame,
                    columns=list(feature_columns),
                    categorical_columns=set(categorical_columns),
                    datetime_columns=set(datetime_columns),
                )
                y = frame[target_column]
                categorical = [
                    index
                    for index, column in enumerate(feature_columns)
                    if column in set(categorical_columns)
                ]
            if temporal_estimator_mode == "pooled":
                estimator_x, estimator_y = _sample_temporal_frame(
                    x,
                    y,
                    total_limit=per_frame_limit,
                    frame_count=production_training_frames,
                    frame_index=frame_number - 1,
                    task=resolved_task,
                    random_state=random_state,
                )
                pooled_training_features.append(estimator_x.reset_index(drop=True))
                pooled_training_labels.append(estimator_y.reset_index(drop=True))
                del frame, x, y, estimator_x, estimator_y
            else:
                estimator_x, estimator_y = sample_training_rows(
                    x,
                    y,
                    limit=per_frame_limit,
                    task=resolved_task,
                    random_state=random_state + frame_number,
                )
                frame_model, seconds, frame_target_classes = fit_production_estimator(
                    estimator_x,
                    estimator_y,
                    target_classes=target_classes,
                )
                if target_classes and frame_target_classes != target_classes:
                    raise ValueError("target classes changed across production frames")
                target_classes = frame_target_classes or target_classes
                training_models.append(frame_model)
                final_model_seconds += seconds
                train_rows += len(estimator_x)
                final_estimator_rows += len(estimator_x)
                del frame, x, y, estimator_x, estimator_y, frame_model
        gc.collect()

    if temporal_estimator_mode == "pooled":
        pooled_train_x = pd.concat(pooled_training_features, ignore_index=True)
        pooled_train_y = pd.concat(pooled_training_labels, ignore_index=True)
        pooled_train_x, pooled_train_y = sample_training_rows(
            pooled_train_x,
            pooled_train_y,
            limit=per_frame_limit,
            task=resolved_task,
            random_state=random_state,
        )
        validation_model, seconds, pooled_target_classes = fit_production_estimator(
            pooled_train_x,
            pooled_train_y,
            target_classes=target_classes,
        )
        target_classes = pooled_target_classes or target_classes
        final_model_seconds += seconds
        train_rows = len(pooled_train_x)
    else:
        validation_model = FrameEnsemble(
            training_models,
            task=resolved_task,
            target_classes=target_classes,
        )

    for frame_number, descriptor in enumerate(validation_descriptors, start=1):
        with _table_workspace(
            connection,
            production_tables,
            sample_rows=sample_rows,
            random_state=random_state,
            search=False,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
        ) as workspace:
            frame, _ = materialize_production_frame(
                workspace,
                descriptor=descriptor,
                split_value="validation",
                allow_plan_learning=False,
            )
            missing_features = set(feature_columns) - set(frame.columns)
            if missing_features:
                raise ValueError(
                    "Frozen GraphReduce plan did not reproduce validation "
                    f"features: {sorted(missing_features)[:10]}"
                )
            x = prepare_prediction_features(
                frame,
                columns=list(feature_columns),
                categorical_columns=set(categorical_columns),
                datetime_columns=set(datetime_columns),
            )
            y = frame[target_column]
            validation_truth.append(y.to_numpy(copy=True))
            validation_predictions.append(
                _predict_values(validation_model, x, task=resolved_task)
            )
            validation_rows += len(frame)
            if temporal_estimator_mode == "pooled":
                estimator_x, estimator_y = _sample_temporal_frame(
                    x,
                    y,
                    total_limit=per_frame_limit,
                    frame_count=validation_frame_count,
                    frame_index=frame_number - 1,
                    task=resolved_task,
                    random_state=random_state + production_training_frames,
                )
                pooled_validation_features.append(estimator_x.reset_index(drop=True))
                pooled_validation_labels.append(estimator_y.reset_index(drop=True))
                del frame, x, y, estimator_x, estimator_y
            else:
                estimator_x, estimator_y = sample_training_rows(
                    x,
                    y,
                    limit=per_frame_limit,
                    task=resolved_task,
                    random_state=(
                        random_state + production_training_frames + frame_number
                    ),
                )
                if (
                    _is_classification(resolved_task)
                    and estimator_y.nunique(dropna=True) < 2
                ):
                    # Autocomplete label days can contain a single observed
                    # category. They remain valid validation observations above,
                    # but cannot instantiate an independent CatBoost frame model.
                    logger.info(
                        "frame_ensemble_refit_skipped",
                        reason="single_class_frame",
                        split="validation",
                        rows=len(estimator_y),
                    )
                    del frame, x, y, estimator_x, estimator_y
                    continue
                frame_model, seconds, frame_target_classes = fit_production_estimator(
                    estimator_x,
                    estimator_y,
                    target_classes=target_classes,
                )
                if target_classes and frame_target_classes != target_classes:
                    raise ValueError("target classes changed across production frames")
                target_classes = frame_target_classes or target_classes
                validation_frame_models.append(frame_model)
                final_model_seconds += seconds
                final_estimator_rows += len(estimator_x)
                del frame, x, y, estimator_x, estimator_y, frame_model
        gc.collect()

    if temporal_estimator_mode == "pooled":
        final_train_x = pd.concat(
            [pooled_train_x, *pooled_validation_features],
            ignore_index=True,
        )
        final_train_y = pd.concat(
            [pooled_train_y, *pooled_validation_labels],
            ignore_index=True,
        )
        final_train_x, final_train_y = sample_training_rows(
            final_train_x,
            final_train_y,
            limit=per_frame_limit,
            task=resolved_task,
            random_state=random_state,
        )
        final_model, seconds, final_target_classes = fit_production_estimator(
            final_train_x,
            final_train_y,
            target_classes=target_classes,
        )
        if target_classes and final_target_classes != target_classes:
            raise ValueError("target classes changed during pooled production refit")
        target_classes = final_target_classes or target_classes
        final_model_seconds += seconds
        final_estimator_rows = len(final_train_x)
        del final_train_x, final_train_y
    else:
        final_model = FrameEnsemble(
            [*training_models, *validation_frame_models],
            task=resolved_task,
            target_classes=target_classes,
        )

    full_metric, full_validation_score = _score_prediction_chunks(
        validation_truth,
        validation_predictions,
        task=resolved_task,
        target_classes=target_classes,
    )
    if (
        temporal_estimator_mode == "pooled"
        and preselected_config is None
        and not rerank_trials
        and search_full_data
        and model_backend == resolved_final_model_backend
        and len(full_train_cutoffs) == search_training_frame_count
        and full_validation_frames is None
    ):
        selection_replay_delta = abs(
            full_validation_score - recommended.validation_score
        )
        logger.info(
            "selection_replay_verified",
            search_score=round(recommended.validation_score, 12),
            replay_score=round(full_validation_score, 12),
            absolute_delta=selection_replay_delta,
            training_frames=len(full_train_cutoffs),
        )
        if selection_replay_delta > 1e-9:
            raise RuntimeError(
                "Selected graph did not reproduce its full-data validation "
                "score under the pooled temporal estimator: "
                f"search={recommended.validation_score:.12f}, "
                f"replay={full_validation_score:.12f}, "
                f"delta={selection_replay_delta:.12f}"
            )
    production_plan = _plan_with_schema(
        selected_plan,
        columns=feature_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
    )
    plan_fingerprint = _execution_plan_fingerprint(production_plan)

    test_output_chunks: list[pd.DataFrame] = []
    test_truth: list[np.ndarray] = []
    test_estimates: list[np.ndarray] = []
    prediction_tables = production_tables
    if prediction_target_history is not None:
        prediction_tables = dict(production_tables)
        prediction_tables[TARGET_HISTORY_TABLE] = replace(
            production_tables[TARGET_HISTORY_TABLE],
            source=prediction_target_history,
            search_source=prediction_target_history,
        )
    if graph_labels is not None:
        test_descriptors: Iterable[Any] = graph_labels.test_cutoffs
    else:
        test_descriptors = tuple(_training_groups(external_test_labels, labels_spec))
    root = production_tables[root_name]
    root_prefix = root.prefix or f"{_slug(root_name, 'n0')[:10]}0"
    for descriptor in test_descriptors:
        with _table_workspace(
            connection,
            prediction_tables,
            sample_rows=sample_rows,
            random_state=random_state,
            search=False,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
        ) as workspace:
            if graph_labels is not None:
                test_frame = _materialize_graph_labels_at_cutoff(
                    workspace,
                    prediction_tables,
                    production_relationships,
                    root_name,
                    recommended.config,
                    graph_labels,
                    cut_date=pd.Timestamp(descriptor).to_pydatetime(),
                    split_value="test",
                    split_marker=split_marker,
                    cutoff_marker=cutoff_marker,
                    compute_period_days=compute_period_days,
                    verbose=verbose,
                    execution_plan=production_plan,
                    train=False,
                    infer_ts_periods=False,
                )
            else:
                cutoff, test_labels = descriptor
                test_frame = _materialize_at_cutoff(
                    workspace,
                    prediction_tables,
                    production_relationships,
                    root_name,
                    recommended.config,
                    test_labels,
                    labels_spec,
                    cut_date=cutoff,
                    compute_period_days=compute_period_days,
                    verbose=verbose,
                    execution_plan=production_plan,
                    train=False,
                    infer_ts_periods=False,
                )
            test_frame = _drop_graph_markers(test_frame, split_marker=split_marker)
            test_x = prepare_prediction_features(
                test_frame,
                columns=list(feature_columns),
                categorical_columns=set(categorical_columns),
                datetime_columns=set(datetime_columns),
            )
            estimates = _predict_values(final_model, test_x, task=resolved_task)
            output = pd.DataFrame(index=test_frame.index)
            for key in _key_parts(root.key):
                source_key = key if key in test_frame else f"{root_prefix}_{key}"
                if source_key in test_frame:
                    output[key] = test_frame[source_key]
            for column in (
                cutoff_marker,
                *(
                    (labels_spec.timestamp,)
                    if labels_spec and labels_spec.timestamp
                    else ()
                ),
            ):
                if column in test_frame:
                    output[column] = test_frame[column]
            output["prediction"] = estimates
            test_output_chunks.append(output.reset_index(drop=True))
            if target_column in test_frame and test_frame[target_column].notna().all():
                test_truth.append(test_frame[target_column].to_numpy(copy=True))
                test_estimates.append(estimates)
            del test_frame, test_x, output
        gc.collect()

    test_predictions = (
        pd.concat(test_output_chunks, ignore_index=True) if test_output_chunks else None
    )
    test_score = None
    if test_truth:
        _test_metric, test_score = _score_prediction_chunks(
            test_truth,
            test_estimates,
            task=resolved_task,
            target_classes=target_classes,
        )

    fitted_model = FittedModel(
        config=recommended.config,
        execution_plan=production_plan,
        plan_fingerprint=plan_fingerprint,
        estimator=final_model,
        validation_estimator=validation_model,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        target=target_column,
        task=resolved_task,
        metric=full_metric,
        validation_score=full_validation_score,
        train_rows=train_rows,
        validation_rows=validation_rows,
        training_frames=production_training_frames,
        model_backend=resolved_final_model_backend,
        target_classes=target_classes,
        target_history=prediction_target_history,
        target_history_key=(labels_spec.key if labels_spec is not None else None),
        peer_context=(labels_spec.peer_context if labels_spec is not None else None),
        test_predictions=test_predictions,
        test_score=test_score,
    )
    if preselected_config is not None:
        best = recommended = Trial(
            config=recommended.config,
            metric=full_metric,
            validation_score=full_validation_score,
            objective_score=_objective_score(full_metric, full_validation_score),
            feature_count=len(feature_columns),
            train_rows=train_rows,
            validation_rows=validation_rows,
            feature_seconds=0.0,
            model_seconds=final_model_seconds,
            model=None,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            execution_plan=production_plan,
        )
        trials = [best]
    logger.info(
        "temporal_refit_completed",
        metric=full_metric,
        validation_score=round(full_validation_score, 6),
        model_backend=resolved_final_model_backend,
        training_frames=production_training_frames,
        train_rows=source_train_rows,
        estimator_train_rows=train_rows,
        validation_rows=validation_rows,
        features=len(feature_columns),
        plan_operations=len(production_plan.get("records", [])),
        plan_fingerprint=plan_fingerprint,
        final_model_seconds=round(final_model_seconds, 3),
        final_estimator_rows=final_estimator_rows,
        temporal_estimator_mode=temporal_estimator_mode,
        estimator_models=(
            len(final_model.models)
            if temporal_estimator_mode == "frame_ensemble"
            else 1
        ),
        source_frames=total_fit_frames,
        test_rows=0 if test_predictions is None else len(test_predictions),
        test_score=test_score,
    )
    return FitResult(
        task=resolved_task,
        metric=best.metric,
        best_trial=best,
        recommended_trial=recommended,
        trials=tuple(trials),
        confirmation_trials=tuple(confirmation_trials),
        rerank_trials=tuple(rerank_trials),
        fitted_model=fitted_model,
        feature_audit=feature_audit,
    )


def predict(
    fitted: FitResult,
    parent_node: Table | Source,
    prediction_node: Labels | Source,
    *,
    tables: Sequence[Table | Source] | Mapping[str, Table | Source] = (),
    relationships: Sequence[Relationship | Mapping[str, Any]] = (),
    parent_key: Key | Sequence[str] | None = None,
    label_key: Key | Sequence[str] | None = None,
    parent_date: str | None = None,
    parent_timeless: bool = False,
    label_timestamp: str | None = None,
    compute_period_days: int = 3650,
    connection: duckdb.DuckDBPyConnection | None = None,
    duckdb_memory_limit: str | None = "128GB",
    duckdb_max_temp_directory_size: str | None = "128GB",
    verbose: bool = False,
    use_validation_model: bool = False,
) -> pd.DataFrame:
    """Replay a fitted production plan for new point-in-time entity rows.

    No GraphReduce feature discovery runs here: the production execution plan
    and ordered model schema captured by :func:`fit` are mandatory.
    """

    artifact = fitted.fitted_model
    if artifact is None:
        raise ValueError("The FitResult does not contain a production fitted model")
    if duckdb_memory_limit is not None and (
        not isinstance(duckdb_memory_limit, str) or not duckdb_memory_limit.strip()
    ):
        raise ValueError("duckdb_memory_limit must be a non-empty string or None")
    if duckdb_max_temp_directory_size is not None and (
        not isinstance(duckdb_max_temp_directory_size, str)
        or not duckdb_max_temp_directory_size.strip()
    ):
        raise ValueError(
            "duckdb_max_temp_directory_size must be a non-empty string or None"
        )
    parent = coerce_table(
        parent_node,
        key=parent_key,
        date=parent_date,
        timeless=parent_timeless,
    )
    labels_spec = coerce_labels(
        prediction_node,
        target=artifact.target,
        key=label_key,
        timestamp=label_timestamp,
        split=None,
    )
    if artifact.peer_context is not None:
        if (
            labels_spec.peer_context is not None
            and labels_spec.peer_context is not artifact.peer_context
        ):
            raise ValueError("Prediction peer context conflicts with fitted context")
        labels_spec = replace(labels_spec, peer_context=artifact.peer_context)
    if parent.key is None or labels_spec.key is None:
        raise ValueError("parent_key and label_key are required for prediction")
    if labels_spec.timestamp and parent.date is None and not parent.timeless:
        raise ValueError(
            "parent_date is required for temporal prediction unless the root "
            "entity is explicitly timeless"
        )
    root_name, normalized_tables = _normalize_tables(parent, tables)
    normalized_relationships = tuple(
        coerce_relationship(item) for item in relationships
    )
    if artifact.target_history is not None:
        if TARGET_HISTORY_TABLE in normalized_tables:
            raise ValueError(
                f"{TARGET_HISTORY_TABLE!r} is supplied by the fitted model"
            )
        normalized_tables[TARGET_HISTORY_TABLE] = Table(
            artifact.target_history,
            name=TARGET_HISTORY_TABLE,
            date=TARGET_HISTORY_TIME,
            prefix="targethist",
            columns=(
                *_key_parts(artifact.target_history_key),
                TARGET_HISTORY_TIME,
                TARGET_HISTORY_VALUE,
            ),
            sample_in_search=False,
            cutoff_boundary="strict",
            auto_base_predicate_max=0,
        )
        normalized_relationships = (
            *normalized_relationships,
            Relationship(
                parent=root_name,
                child=TARGET_HISTORY_TABLE,
                parent_key=normalized_tables[root_name].key,
                child_key=artifact.target_history_key,
            ),
        )

    rows = _load_frame_source(
        labels_spec.source,
        connection=connection,
        sample_rows=1,
        random_state=0,
        sample=False,
        name="prediction_rows",
        duckdb_memory_limit=duckdb_memory_limit,
        duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
    )
    required = {
        *_key_parts(labels_spec.key),
        *([labels_spec.timestamp] if labels_spec.timestamp else []),
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Prediction source is missing columns: {sorted(missing)}")
    order_column = "__kurversc_prediction_order__"
    while order_column in rows:
        order_column += "_"
    rows = rows.copy()
    rows[order_column] = range(len(rows))
    estimator = (
        artifact.validation_estimator if use_validation_model else artifact.estimator
    )
    probability_columns = tuple(
        f"prediction_{index}" for index in range(len(artifact.target_classes))
    )
    predictions: list[pd.DataFrame] = []
    for cutoff, prediction_rows in _training_groups(rows, labels_spec):
        with _table_workspace(
            connection,
            normalized_tables,
            sample_rows=1,
            random_state=0,
            search=False,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
        ) as workspace:
            frame = _materialize_at_cutoff(
                workspace,
                normalized_tables,
                normalized_relationships,
                root_name,
                artifact.config,
                prediction_rows,
                labels_spec,
                cut_date=cutoff,
                execution_plan=artifact.execution_plan,
                compute_period_days=compute_period_days,
                verbose=verbose,
                train=False,
                infer_ts_periods=False,
            )
            inputs = prepare_prediction_features(
                frame,
                columns=list(artifact.feature_columns),
                categorical_columns=set(artifact.categorical_columns),
                datetime_columns=set(artifact.datetime_columns),
            )
            values = _predict_values(estimator, inputs, task=artifact.task)
            result_frame = pd.DataFrame(
                {order_column: frame[order_column].to_numpy(copy=True)}
            )
            if artifact.task == "multiclass_classification":
                if values.ndim != 2 or values.shape[1] != len(probability_columns):
                    raise ValueError(
                        "Multiclass prediction shape does not match classes"
                    )
                for index, column in enumerate(probability_columns):
                    result_frame[column] = values[:, index]
            else:
                result_frame["prediction"] = values
            predictions.append(result_frame)
            del frame, inputs
        gc.collect()
    prediction_frame = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=[order_column, *probability_columns, "prediction"])
    )
    if (
        len(prediction_frame) != len(rows)
        or prediction_frame[order_column].duplicated().any()
        or set(prediction_frame[order_column]) != set(rows[order_column])
    ):
        raise ValueError(
            "Frozen graph prediction did not return exactly one row for "
            "every requested entity/timestamp"
        )
    ordered = rows.sort_values(order_column, kind="stable")
    output = ordered.drop(columns=order_column).copy()
    prediction_by_order = prediction_frame.set_index(order_column)
    if artifact.task == "multiclass_classification":
        output.loc[:, probability_columns] = prediction_by_order.reindex(
            ordered[order_column]
        )[list(probability_columns)].to_numpy()
    else:
        output["prediction"] = prediction_by_order.reindex(ordered[order_column])[
            "prediction"
        ].to_numpy()
    return output.reset_index(drop=True)
