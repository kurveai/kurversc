"""Research-only annotation learned from ordinary, already observed events.

Fit one bounded response model before the earliest consumed training anchor.
It predicts an ordinary binary source payload, NOT a predictive-task target.
The frozen model is applied to past rows and compressed at the root grain.
This tests joint event/object/root context, rather than separate scalar means.
"""

from dataclasses import asdict
from pathlib import Path

from catboost import CatBoostClassifier
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from kurversc.signal_challengers import ChallengerUnavailable
from scripts.activity_timeline_challenger import literal, quote
from scripts.enum_event_history_20261020 import source_metadata
from scripts.marked_response_history import discover_response_path, routed_sql


class SourceResponseAnnotation:
    maximum_columns = 30

    def __init__(self, database, root, *, training_cutoff, horizon, temporary,
                 maximum_training_rows=200_000, model_iterations=100):
        self.database, self.root = Path(database), root
        self.first, self.horizon = pd.Timestamp(training_cutoff), pd.Timedelta(horizon)
        if (pd.isna(self.first) or self.first.tz is not None or self.horizon <= pd.Timedelta(0)
                or not 20 <= maximum_training_rows <= 200_000 or not 1 <= model_iterations <= 100):
            raise ValueError("Naive training clock and bounded annotation budgets required")
        Path(temporary).mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(config=dict(memory_limit="4GB", threads=1,
            temp_directory=str(temporary), max_temp_directory_size="80GB"))
        self.tables = {p.stem: (p, source_metadata(p)) for p in sorted(self.database.glob("*.parquet"))}
        try:
            # The legacy discovery helper is safe only on ordinary clocks.
            if any(meta["available_at"] and meta["available_at"] != meta["time_col"]
                    for _, meta in self.tables.values()):
                raise ValueError("Explicit delayed source arrival needs availability-safe response discovery")
            self.path = discover_response_path(self.database, root, self.first, self.con)
            self.source, self.root_expression, self.clocks = routed_sql(self.database, self.path)
            root_meta = self.tables[root][1]
            if not isinstance(root_meta["pkey_col"], str):
                raise ValueError("A scalar declared root key is required")
            self.aliases = [("e", self.path.event, 6), ("a", root, 4)]
            self.source += (f" JOIN read_parquet({literal(self.tables[root][0])}) a ON "
                f"{self.root_expression}=a.{quote(root_meta['pkey_col'])}")
            if root_meta["time_col"]:
                self.clocks.append(f"CAST(a.{quote(root_meta['time_col'])} AS TIMESTAMP)")
            if self.path.ancestor:
                self.aliases.append(("p", self.path.ancestor, 6))
            # Dimensions may add payload, never fan out a source row.
            dimensions = []
            for alias, table, _ in list(self.aliases):
                if alias == "a":
                    continue
                for fk, target in sorted((self.tables[table][1]["fkey_col_to_pkey_table"] or {}).items()):
                    if target not in {root, self.path.event, self.path.ancestor} and target in self.tables:
                        dimensions.append((alias, fk, target))
            for i, (alias, fk, target) in enumerate(dimensions[:2]):
                file, meta = self.tables[target]
                key = meta["pkey_col"]
                if not isinstance(key, str):
                    continue
                invalid = self.con.execute(f"SELECT COUNT(*)-COUNT(DISTINCT {quote(key)}) "
                    f"FROM read_parquet({literal(file)}) WHERE {quote(key)} IS NOT NULL").fetchone()[0]
                if invalid:
                    raise ValueError("Declared dimension key is non-unique; fanout forbidden")
                d = f"d{i}"
                visible = (f" AND CAST({d}.{quote(meta['time_col'])} AS TIMESTAMP)<{{ceiling}}"
                    if meta["time_col"] else "")
                self.source += (f" LEFT JOIN read_parquet({literal(file)}) {d} ON "
                    f"{alias}.{quote(fk)}={d}.{quote(key)}"+visible)
                self.aliases.append((d, target, 6))
            # Root identity is routing only, so a duplicate must not multiply events.
            invalid = self.con.execute(f"SELECT COUNT(*)-COUNT(DISTINCT {quote(root_meta['pkey_col'])}) "
                f"FROM read_parquet({literal(self.tables[root][0])}) WHERE {quote(root_meta['pkey_col'])} IS NOT NULL").fetchone()[0]
            if invalid:
                raise ValueError("Declared root key is non-unique")
            candidates = []
            for alias, table, quota in self.aliases:
                file, meta = self.tables[table]
                excluded = {meta["pkey_col"], meta["time_col"], meta["available_at"],
                    *(meta["fkey_col_to_pkey_table"] or {})}
                if alias == "e":
                    excluded.add(self.path.response)
                fields = [f for f in pq.ParquetFile(file).schema_arrow if f.name not in excluded
                    and (pa.types.is_integer(f.type) or pa.types.is_floating(f.type)
                         or pa.types.is_boolean(f.type) or pa.types.is_string(f.type)
                         or pa.types.is_large_string(f.type))]
                candidates += [(alias, f.name, quota) for f in fields]
            if not candidates:
                raise ValueError("No nonstructural event-context payload")
            expressions = [f"{alias}.{quote(name)} AS f{i}" for i, (alias, name, _) in enumerate(candidates)]
            sampled = self._rows(expressions, self.first, limit=20_000)
            self.fields, self.categorical = [], []
            taken = {}
            for i, (alias, name, quota) in enumerate(candidates):
                values = sampled[f"f{i}"].dropna()
                cardinality = values.nunique()
                if len(values) < 2 or cardinality < 2 or taken.get(alias, 0) >= quota:
                    continue
                numeric = pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
                integral = numeric and np.equal(pd.to_numeric(values), np.floor(pd.to_numeric(values))).all()
                if not numeric and (cardinality > 4096 or values.astype(str).str.len().mean() > 120):
                    continue
                if integral and cardinality > 256 and cardinality/max(1, len(values)) > .5:
                    continue  # near-unique undeclared integer identities
                column = f"f{len(self.fields)}"
                self.fields.append(dict(expression=f"{alias}.{quote(name)}", column=column,
                    source=next(t for a, t, _ in self.aliases if a == alias), name=name,
                    numeric_category=bool(integral)))
                if not numeric or integral:
                    self.categorical.append(column)
                taken[alias] = taken.get(alias, 0)+1
                if len(self.fields) == 24:
                    break
            if not self.fields:
                raise ValueError("No reusable event context after bounded profiling")
            columns = [f"{f['expression']} AS {f['column']}" for f in self.fields]
            rows = self._rows(columns, self.first, limit=maximum_training_rows)
            if len(rows) < 20 or rows["mark"].nunique() != 2:
                raise ValueError("Insufficient past ordinary response supervision")
            self.model = CatBoostClassifier(iterations=model_iterations, depth=5, learning_rate=.08,
                l2_leaf_reg=10., loss_function="Logloss", random_seed=0, thread_count=1,
                verbose=False, allow_writing_files=False)
            self.model.fit(self.prepare(rows), rows["mark"], cat_features=self.categorical)
            self.audit = dict(strategy="frozen_ordinary_source_response_annotation", path=asdict(self.path),
                training_ceiling=str(self.first), training_rows=len(rows), training_rows_cap=maximum_training_rows,
                model_context_fields=[dict(source=f["source"], name=f["name"]) for f in self.fields],
                model_iterations=model_iterations, model_thread_count=1, maximum_context_columns=24,
                maximum_output_columns=30, task_labels_used=False, identifiers_emitted=False,
                source_response_column_excluded_from_model_inputs=True,
                model_fit_strictly_before_earliest_training_query=True,
                new_graph_configurations=0)
        except BaseException:
            self.con.close()
            raise

    def _rows(self, expressions, ceiling, *, limit=None, requested=False):
        stamp = f"TIMESTAMP {literal(ceiling)}"
        source = self.source.replace("{ceiling}", stamp)
        where = " AND ".join(f"{clock}<{stamp}" for clock in self.clocks)
        if requested:
            source += f" JOIN requested_annotation_roots q ON CAST({self.root_expression} AS VARCHAR)=q.entity"
        # Discovery uses a bounded training profile. A numeric payload may
        # later expand beyond {0,1}; never silently discard those source rows.
        invalid = self.con.execute(f"SELECT 1 FROM {source} WHERE {where} "
            f"AND e.{quote(self.path.response)} IS NOT NULL AND e.{quote(self.path.response)} NOT IN (0,1) LIMIT 1").fetchone()
        if invalid is not None:
            raise ChallengerUnavailable("Discovered ordinary response is not binary at the authorized source boundary")
        # Canonical source order is independent of the physical Parquet order.
        order = ",".join([self.root_expression, self.clocks[0], *[e.rsplit(" AS ", 1)[0] for e in expressions],
            f"e.{quote(self.path.response)}"])
        suffix = f" ORDER BY hash({order}),{order}"+(f" LIMIT {int(limit)}" if limit else "")
        return self.con.execute(f"SELECT CAST({self.root_expression} AS VARCHAR) entity,{self.clocks[0]} event_time,"
            f"CAST(e.{quote(self.path.response)} AS DOUBLE) mark,{','.join(expressions)} FROM {source} "
            f"WHERE {where} AND e.{quote(self.path.response)} IN (0,1)"+suffix).df()

    def prepare(self, frame):
        result = frame[[f["column"] for f in self.fields]].copy()
        numeric_categories = {f["column"] for f in self.fields if f["numeric_category"]}
        for c in result:
            if c in self.categorical:
                if c in numeric_categories:
                    result[c] = pd.to_numeric(result[c], errors="coerce").map(
                        lambda value: "__missing__" if pd.isna(value) else format(float(value), ".17g"))
                else:
                    result[c] = result[c].astype("string").fillna("__missing__").astype(str)
            else:
                result[c] = pd.to_numeric(result[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return result

    def transform(self, requests, *, phase_boundary):
        if (list(requests) != ["__event_entity", "__event_cutoff"] or requests.empty
                or requests.isna().any().any()):
            raise ValueError("Only complete entity/time routing requests accepted")
        times, boundary = pd.to_datetime(requests["__event_cutoff"]), pd.Timestamp(phase_boundary)
        if times.min() < self.first or boundary < self.first:
            raise ValueError("Frozen annotation must precede every query")
        blocks, self.support = [], np.zeros(len(requests), bool)
        expressions = [f"{f['expression']} AS {f['column']}" for f in self.fields]
        for query in sorted(times.unique()):
            query = pd.Timestamp(query)
            positions = np.flatnonzero(times.eq(query).to_numpy())
            ids = requests.iloc[positions]["__event_entity"].astype(str).to_numpy()
            self.con.register("requested_annotation_roots", pd.DataFrame(dict(entity=np.unique(ids))))
            rows = self._rows(expressions, min(query, boundary), requested=True)
            # Cap only the isolated research materializer, never silently sample.
            if len(rows) > 5_000_000:
                raise ValueError("Past annotation extraction exceeds research capacity")
            if len(rows):
                rows["expected"] = self.model.predict_proba(self.prepare(rows))[:, 1]
            else:
                rows["expected"] = pd.Series(dtype=float)
            rows["residual"] = rows["mark"]-rows["expected"]
            rows["log_failure"] = np.log1p(-np.clip(rows["expected"], 0, 1-1e-8))
            self.support[positions] = np.isin(ids, rows["entity"].unique())
            summaries = []
            for w, scale in enumerate((None, 1., 4.)):
                visible = rows if scale is None else rows.loc[rows.event_time.ge(query-scale*self.horizon)]
                grouped = visible.groupby("entity", sort=True)
                summary = grouped.agg(count=("mark", "size"), mark_rate=("mark", "mean"),
                    expected_mean=("expected", "mean"), expected_std=("expected", "std"),
                    expected_max=("expected", "max"), residual_mean=("residual", "mean"),
                    residual_sum=("residual", "sum"), log_failure=("log_failure", "sum"))
                summary["positive_expected"] = visible.loc[visible.mark.eq(1)].groupby("entity").expected.mean()
                summary["any_expected"] = -np.expm1(summary.log_failure)
                summary = summary.reindex(ids).reset_index(drop=True)
                summary.columns = [f"source_annotation_w{w}_{c}" for c in summary]
                summaries.append(summary)
            block = pd.concat(summaries, axis=1)
            block["__position"] = positions
            blocks.append(block)
        result = pd.concat(blocks).sort_values("__position").drop(columns="__position").reset_index(drop=True).astype("float32")
        if len(result.columns) > self.maximum_columns or np.isinf(result.to_numpy()).any():
            raise ValueError("Invalid bounded source annotation")
        return result

    def close(self):
        self.con.close()
