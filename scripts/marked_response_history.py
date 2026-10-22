"""Bounded, schema-discovered response timing and context research features.

This is not registered in GraphReduce or KurveRSC. Ordinary observed binary
responses are not predictive-task labels. Routing keys are never predictors.
Both event and ancestor clocks obey min(query time, frozen knowledge time).
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.activity_timeline_challenger import literal, quote, table_metadata


class ResponsePathUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class ResponsePath:
    event: str
    event_time: str
    root_key: str
    response: str
    context: str | None
    ancestor: str | None = None
    event_ancestor_key: str | None = None
    ancestor_key: str | None = None
    ancestor_time: str | None = None


def routed_sql(database, path):
    event = f"read_parquet({literal(Path(database) / (path.event + '.parquet'))}) e"
    ancestor = ""
    root = f"e.{quote(path.root_key)}"
    times = [f"CAST(e.{quote(path.event_time)} AS TIMESTAMP)"]
    if path.ancestor:
        ancestor = (f" JOIN read_parquet({literal(Path(database) / (path.ancestor + '.parquet'))}) p"
                    f" ON e.{quote(path.event_ancestor_key)}=p.{quote(path.ancestor_key)}")
        root = f"p.{quote(path.root_key)}"
        if path.ancestor_time:
            times.append(f"CAST(p.{quote(path.ancestor_time)} AS TIMESTAMP)")
    return event + ancestor, root, times


def discover_response_path(database, root, cutoff, connection):
    """Profile only training-visible payloads along declared <=2-hop paths."""
    database = Path(database)
    tables = {p.stem: (p, table_metadata(p)) for p in sorted(database.glob("*.parquet"))}
    candidates = []
    for event, (file, meta) in tables.items():
        foreign = meta["fkey_col_to_pkey_table"] or {}
        if event == root or not meta["time_col"] or len(foreign) < 2:
            continue
        structural = {meta["pkey_col"], meta["time_col"], *foreign}
        numeric = [field.name for field in pq.ParquetFile(file).schema_arrow
                   if field.name not in structural and (pa.types.is_integer(field.type)
                   or pa.types.is_floating(field.type) or pa.types.is_boolean(field.type))]
        if not numeric:
            continue
        routes = []
        for key, destination in sorted(foreign.items()):
            if destination == root:
                routes.append(ResponsePath(event, meta["time_col"], key, "", None))
            elif destination in tables and destination != event:
                parent = tables[destination][1]
                if not parent["pkey_col"]:
                    continue
                for parent_root_key, parent_root in sorted((parent["fkey_col_to_pkey_table"] or {}).items()):
                    if parent_root == root:
                        routes.append(ResponsePath(event, meta["time_col"], parent_root_key, "", None,
                            destination, key, parent["pkey_col"], parent["time_col"]))
        for route in routes:
            if route.ancestor:
                parent_file = database / (route.ancestor + ".parquet")
                bad = connection.execute(f"SELECT COUNT(*)-COUNT(DISTINCT {quote(route.ancestor_key)})"
                    f" FROM read_parquet({literal(parent_file)}) WHERE {quote(route.ancestor_key)} IS NOT NULL").fetchone()[0]
                if bad:
                    raise ValueError("Declared ancestor key is not unique; join fanout forbidden")
            source, root_expr, times = routed_sql(database, route)
            visible = " AND ".join(f"{clock}<TIMESTAMP {literal(pd.Timestamp(cutoff))}" for clock in times)
            names = ",".join(f"e.{quote(name)}" for name in numeric)
            sample = connection.execute(f"SELECT {names} FROM {source} WHERE {visible}"
                f" AND {root_expr} IS NOT NULL LIMIT 20000").df()
            responses, contexts = [], []
            for name in numeric:
                values = pd.to_numeric(sample[name], errors="coerce").dropna()
                unique = values.unique()
                if len(values) < 2:
                    continue
                if len(unique) == 2 and set(unique) == {0, 1}:
                    responses.append((float(values.mean()), name))
                elif 2 <= len(unique) <= 32 and np.equal(values, np.floor(values)).all():
                    contexts.append((len(unique), name))
            context = sorted(contexts, key=lambda item: (-item[0], item[1]))[0][1] if contexts else None
            for prevalence, name in responses:
                candidate = ResponsePath(**{**asdict(route), "response": name, "context": context})
                # A rare ordinary positive mark is preferred, without task labels.
                candidates.append((prevalence, event, name, route.ancestor or "", candidate))
    if not candidates:
        raise ResponsePathUnavailable("No training-visible binary response on a declared timed association")
    return sorted(candidates, key=lambda row: row[:4])[0][-1]


class MarkedResponseHistory:
    """At most 51 scalar values from one marked temporal association."""

    def __init__(self, database, root, *, training_cutoff, horizon, temporary,
                 threads=2, memory_limit="8GB"):
        self.database = Path(database)
        self.first_query, self.horizon = pd.Timestamp(training_cutoff), pd.Timedelta(horizon)
        if pd.isna(self.first_query) or self.first_query.tz is not None or self.horizon <= pd.Timedelta(0):
            raise ValueError("Naive training timestamp and positive horizon required")
        Path(temporary).mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(config=dict(memory_limit=memory_limit, threads=threads,
            temp_directory=str(temporary), max_temp_directory_size="40GB"))
        try:
            self.path = discover_response_path(database, root, self.first_query, self.con)
        except BaseException:
            self.con.close()
            raise
        self.audit = dict(strategy="marked_response_history", path=asdict(self.path),
            maximum_paths=1, maximum_features=51, profiling_rows=20000,
            discovery_cutoff=str(self.first_query), source_task_labels_used=False,
            identifiers_emitted=False, exact_subday_timestamps=True, history_days=730,
            windows_horizons=[1/3, 1, 4], context_shrinkage=5.)
        self.support = None

    def transform(self, requests, *, phase_boundary):
        times = pd.to_datetime(requests["__event_cutoff"])
        boundary = pd.Timestamp(phase_boundary)
        if (requests.empty or times.isna().any() or requests["__event_entity"].isna().any()
                or times.min() < self.first_query or boundary < self.first_query):
            raise ValueError("Authorized complete nonempty requests required")
        source, root_expr, clocks = routed_sql(self.database, self.path)
        chunks, timing_support, context_support, phases = [], np.zeros(len(requests), bool), np.zeros(len(requests), bool), []
        seconds = self.horizon.total_seconds()
        for cutoff in sorted(times.unique()):
            cutoff = pd.Timestamp(cutoff)
            positions = np.flatnonzero(times.eq(cutoff).to_numpy())
            ids = requests.iloc[positions]["__event_entity"].astype(str).to_numpy()
            self.con.register("requested_roots", pd.DataFrame(dict(entity=np.unique(ids))))
            stamp = f"TIMESTAMP {literal(cutoff)}"
            ceiling = min(cutoff, boundary)
            visible = " AND ".join(f"{clock}<TIMESTAMP {literal(ceiling)}" for clock in clocks)
            context = f"CAST(e.{quote(self.path.context)} AS VARCHAR)" if self.path.context else "'__none__'"
            self.con.execute(f"""CREATE OR REPLACE TEMP TABLE marked_events AS
                SELECT CAST({root_expr} AS VARCHAR) entity,{clocks[0]} event_time,
                    CAST(e.{quote(self.path.response)} AS DOUBLE) response,
                    COALESCE({context},'__missing__') context
                FROM {source} JOIN requested_roots r ON CAST({root_expr} AS VARCHAR)=r.entity
                WHERE {visible} AND {clocks[0]}>={stamp}-INTERVAL 730 DAY
                  AND e.{quote(self.path.response)} IS NOT NULL""")
            if self.con.execute("SELECT COUNT(*) FROM marked_events WHERE response NOT IN (0,1)").fetchone()[0]:
                raise ValueError("Discovered binary response changed domain")
            summary = self.con.execute(f"""SELECT entity,COUNT(*) mr_exposures,
                SUM(response) mr_positive_events,
                COUNT(DISTINCT event_time) FILTER(WHERE response=1) mr_positive_arrivals,
                EPOCH({stamp}-MAX(event_time) FILTER(WHERE response=1))/{seconds} mr_positive_recency,
                EPOCH({stamp}-MAX(event_time) FILTER(WHERE response=0))/{seconds} mr_negative_recency
                FROM marked_events GROUP BY entity""").df().set_index("entity").reindex(ids)
            gaps = self.con.execute(f"""WITH arrivals AS (
                SELECT DISTINCT entity,event_time FROM marked_events WHERE response=1
            ), gaps AS (
                SELECT *,EPOCH(event_time-LAG(event_time) OVER(PARTITION BY entity ORDER BY event_time)) gap
                FROM arrivals
            ), aged AS (
                SELECT *,EPOCH({stamp}-MAX(event_time) OVER(PARTITION BY entity)) age FROM gaps
            ) SELECT entity,COUNT(gap) mr_gaps,AVG(gap)/{seconds} mr_mean_gap,
                MEDIAN(gap)/{seconds} mr_median_gap,STDDEV_POP(gap)/NULLIF(AVG(gap),0) mr_gap_cv,
                SUM((gap>age)::INTEGER) surviving,SUM((gap>age+{seconds})::INTEGER) future_surviving
                FROM aged GROUP BY entity""").df().set_index("entity").reindex(ids)
            for name in ("mr_gaps", "mr_mean_gap", "mr_median_gap", "mr_gap_cv"):
                summary[name] = gaps[name].to_numpy()
            n = gaps.mr_gaps.fillna(0).to_numpy()
            summary["mr_conditional_tail"] = np.where(n>=3,
                (gaps.future_surviving.fillna(0)+1)/(gaps.surviving.fillna(0)+1),np.nan)
            summary["mr_current_tail_fraction"] = np.where(n>=3,gaps.surviving/np.maximum(n,1),np.nan)
            summary["mr_age_over_mean_gap"] = np.where(n>=3,summary.mr_positive_recency/summary.mr_mean_gap,np.nan)
            timing_support[positions] = summary.mr_exposures.fillna(0).to_numpy() >= 3
            for number, scale in enumerate((1/3, 1, 4)):
                condition = f"event_time>={stamp}-INTERVAL {literal(str(seconds*scale)+' seconds')}"
                block = self.con.execute(f"""SELECT entity,COUNT(*) mr_w{number}_exposures,
                    SUM(response) mr_w{number}_positives,AVG(response) mr_w{number}_rate,
                    COUNT(DISTINCT event_time) FILTER(WHERE response=1) mr_w{number}_positive_arrivals,
                    COUNT(DISTINCT DATE_TRUNC('hour',event_time)) FILTER(WHERE response=1) mr_w{number}_positive_hours
                    FROM marked_events WHERE {condition} GROUP BY entity""").df().set_index("entity").reindex(ids)
                summary = pd.concat([summary.reset_index(drop=True),block.reset_index(drop=True)],axis=1)
            own = self.con.execute("SELECT entity,AVG(response) event_rate FROM marked_events GROUP BY entity").df()
            self.con.register("marked_priors",own)
            for number, scale in enumerate((None,1/3,1,4)):
                condition = "TRUE" if scale is None else f"event_time>={stamp}-INTERVAL {literal(str(seconds*scale)+' seconds')}"
                block = self.con.execute(f"""WITH strata AS (
                    SELECT entity,context,COUNT(*) n,SUM(response) clicks FROM marked_events
                    WHERE {condition} GROUP BY entity,context
                ), weighted AS (
                    SELECT *,n::DOUBLE/SUM(n) OVER(PARTITION BY entity) AS context_fraction,
                        (clicks+5.*event_rate)/(n+5.) AS posterior_response FROM strata JOIN marked_priors USING(entity)
                ) SELECT entity,COUNT(*) mc_w{number}_contexts,
                    SUM((clicks>0)::INTEGER) mc_w{number}_positive_contexts,
                    MAX(context_fraction) mc_w{number}_largest_share,-SUM(context_fraction*LN(context_fraction)) mc_w{number}_entropy,
                    AVG(posterior_response) mc_w{number}_rate_mean,STDDEV_POP(posterior_response) mc_w{number}_rate_std
                    FROM weighted GROUP BY entity""").df().set_index("entity").reindex(ids)
                if number == 0:
                    context_support[positions] = timing_support[positions] & (block.mc_w0_contexts.fillna(0).to_numpy()>=2) & (self.path.context is not None)
                summary = pd.concat([summary.reset_index(drop=True),block.reset_index(drop=True)],axis=1)
            for name in summary:
                if name.endswith(("exposures","events","arrivals","hours","positives","contexts")) or name=="mr_gaps":
                    summary[name]=summary[name].fillna(0)
            if len(summary.columns)>51:
                raise ValueError("Response feature budget exceeded")
            summary["__position"]=positions
            chunks.append(summary)
            phases.append(dict(query=str(cutoff),knowledge_ceiling=str(ceiling),
                source_rows=int(self.con.execute("SELECT COUNT(*) FROM marked_events").fetchone()[0])))
        self.support = dict(timing=timing_support,context=context_support,combined=timing_support)
        self.last_audit = dict(**self.audit,phase_boundary=str(boundary),phases=phases,
            supported_rows={k:int(v.sum()) for k,v in self.support.items()})
        return pd.concat(chunks).sort_values("__position").drop(columns="__position").reset_index(drop=True).replace([np.inf,-np.inf],np.nan).astype("float32")

    def close(self):
        self.con.close()
