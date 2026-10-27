"""Object-history summaries for frozen multi-cutoff forecasting phases.

The query clock determines ages/windows. The independent knowledge clock
limits source visibility. Advancing the former must never advance the latter.
The original single-cutoff provider is left immutable for prior experiments.
"""

import numpy as np
import pandas as pd

from scripts.activity_timeline_challenger import literal, quote
from scripts.object_breadth_challenger import ObjectBreadth


class PhaseFrozenObjectBreadth(ObjectBreadth):
    def __init__(self, *args, **kwargs):
        requested_threads = kwargs.get("threads", 1)
        if isinstance(requested_threads, bool) or not isinstance(requested_threads, int) or requested_threads < 1:
            raise ValueError("Positive integer object-history query thread count required")
        # The pinned DuckDB engine can SIGSEGV in TupleDataCollection::Combine
        # on these nested, high-cardinality grouped aggregates. A Python
        # exception handler cannot recover from that native process crash.
        # Avoid the unsafe parallel-combine path on the FIRST execution,
        # without changing the SQL, source rows, windows or request cohorts.
        kwargs["threads"] = 1
        super().__init__(*args, **kwargs)
        self.query_recoveries = []
        self.query_execution = dict(requested_threads=requested_threads,
            effective_threads=1, strategy="serial_from_first_query",
            query_changed=False, source_sampling_changed=False)
        self.audit["query_execution"] = self.query_execution

    def _summary(self, query, requested):
        # Unknown engine/data errors remain fatal. Do not silently retry,
        # drop a feature family, or call incomplete evidence successful.
        return self.con.execute(query).df()

    def transform(self, requests, *, phase_boundary):
        times = pd.to_datetime(requests["__event_cutoff"])
        phase = pd.Timestamp(phase_boundary)
        if requests.empty or times.isna().any() or requests["__event_entity"].isna().any():
            raise ValueError("Complete nonempty routing/query keys required")
        if times.min() < self.first_query or phase < self.first_query:
            raise ValueError("Requests/knowledge cannot precede discovery")
        chunks = []
        for cutoff in sorted(times.unique()):
            positions = np.flatnonzero(times.eq(cutoff).to_numpy())
            ids = requests.iloc[positions]["__event_entity"].astype(str).to_numpy()
            requested = pd.DataFrame({"entity": np.unique(ids)})
            self.con.register("requested_roots", requested)
            stamp = f"TIMESTAMP {literal(pd.Timestamp(cutoff))}"
            visible_until = f"TIMESTAMP {literal(min(pd.Timestamp(cutoff), phase))}"
            blocks = []
            for number, path in enumerate(self.paths):
                object_fields, root_fields = [], []
                for w, factor in enumerate((1 / 3, 1, 4)):
                    seconds = self.days * 86400 * factor
                    recent = f"event_time>={stamp}-INTERVAL {literal(str(seconds) + ' seconds')}"
                    object_fields.append(f"COUNT(*) FILTER(WHERE {recent}) n{w}")
                    new = f"first_time>={stamp}-INTERVAL {literal(str(seconds) + ' seconds')}"
                    prefix = f"ob_path{number}_w{w}"
                    root_fields += [f"COUNT(*) FILTER(WHERE n{w}>0) {prefix}_objects",
                        f"SUM(n{w}) {prefix}_events", f"SUM((n{w}=1)::INTEGER) {prefix}_singletons",
                        f"COUNT(*) FILTER(WHERE {new}) {prefix}_new_objects",
                        f"SUM(n{w}) FILTER(WHERE {new}) {prefix}_new_events",
                        f"MAX(n{w})/NULLIF(SUM(n{w}),0)::DOUBLE {prefix}_maximum_share",
                        f"SUM(n{w}::DOUBLE*n{w})/NULLIF(SUM(n{w})::DOUBLE*SUM(n{w}),0) {prefix}_concentration"]
                query = f"""WITH e AS (
                    SELECT CAST({quote(path.entity)} AS VARCHAR) entity,
                        CAST({quote(path.object_key)} AS VARCHAR) object_id,
                        {quote(path.timestamp)}::TIMESTAMP event_time
                    FROM read_parquet({literal(self.database / (path.table + '.parquet'))})
                    WHERE {quote(path.object_key)} IS NOT NULL
                    {f'AND {quote(path.available_at)}<= {visible_until}' if path.available_at else ''}
                ), visible AS (
                    SELECT e.* FROM e JOIN requested_roots USING(entity)
                    WHERE event_time<{visible_until} AND event_time>={stamp}-INTERVAL 730 DAY
                ), objects AS (
                    SELECT entity,object_id,MIN(event_time) first_time,{','.join(object_fields)}
                    FROM visible GROUP BY entity,object_id
                ) SELECT entity,{','.join(root_fields)} FROM objects GROUP BY entity"""
                summary = self._summary(query, requested).set_index("entity")
                block = summary.reindex(ids).reset_index(drop=True)
                for name in block:
                    if not name.endswith(("_share", "_concentration")):
                        block[name] = block[name].fillna(0)
                blocks.append(block)
            frame = pd.concat(blocks, axis=1)
            frame["__position"] = positions
            chunks.append(frame)
        self.last_audit = dict(knowledge_boundary=str(phase),
            latest_query=str(times.max()), strict_source_visibility=True,
            windows_anchored_at="each forecast query, not the knowledge boundary",
            query_execution=dict(self.query_execution),
            query_recoveries=list(self.query_recoveries))
        return pd.concat(chunks).sort_values("__position").drop(columns="__position").reset_index(drop=True).astype("float32")
