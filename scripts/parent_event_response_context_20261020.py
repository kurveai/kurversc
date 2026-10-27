"""Ordinary event-response priors shared through nonstructural parent values.

Unlike mature task-target peers, the supervision here is an already observed
binary mark on a timed event. A root without events can borrow other roots'
response behavior through ordinary parent categories. Metadata routes joins;
keys and task outcomes never enter the payload. Own-root evidence is removed.
"""

from dataclasses import asdict
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.activity_timeline_challenger import literal, quote
from scripts.enum_event_history_20261020 import source_metadata
from scripts.marked_response_history import discover_response_path, routed_sql


class ParentEventResponseContext:
    maximum_columns = 48

    def __init__(self, database, root, *, discovery_cutoff, horizon, temporary, earliest_query=None):
        self.database, self.root = Path(database), root
        self.first, self.horizon = pd.Timestamp(discovery_cutoff), pd.Timedelta(horizon)
        self.minimum_query = pd.Timestamp(earliest_query) if earliest_query is not None else self.first
        if pd.isna(self.first) or self.first.tz is not None or self.horizon <= pd.Timedelta(0):
            raise ValueError("Training-only naive discovery clock and positive horizon required")
        if pd.isna(self.minimum_query) or self.minimum_query.tz is not None or self.minimum_query > self.first:
            raise ValueError("Earliest consumed training anchor must not follow training vocabulary discovery")
        Path(temporary).mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(config=dict(memory_limit="4GB", threads=1,
            temp_directory=str(temporary), max_temp_directory_size="40GB"))
        try:
            self.con.execute("SET enable_progress_bar=false")
            self.path = discover_response_path(self.database, root, self.first, self.con)
            self.root_file = self.database / (root+".parquet")
            self.meta = source_metadata(self.root_file)
            if not isinstance(self.meta["pkey_col"], str):
                raise ValueError("A scalar unique parent key is required")
            duplicates = self.con.execute(f"SELECT COUNT(*)-COUNT(DISTINCT {quote(self.meta['pkey_col'])}) "
                f"FROM read_parquet({literal(self.root_file)}) WHERE {quote(self.meta['pkey_col'])} IS NOT NULL").fetchone()[0]
            if duplicates:
                raise ValueError("Declared parent key is not unique; refuse fanout")
            excluded = {self.meta["pkey_col"], self.meta["time_col"], self.meta["available_at"],
                *(self.meta["fkey_col_to_pkey_table"] or {})}
            candidates = [f.name for f in pq.ParquetFile(self.root_file).schema_arrow if f.name not in excluded
                and (pa.types.is_integer(f.type) or pa.types.is_floating(f.type) or pa.types.is_boolean(f.type)
                     or pa.types.is_string(f.type) or pa.types.is_large_string(f.type))]
            root_visible = self.root_visibility(self.first, alias="a")
            sample = self.con.execute(f"SELECT * FROM read_parquet({literal(self.root_file)}) a "
                f"WHERE {root_visible} ORDER BY {quote(self.meta['pkey_col'])} LIMIT 20000").df()
            self.fields = []
            for c in candidates:
                values = sample[c].dropna()
                if not 2 <= values.nunique() <= 4096 or values.nunique()/max(1, len(values)) > .5:
                    continue
                if pd.api.types.is_numeric_dtype(values.dtype):
                    numeric = pd.to_numeric(values).to_numpy(float)
                    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
                        continue
                self.fields.append(c)
                if len(self.fields) == 4:
                    break
            if not self.fields:
                raise ValueError("No bounded nonstructural nominal parent attributes")
            self.clocks = routed_sql(self.database, self.path)[2]
            for table, alias in ((self.path.event, "e"), (self.path.ancestor, "p")):
                if table:
                    visible = source_metadata(self.database / (table+".parquet"))["available_at"]
                    if visible:
                        self.clocks.append(f"CAST({alias}.{quote(visible)} AS TIMESTAMP)")
            # The legacy discovery helper does not know later availability
            # clocks. Reject those paths instead of discovering from late data.
            if len(self.clocks) != len(routed_sql(self.database, self.path)[2]):
                raise ValueError("Explicit event availability needs availability-safe mark discovery")
        except BaseException:
            self.con.close()
            raise
        self.columns = [f"parent_response_{i}_w{w}_{metric}" for i in range(len(self.fields))
            for w in range(3) for metric in ("peer_support", "event_rate", "root_rate", "positive_root_share")]
        self.audit = dict(strategy="other_root_ordinary_mark_parent_context", path=asdict(self.path),
            parent_fields=self.fields, discovery_cutoff=str(self.first), columns=len(self.columns),
            earliest_training_query=str(self.minimum_query), vocabulary_target_free=True,
            maximum_columns=48, maximum_parent_fields=4, maximum_parent_cardinality=4096,
            windows_horizons=[None, 1., 4.], prior_strength=20.,
            task_targets_used=False, raw_identifiers_emitted=False, own_root_excluded=True,
            event_visibility="strict event and ancestor clocks before min(query, phase)",
            root_visibility="declared parent clock/availability also strictly before ceiling",
            entity_balanced_rate=True, query_batch_statistics_used=False)

    def root_visibility(self, ceiling, *, alias):
        columns = set(c for c in (self.meta["time_col"], self.meta["available_at"]) if c)
        return " AND ".join(f"CAST({alias}.{quote(c)} AS TIMESTAMP)<TIMESTAMP {literal(ceiling)}"
            for c in sorted(columns)) or "TRUE"

    def transform(self, requests, *, phase_boundary):
        if list(requests) != ["__event_entity", "__event_cutoff"] or requests.empty or requests.isna().any().any():
            raise ValueError("Only complete masked entity/time routing requests are allowed")
        times, boundary = pd.to_datetime(requests["__event_cutoff"]), pd.Timestamp(phase_boundary)
        if times.min() < self.minimum_query or pd.isna(boundary) or boundary < self.first:
            raise ValueError("No pre-discovery source extraction")
        source, root, event_clocks = routed_sql(self.database, self.path)
        response = f"CAST(e.{quote(self.path.response)} AS DOUBLE)"
        pieces = []
        for t in sorted(times.unique()):
            t = pd.Timestamp(t)
            ceiling = min(t, boundary)
            positions = np.flatnonzero(times.eq(t).to_numpy())
            ids = requests.iloc[positions]["__event_entity"].astype(str).to_numpy()
            self.con.register("parent_response_queries", pd.DataFrame(dict(position=np.arange(len(ids)), entity=ids)))
            visibility = " AND ".join(f"{c}<TIMESTAMP {literal(ceiling)}" for c in self.clocks)
            parent_columns = ",".join(f"CAST(a.{quote(c)} AS VARCHAR) attr{i}" for i, c in enumerate(self.fields))
            self.con.execute(f"""CREATE OR REPLACE TEMP TABLE parent_response_events AS
                SELECT CAST({root} AS VARCHAR) entity,{event_clocks[0]} event_time,{response} response,{parent_columns}
                FROM {source} JOIN read_parquet({literal(self.root_file)}) a
                  ON CAST({root} AS VARCHAR)=CAST(a.{quote(self.meta['pkey_col'])} AS VARCHAR)
                WHERE {visibility} AND {self.root_visibility(ceiling, alias='a')}
                    AND {response} IS NOT NULL AND {event_clocks[0]}>=TIMESTAMP {literal(t)}-INTERVAL 3650 DAY""")
            if self.con.execute("SELECT COUNT(*) FROM parent_response_events WHERE response NOT IN (0,1)").fetchone()[0]:
                raise ValueError("Discovered ordinary binary mark changed domain")
            self.con.execute(f"""CREATE OR REPLACE TEMP TABLE parent_response_requests AS
                SELECT q.*,{parent_columns} FROM parent_response_queries q
                LEFT JOIN read_parquet({literal(self.root_file)}) a
                  ON q.entity=CAST(a.{quote(self.meta['pkey_col'])} AS VARCHAR)
                  AND {self.root_visibility(ceiling, alias='a')}""")
            blocks = []
            for w, scale in enumerate((None, 1., 4.)):
                condition = "TRUE" if scale is None else f"event_time>=TIMESTAMP {literal(t)}-INTERVAL {literal(str(scale*self.horizon.total_seconds())+' seconds')}"
                self.con.execute(f"""CREATE OR REPLACE TEMP TABLE parent_response_roots AS
                    SELECT entity,{','.join('attr'+str(i) for i in range(len(self.fields)))},COUNT(*) n,SUM(response) positives
                    FROM parent_response_events WHERE {condition}
                    GROUP BY entity,{','.join('attr'+str(i) for i in range(len(self.fields)))}""")
                for i in range(len(self.fields)):
                    # All totals come from source roots, not the queried batch.
                    # Unique root contributions make leave-own-root subtraction exact.
                    result = self.con.execute(f"""WITH totals AS (
                        SELECT COUNT(*) roots,COALESCE(SUM(n),0) n,COALESCE(SUM(positives),0) p,
                            COALESCE(SUM(positives/n),0) rates,COALESCE(SUM((positives>0)::INTEGER),0) positive_roots
                        FROM parent_response_roots
                    ), categories AS (
                        SELECT attr{i},COUNT(*) roots,SUM(n) n,SUM(positives) p,
                            SUM(positives/n) rates,SUM((positives>0)::INTEGER) positive_roots
                        FROM parent_response_roots WHERE attr{i} IS NOT NULL GROUP BY attr{i}
                    ), reduced AS (
                        SELECT q.position,
                            COALESCE(c.roots,0)-CASE WHEN q.attr{i} IS NOT NULL THEN (o.entity IS NOT NULL)::INTEGER ELSE 0 END peers,
                            COALESCE(c.n,0)-CASE WHEN q.attr{i} IS NOT NULL THEN COALESCE(o.n,0) ELSE 0 END cn,
                            COALESCE(c.p,0)-CASE WHEN q.attr{i} IS NOT NULL THEN COALESCE(o.positives,0) ELSE 0 END cp,
                            COALESCE(c.rates,0)-CASE WHEN q.attr{i} IS NOT NULL THEN COALESCE(o.positives/o.n,0) ELSE 0 END cr,
                            COALESCE(c.positive_roots,0)-CASE WHEN q.attr{i} IS NOT NULL THEN COALESCE((o.positives>0)::INTEGER,0) ELSE 0 END cpr,
                            (t.p-COALESCE(o.positives,0))/NULLIF(t.n-COALESCE(o.n,0),0) event_prior,
                            (t.rates-COALESCE(o.positives/o.n,0))/NULLIF(t.roots-(o.entity IS NOT NULL)::INTEGER,0) root_prior,
                            (t.positive_roots-COALESCE((o.positives>0)::INTEGER,0))/NULLIF(t.roots-(o.entity IS NOT NULL)::INTEGER,0) positive_prior
                        FROM parent_response_requests q CROSS JOIN totals t
                        LEFT JOIN categories c ON q.attr{i}=c.attr{i}
                        LEFT JOIN parent_response_roots o ON q.entity=o.entity
                    ) SELECT position,peers peer_support,
                        (cp+20.*event_prior)/(cn+20.) event_rate,
                        (cr+20.*root_prior)/(peers+20.) root_rate,
                        (cpr+20.*positive_prior)/(peers+20.) positive_root_share
                    FROM reduced ORDER BY position""").df().drop(columns="position")
                    result.columns = [f"parent_response_{i}_w{w}_{c}" for c in result]
                    blocks.append(result)
            block = pd.concat(blocks, axis=1).reindex(columns=self.columns)
            block["__position"] = positions
            pieces.append(block)
        result = pd.concat(pieces).sort_values("__position").drop(columns="__position").reset_index(drop=True).astype("float32")
        if len(result.columns) > 48 or np.isinf(result.to_numpy()).any():
            raise ValueError("Parent response context exceeded its bounded representation")
        return result

    def close(self):
        self.con.close()
