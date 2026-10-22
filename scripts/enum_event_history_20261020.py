"""Bounded type-specific counts/recency on ordinary schema-declared events.

An undeclared low-cardinality numeric code can be an event TYPE even when its
name ends in Id. Do not confuse it with a declared structural key. Profile
only rows visible at the first consumed training anchor, retain a fixed small
vocabulary, and expose no raw identities or predictive-task outcomes.
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
from scripts.marked_response_history import ResponsePath, routed_sql


def source_metadata(path):
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    return {name: json.loads(metadata.get(name.encode(), b"null")) for name in
        ("fkey_col_to_pkey_table", "pkey_col", "time_col", "available_at")}


class EnumEventHistory:
    maximum_fields = 3
    maximum_values = 8
    maximum_columns = 162

    def __init__(self, database, root, *, discovery_cutoff, horizon, temporary):
        self.database, self.root = Path(database), root
        self.first, self.horizon = pd.Timestamp(discovery_cutoff), pd.Timedelta(horizon)
        if pd.isna(self.first) or self.horizon <= pd.Timedelta(0) or self.first.tz is not None:
            raise ValueError("Naive training-only discovery clock and positive horizon required")
        Path(temporary).mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(config=dict(memory_limit="2GB", threads=1,
            temp_directory=str(temporary), max_temp_directory_size="40GB"))
        self.tables = {p.stem: (p, source_metadata(p)) for p in sorted(self.database.glob("*.parquet"))}
        try:
            self.fields = self.discover()
        except BaseException:
            self.con.close()
            raise
        self.columns = [f"enum_event_{i}_{slot}_{metric}" for i, (_, vocabulary) in enumerate(self.fields)
            for slot in range(len(vocabulary)+1) for metric in ("count", "quarter_h", "one_h", "four_h", "last_age_h", "decayed_count")]
        self.audit = dict(strategy="schema_discovered_enum_event_history", root=root,
            fields=[dict(path=asdict(path), vocabulary=values, other_bucket=True) for path, values in self.fields],
            discovery_cutoff=str(self.first), maximum_fields=3, maximum_values=8, maximum_columns=162,
            source_rows_profiled_per_route=20_000, history_days=3650, task_labels_used=False,
            structural_keys_excluded_by_metadata=True, suffix_alone_does_not_discard_enums=True,
            event_and_ancestor_times="strictly before min(query time, phase boundary)",
            explicit_availability="also strictly before the same ceiling", raw_identifiers_emitted=False)

    def clocks(self, path):
        _, _, clocks = routed_sql(self.database, path)
        for table, alias in ((path.event, "e"), (path.ancestor, "p")):
            if table and self.tables[table][1].get("available_at"):
                clocks.append(f"CAST({alias}.{quote(self.tables[table][1]['available_at'])} AS TIMESTAMP)")
        return clocks

    def discover(self):
        candidates, routes = [], []
        for event, (file, meta) in self.tables.items():
            if event == self.root or not meta["time_col"]:
                continue
            excluded = {meta["pkey_col"], meta["time_col"], meta.get("available_at"), *(meta["fkey_col_to_pkey_table"] or {})}
            fields = [f.name for f in pq.ParquetFile(file).schema_arrow if f.name not in excluded
                and (pa.types.is_integer(f.type) or pa.types.is_floating(f.type) or pa.types.is_boolean(f.type))]
            for key, destination in sorted((meta["fkey_col_to_pkey_table"] or {}).items()):
                if destination == self.root:
                    routes.append((ResponsePath(event, meta["time_col"], key, "", None), fields))
                elif destination in self.tables and destination != event:
                    parent = self.tables[destination][1]
                    if parent["pkey_col"]:
                        for parent_key, target in sorted((parent["fkey_col_to_pkey_table"] or {}).items()):
                            if target == self.root:
                                routes.append((ResponsePath(event, meta["time_col"], parent_key, "", None,
                                    destination, key, parent["pkey_col"], parent["time_col"]), fields))
        for route, fields in sorted(routes, key=lambda r: (bool(r[0].ancestor), r[0].event, r[0].root_key, r[0].ancestor or ""))[:32]:
            if not fields:
                continue
            if route.ancestor:
                p = self.database / (route.ancestor+".parquet")
                invalid = self.con.execute(f"SELECT COUNT(*)-COUNT(DISTINCT {quote(route.ancestor_key)}) FROM read_parquet({literal(p)}) WHERE {quote(route.ancestor_key)} IS NOT NULL").fetchone()[0]
                if invalid:
                    raise ValueError("A declared ancestor key is not unique; refuse fanout")
            source, root, _ = routed_sql(self.database, route)
            visible = " AND ".join(f"{c}<TIMESTAMP {literal(self.first)}" for c in self.clocks(route))
            order = self.tables[route.event][1]["pkey_col"]
            ordered = f" ORDER BY e.{quote(order)}" if order else ""
            sample = self.con.execute(f"SELECT {','.join('e.'+quote(f) for f in fields)} FROM {source}"
                f" WHERE {visible} AND {root} IS NOT NULL{ordered} LIMIT 20000").df()
            for field in fields:
                values = pd.to_numeric(sample[field], errors="coerce").dropna()
                if (len(values) < 20 or not np.isfinite(values.to_numpy(float)).all()
                        or not np.equal(values, np.floor(values)).all() or not 2 <= values.nunique() <= 32
                        or values.abs().max() > 2**53 or values.nunique()/len(values) > .1):
                    continue
                counts = values.value_counts()
                vocabulary = [float(v) for v in sorted(counts.index, key=lambda v: (-int(counts[v]), float(v)))[:8]]
                path = ResponsePath(**{**asdict(route), "response": field})
                candidates.append((-int(values.nunique()), route.event, field, route.root_key, route.ancestor or "", path, vocabulary))
        if not candidates:
            raise ValueError("No training-visible low-cardinality ordinary event mark")
        return [(r[-2], r[-1]) for r in sorted(candidates, key=lambda r: r[:5])[:3]]

    def transform(self, requests, *, phase_boundary):
        if list(requests) != ["__event_entity", "__event_cutoff"] or requests.empty or requests.isna().any().any():
            raise ValueError("Only complete ordered masked entity/time requests may enter event history")
        times, boundary = pd.to_datetime(requests["__event_cutoff"]), pd.Timestamp(phase_boundary)
        if times.min() < self.first or boundary < self.first or times.dt.tz is not None or boundary.tz is not None:
            raise ValueError("No pre-discovery or timezone-ambiguous requests")
        h, parts = self.horizon.total_seconds(), []
        for clock in sorted(times.unique()):
            clock = pd.Timestamp(clock)
            pos = np.flatnonzero(times.eq(clock).to_numpy())
            ids = requests.iloc[pos]["__event_entity"].astype(str).to_numpy()
            self.con.register("enum_roots", pd.DataFrame(dict(entity=np.unique(ids))))
            stamp = f"TIMESTAMP {literal(clock)}"
            ceiling = f"TIMESTAMP {literal(min(clock, boundary))}"
            blocks = []
            for number, (path, vocabulary) in enumerate(self.fields):
                source, root, clocks = routed_sql(self.database, path)
                visible = " AND ".join(f"{c}<{ceiling}" for c in self.clocks(path))
                mark = f"CAST(e.{quote(path.response)} AS DOUBLE)"
                slot = "CASE "+" ".join(f"WHEN {mark}={v!r} THEN {i}" for i, v in enumerate(vocabulary))+f" ELSE {len(vocabulary)} END"
                rows = self.con.execute(f"""WITH events AS (
                    SELECT CAST({root} AS VARCHAR) entity,{clocks[0]} event_time,{slot} slot
                    FROM {source} JOIN enum_roots r ON CAST({root} AS VARCHAR)=r.entity
                    WHERE {visible} AND {clocks[0]}>={stamp}-INTERVAL 3650 DAY
                ) SELECT entity,slot,COUNT(*) count,
                    COUNT(*) FILTER(WHERE event_time>={stamp}-INTERVAL {literal(str(h/4)+' seconds')}) quarter_h,
                    COUNT(*) FILTER(WHERE event_time>={stamp}-INTERVAL {literal(str(h)+' seconds')}) one_h,
                    COUNT(*) FILTER(WHERE event_time>={stamp}-INTERVAL {literal(str(4*h)+' seconds')}) four_h,
                    EPOCH({stamp}-MAX(event_time))/{h!r} last_age_h,
                    SUM(EXP(-LN(2.)*EPOCH({stamp}-event_time)/{4*h!r})) decayed_count
                    FROM events GROUP BY entity,slot""").df()
                for index in range(len(vocabulary)+1):
                    block = rows.loc[rows.slot.eq(index)].drop(columns="slot").set_index("entity").reindex(ids).reset_index(drop=True)
                    names = ["count", "quarter_h", "one_h", "four_h", "decayed_count"]
                    block[names] = block[names].fillna(0.)
                    blocks.append(block.add_prefix(f"enum_event_{number}_{index}_"))
            block = pd.concat(blocks, axis=1)
            block["__position"] = pos
            parts.append(block)
        result = pd.concat(parts).sort_values("__position").drop(columns="__position").reindex(columns=self.columns).reset_index(drop=True).astype("float32")
        if len(result.columns) > self.maximum_columns or np.isinf(result.to_numpy()).any():
            raise ValueError("Complete enum-event block exceeds its bounded finite representation")
        return result

    def close(self):
        self.con.close()
