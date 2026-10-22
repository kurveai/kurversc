"""Bounded, identity-aware event breadth and object novelty, without raw IDs.

Foreign keys are routing/grouping keys, never predictors. Preserve the grain
of each contacted object before compressing frequency and arrival summaries.
This distinguishes many events on one object from genuinely broad activity.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from scripts.activity_timeline_challenger import quote, literal, table_metadata


@dataclass(frozen=True)
class ObjectPath:
    table: str
    entity: str
    object_key: str
    object_table: str
    timestamp: str
    available_at: str | None = None


class ObjectBreadth:
    def __init__(self, database, root, *, training_cutoff, horizon, temporary, threads=1):
        self.database = Path(database)
        self.first_query = pd.Timestamp(training_cutoff)
        self.days = pd.Timedelta(horizon).total_seconds()/86400
        if not np.isfinite(self.days) or self.days <= 0:
            raise ValueError("Positive finite horizon required")
        Path(temporary).mkdir(parents=True,exist_ok=True)
        self.con = duckdb.connect(config=dict(memory_limit="2GB",threads=threads,
            temp_directory=str(temporary),max_temp_directory_size="20GB"))
        try:
            metadata = {p.stem:table_metadata(p) for p in sorted(self.database.glob("*.parquet"))}
            candidates = []
            for name,info in metadata.items():
                if not info["time_col"] or name == root:
                    continue
                foreign = info["fkey_col_to_pkey_table"] or {}
                for entity,destination in sorted(foreign.items()):
                    if destination != root:
                        continue
                    for key,obj in sorted(foreign.items()):
                        if obj == root or obj not in metadata or not metadata[obj]["pkey_col"]:
                            continue
                        visible = f"{quote(info['time_col'])}<TIMESTAMP {literal(self.first_query)}"
                        if info.get("available_at"):
                            visible += f" AND {quote(info['available_at'])}<=TIMESTAMP {literal(self.first_query)}"
                        count = self.con.execute(f"SELECT COUNT(DISTINCT {quote(key)}) "
                            f"FROM read_parquet({literal(self.database/(name+'.parquet'))}) "
                            f"WHERE {visible} AND {quote(entity)} IS NOT NULL").fetchone()[0]
                        if count >= 64:
                            candidates.append((-count,ObjectPath(name,entity,key,obj,info["time_col"],info.get("available_at"))))
            self.paths = tuple(p for _,p in sorted(candidates,key=lambda x:(x[0],x[1].table,x[1].object_key))[:2])
            if not self.paths:
                raise ValueError("No training-visible high-cardinality object foreign key")
            self.audit = dict(paths=[asdict(p) for p in self.paths],maximum_paths=2,
                discovery_cutoff=str(self.first_query),minimum_distinct_objects=64,
                history_days=730,windows_in_horizons=[1/3,1,4],maximum_features=42,
                raw_identifiers_used_as_predictors=False,labels_used=False)
        except Exception:
            self.close()
            raise

    def transform(self,requests,*,phase_boundary):
        times = pd.to_datetime(requests["__event_cutoff"])
        if requests.empty or times.isna().any() or requests["__event_entity"].isna().any():
            raise ValueError("Complete nonempty routing/query keys required")
        if times.max()>pd.Timestamp(phase_boundary) or times.min()<self.first_query:
            raise ValueError("Requests exceed the authorized phase or precede discovery")
        chunks = []
        for cutoff in sorted(times.unique()):
            positions = np.flatnonzero(times.eq(cutoff).to_numpy())
            ids = requests.iloc[positions]["__event_entity"].astype(str).to_numpy()
            self.con.register("requested_roots",pd.DataFrame({"entity":np.unique(ids)}))
            stamp = f"TIMESTAMP {literal(pd.Timestamp(cutoff))}"
            blocks = []
            for number,path in enumerate(self.paths):
                object_fields,root_fields = [],[]
                for w,factor in enumerate((1/3,1,4)):
                    seconds = self.days*86400*factor
                    recent = f"event_time>={stamp}-INTERVAL {literal(str(seconds)+' seconds')}"
                    object_fields.append(f"COUNT(*) FILTER(WHERE {recent}) n{w}")
                    new = f"first_time>={stamp}-INTERVAL {literal(str(seconds)+' seconds')}"
                    prefix = f"ob_path{number}_w{w}"
                    root_fields += [f"COUNT(*) FILTER(WHERE n{w}>0) {prefix}_objects",
                        f"SUM(n{w}) {prefix}_events",
                        f"SUM((n{w}=1)::INTEGER) {prefix}_singletons",
                        f"COUNT(*) FILTER(WHERE {new}) {prefix}_new_objects",
                        f"SUM(n{w}) FILTER(WHERE {new}) {prefix}_new_events",
                        f"MAX(n{w})/NULLIF(SUM(n{w}),0)::DOUBLE {prefix}_maximum_share",
                        f"SUM(n{w}::DOUBLE*n{w})/NULLIF(SUM(n{w})::DOUBLE*SUM(n{w}),0) {prefix}_concentration"]
                query = f"""WITH e AS (
                    SELECT CAST({quote(path.entity)} AS VARCHAR) entity,
                        CAST({quote(path.object_key)} AS VARCHAR) object_id,
                        {quote(path.timestamp)}::TIMESTAMP event_time
                    FROM read_parquet({literal(self.database/(path.table+'.parquet'))})
                    WHERE {quote(path.object_key)} IS NOT NULL
                ), visible AS (
                    SELECT e.* FROM e JOIN requested_roots USING(entity)
                    WHERE event_time<{stamp} AND event_time>={stamp}-INTERVAL 730 DAY
                ), objects AS (
                    SELECT entity,object_id,MIN(event_time) first_time,{','.join(object_fields)}
                    FROM visible GROUP BY entity,object_id
                ) SELECT entity,{','.join(root_fields)} FROM objects GROUP BY entity"""
                summary = self.con.execute(query).df().set_index("entity")
                block = summary.reindex(ids).reset_index(drop=True)
                for name in block:
                    if not name.endswith(("_share","_concentration")):
                        block[name] = block[name].fillna(0)
                blocks.append(block)
            frame = pd.concat(blocks,axis=1)
            frame["__position"] = positions
            chunks.append(frame)
        return pd.concat(chunks).sort_values("__position").drop(columns="__position").reset_index(drop=True).astype("float32")

    def close(self):
        self.con.close()
