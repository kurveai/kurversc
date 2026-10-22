"""Bounded exact-time sequence summaries of schema-declared event channels.

Unlike daily recurrence, distinct event timestamps preserve within-day return
gaps. Ordinary events only: no task target, names, ID payload or labels are
used. This is an isolated extension of sequence algebra, not a new registry
family or a production change.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from scripts.activity_timeline_challenger import literal, quote


class SubdaySequence:
    maximum_columns = 224

    def __init__(self, channels, *, horizon, temporary, threads=1):
        self.channels = tuple(channels)
        self.horizon = pd.Timedelta(horizon).total_seconds()
        if not 1 <= len(self.channels) <= 6 or not np.isfinite(self.horizon) or self.horizon <= 0:
            raise ValueError("One to six event channels and a positive horizon required")
        Path(temporary).mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(config=dict(memory_limit="2GB", threads=threads,
            temp_directory=str(temporary), max_temp_directory_size="12GB"))
        pieces = [f"SELECT CAST({quote(c.entity)} AS VARCHAR) entity, "
            f"CAST({quote(c.timestamp)} AS TIMESTAMP) event_time,{i} channel "
            f"FROM read_parquet({literal(c.path)}) WHERE {quote(c.entity)} IS NOT NULL "
            f"AND {quote(c.timestamp)} IS NOT NULL" for i, c in enumerate(self.channels)]
        self.con.execute("CREATE VIEW sequence_source AS " + " UNION ALL ".join(pieces))
        self.audit = dict(strategy="exact_subday_sequence", horizon_seconds=self.horizon,
            channel_count=len(self.channels), maximum_columns=self.maximum_columns,
            history_days=3650, maximum_channel_instants=5_000_000,
            session_gaps_horizons=[1 / 64, 1 / 16], task_labels_used=False,
            source_clock_rule="event_time < min(query_time, phase_boundary)",
            identifiers_emitted=False)

    def transform(self, requests, *, phase_boundary):
        times = pd.to_datetime(requests["__event_cutoff"])
        boundary = pd.Timestamp(phase_boundary)
        if (requests.empty or times.isna().any() or pd.isna(boundary)
                or requests["__event_entity"].isna().any()
                or times.dt.tz is not None or boundary.tz is not None):
            raise ValueError("Nonempty requests with complete naive clocks and entity keys required")
        outputs, phase_audit = [], []
        h = self.horizon
        for clock in sorted(times.unique()):
            clock = pd.Timestamp(clock)
            positions = np.flatnonzero(times.eq(clock).to_numpy())
            ids = requests.iloc[positions]["__event_entity"].astype(str).to_numpy()
            self.con.register("sequence_roots", pd.DataFrame(dict(entity=np.unique(ids))))
            stamp = f"TIMESTAMP {literal(clock)}"
            ceiling = f"TIMESTAMP {literal(min(clock, boundary))}"
            self.con.execute(f"""CREATE OR REPLACE TEMP TABLE sequence_instants AS
                SELECT entity,channel,event_time,COUNT(*) events
                FROM sequence_source JOIN sequence_roots USING(entity)
                WHERE event_time<{ceiling} AND event_time>={stamp}-INTERVAL 3650 DAY
                GROUP BY entity,channel,event_time""")
            instants = self.con.execute("SELECT COUNT(*) FROM sequence_instants").fetchone()[0]
            if instants > self.audit["maximum_channel_instants"]:
                raise ValueError("Complete sequence block exceeds the declared source capacity")
            fields = ["COUNT(*) instants", "SUM(events) events",
                f"MAX(age)/{h!r} age_h",  # age is constant within an entity/channel
                f"EPOCH(MAX(event_time)-MIN(event_time))/{h!r} span_h",
                "COUNT(gap) gap_support", f"AVG(gap)/{h!r} gap_mean_h",
                f"MEDIAN(gap)/{h!r} gap_median_h", f"QUANTILE_CONT(gap,.9)/{h!r} gap_p90_h",
                "STDDEV_POP(gap)/NULLIF(AVG(gap),0) gap_cv",
                f"MAX(gap) FILTER(WHERE recent_rank=1)/{h!r} latest_gap_h",
                f"AVG(gap) FILTER(WHERE recent_rank<=8)/{h!r} recent8_gap_h",
                f"AVG(gap) FILTER(WHERE recent_rank BETWEEN 9 AND 32)/{h!r} older8_32_gap_h",
                f"1+COUNT(*) FILTER(WHERE gap>={h / 64!r}) session_count_h64",
                f"1+COUNT(*) FILTER(WHERE gap>={h / 16!r}) session_count_h16",
                f"COUNT(*) FILTER(WHERE event_time>={stamp}-INTERVAL {literal(str(h) + ' seconds')}) instants_h",
                f"COUNT(*) FILTER(WHERE event_time>={stamp}-INTERVAL {literal(str(h / 4) + ' seconds')}) instants_quarter_h",
                f"SUM(events) FILTER(WHERE event_time>={stamp}-INTERVAL {literal(str(h) + ' seconds')}) events_h",
                f"SUM(events) FILTER(WHERE event_time>={stamp}-INTERVAL {literal(str(h / 4) + ' seconds')}) events_quarter_h",
                "COUNT(*) FILTER(WHERE gap>=age) gap_beyond_age",
                f"COUNT(*) FILTER(WHERE gap>=age+{h!r}) gap_beyond_next_h"]
            result = self.con.execute(f"""WITH expanded AS (
                SELECT * FROM sequence_instants
                UNION ALL
                SELECT entity,-1 channel,event_time,SUM(events) events
                FROM sequence_instants GROUP BY entity,event_time
            ), ordered AS (
                SELECT *,EPOCH(event_time-LAG(event_time) OVER
                    (PARTITION BY entity,channel ORDER BY event_time)) gap,
                    EPOCH({stamp}-MAX(event_time) OVER(PARTITION BY entity,channel)) age,
                    ROW_NUMBER() OVER(PARTITION BY entity,channel ORDER BY event_time DESC) recent_rank
                FROM expanded
            ) SELECT entity,channel,{','.join(fields)} FROM ordered GROUP BY entity,channel""").df()
            blocks = []
            for channel in (-1, *range(len(self.channels))):
                block = result.loc[result.channel.eq(channel)].drop(columns="channel").set_index("entity")
                block = block.reindex(ids).reset_index(drop=True)
                counts = ["instants", "events", "gap_support", "session_count_h64", "session_count_h16",
                    "instants_h", "instants_quarter_h", "events_h", "events_quarter_h",
                    "gap_beyond_age", "gap_beyond_next_h"]
                block[counts] = block[counts].fillna(0)
                block["age_over_median"] = block.age_h / block.gap_median_h.replace(0, np.nan)
                block["age_over_recent"] = block.age_h / block.recent8_gap_h.replace(0, np.nan)
                block["cadence_change"] = block.older8_32_gap_h / block.recent8_gap_h.replace(0, np.nan)
                block["instant_density"] = block.instants / (1 + block.span_h)
                block["session_density"] = block.session_count_h64 / (1 + block.span_h)
                # A shrunk closed-gap sketch, not a fitted survival probability.
                block["conditional_return_sketch"] = np.where(block.gap_support >= 3,
                    1 - (block.gap_beyond_next_h + 1) / (block.gap_beyond_age + 2), np.nan)
                block["last_gap_over_median"] = block.latest_gap_h / block.gap_median_h.replace(0, np.nan)
                blocks.append(block.add_prefix("sq_all_" if channel == -1 else f"sq_channel{channel}_"))
            values = pd.concat(blocks, axis=1)
            if len(values.columns) > self.maximum_columns or np.isinf(values.to_numpy(float)).any():
                raise ValueError("Sequence output exceeds its finite bounded representation")
            values["__position"] = positions
            outputs.append(values)
            phase_audit.append(dict(query_cutoff=str(clock), source_ceiling=str(min(clock, boundary)),
                requested_rows=len(positions), channel_instants=instants))
        output = pd.concat(outputs).sort_values("__position").drop(columns="__position").reset_index(drop=True).astype("float32")
        self.support = output.sq_all_instants.gt(0).to_numpy()
        self.last_audit = dict(phases=phase_audit, rows=len(output), columns=len(output.columns),
            supported_rows=int(self.support.sum()))
        return output

    def close(self):
        self.con.close()
