"""Experimental schema-derived event timelines and mature label state.

No KurveRSC or GraphReduce defaults are changed. These bounded numeric blocks
have no task-specific table/column names and never use an outcome before its
label horizon has completed. Ordinary events obey a strict timestamp cutoff.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def quote(name):
    return '"' + str(name).replace('"', '""') + '"'


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def table_metadata(path):
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    return {name: json.loads(metadata.get(name.encode(), b"null"))
            for name in ("fkey_col_to_pkey_table", "pkey_col", "time_col",
                         "available_at", "label_available_at")}


@dataclass(frozen=True)
class EventChannel:
    path: Path
    entity: str
    timestamp: str
    available_at: str | None = None


def discover_channels(database, root, max_channels=6):
    channels = []
    for path in sorted(Path(database).glob("*.parquet")):
        metadata = table_metadata(path)
        if path.stem == root or not metadata["time_col"]:
            continue
        for key, destination in sorted((metadata["fkey_col_to_pkey_table"] or {}).items()):
            if destination == root:
                channels.append(EventChannel(path, key, metadata["time_col"],metadata.get("available_at")))
    return tuple(channels[:max_channels])


class ActivityTimeline:
    """Union timed reverse-FK children before reducing to activity sessions."""

    windows = (7, 30, 90, 365, 730)

    def __init__(self, channels, *, temp_directory, threads=4):
        if not channels or len(channels) > 6:
            raise ValueError("Between one and six declared event channels required")
        self.channels = tuple(channels)
        Path(temp_directory).mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(config={"memory_limit": "4GB",
            "threads": threads, "temp_directory": str(temp_directory),
            "max_temp_directory_size": "40GB"})
        pieces = []
        for number, channel in enumerate(channels):
            pieces.append(f"SELECT {quote(channel.entity)}::BIGINT entity, "
                f"{quote(channel.timestamp)}::TIMESTAMP event_time, "
                f"{quote(channel.available_at or channel.timestamp)}::TIMESTAMP visible_at, {number} channel "
                f"FROM read_parquet({literal(channel.path)}) "
                f"WHERE {quote(channel.entity)} IS NOT NULL "
                f"AND {quote(channel.timestamp)} IS NOT NULL")
        self.connection.execute("CREATE VIEW event_rows AS " + " UNION ALL ".join(pieces))

    def transform(self, requests, *, phase_boundary):
        """Preserve request row order, including duplicated entities/cutoffs."""
        if requests["__event_cutoff"].isna().any():
            raise ValueError("Complete event cutoffs required")
        cutoffs = pd.to_datetime(requests["__event_cutoff"])
        if cutoffs.max() > pd.Timestamp(phase_boundary):
            raise ValueError("Request exceeds the authorized database phase")
        output = []
        for cutoff in sorted(cutoffs.unique()):
            positions = np.flatnonzero(cutoffs.eq(cutoff).to_numpy())
            ids = pd.DataFrame({"entity": pd.to_numeric(
                requests.iloc[positions]["__event_entity"]).astype("int64")})
            self.connection.register("requested_roots", ids.drop_duplicates())
            stamp = f"TIMESTAMP {literal(pd.Timestamp(cutoff))}"
            fields = [
                "COUNT(*) tl_active_days_730",
                "SUM(events) tl_events_730",
                "MIN(age_days) tl_recency_days",
                "MAX(age_days)-MIN(age_days) tl_active_span_days",
                "AVG(gap_days) tl_gap_mean_days",
                "STDDEV_POP(gap_days) tl_gap_std_days",
                "MEDIAN(gap_days) tl_gap_median_days",
                "QUANTILE_CONT(gap_days,0.9) tl_gap_p90_days",
                "MAX(gap_days) tl_gap_max_days",
                "AVG(gap_days) FILTER (WHERE recent_rank<=10) tl_recent_gap_mean",
                "AVG(gap_days) FILTER (WHERE recent_rank BETWEEN 11 AND 30) tl_older_gap_mean",
                "COUNT(*) FILTER (WHERE gap_days>=30) tl_long_gap_count",
                "AVG(channels) tl_channels_per_day",
                "MAX(events) tl_max_daily_events",
                "STDDEV_POP(events) tl_daily_events_std",
            ]
            for window in self.windows:
                condition = f"age_days<{window}"
                fields.extend([
                    f"SUM(events) FILTER (WHERE {condition}) tl_events_{window}",
                    f"COUNT(*) FILTER (WHERE {condition}) tl_days_{window}",
                    f"COUNT(DISTINCT DATE_TRUNC('week',event_day)) FILTER (WHERE {condition}) tl_weeks_{window}",
                    f"COUNT(*) FILTER (WHERE {condition} AND channels>1) tl_multichannel_days_{window}",
                    f"MAX(events) FILTER (WHERE {condition}) tl_daily_peak_{window}",
                ])
            fields.extend([f"COUNT(*) FILTER (WHERE EXTRACT(dow FROM event_day)={day}) tl_weekday_{day}"
                           for day in range(7)])
            query = f"""
                WITH recent AS (
                    SELECT e.* FROM event_rows e JOIN requested_roots r USING(entity)
                    WHERE event_time<{stamp} AND visible_at<={stamp} AND event_time>={stamp}-INTERVAL 730 DAY
                ), days AS (
                    SELECT entity,CAST(event_time AS DATE) event_day,COUNT(*) events,
                        COUNT(DISTINCT channel) channels
                    FROM recent GROUP BY entity,event_day
                ), gaps AS (
                    SELECT *, DATE_DIFF('day',event_day,{stamp}) age_days,
                        DATE_DIFF('day',LAG(event_day) OVER(PARTITION BY entity ORDER BY event_day),event_day) gap_days,
                        ROW_NUMBER() OVER(PARTITION BY entity ORDER BY event_day DESC) recent_rank
                    FROM days
                )
                SELECT entity,{','.join(fields)} FROM gaps GROUP BY entity
            """
            summary = self.connection.execute(query).df().set_index("entity")
            channel_fields = []
            for number in range(len(self.channels)):
                channel_fields.extend([
                    f"COUNT(*) FILTER (WHERE channel={number} AND event_time>={stamp}-INTERVAL 90 DAY) tl_channel{number}_events_90",
                    f"DATE_DIFF('day',MAX(event_time) FILTER (WHERE channel={number}),{stamp}) tl_channel{number}_recency",
                ])
            channels = self.connection.execute(f"SELECT entity,{','.join(channel_fields)} "
                f"FROM event_rows e JOIN requested_roots r USING(entity) WHERE event_time<{stamp} AND visible_at<={stamp} "
                f"AND event_time>={stamp}-INTERVAL 730 DAY GROUP BY entity").df().set_index("entity")
            summary = summary.join(channels)
            result = summary.reindex(ids.entity.to_numpy()).reset_index(drop=True)
            counts = [c for c in result if any(part in c for part in
                      ("_events_", "_days_", "_weeks_", "_count", "_weekday_"))]
            result[counts] = result[counts].fillna(0)
            result["tl_gap_cv"] = result.tl_gap_std_days / (1+result.tl_gap_mean_days)
            result["tl_normalized_recency"] = result.tl_recency_days / (1+result.tl_recent_gap_mean)
            result["tl_cadence_change"] = (1+result.tl_older_gap_mean)/(1+result.tl_recent_gap_mean)
            for short, long in ((7,30),(30,90),(90,365)):
                result[f"tl_active_day_rate_change_{short}_{long}"] = (
                    (result[f"tl_days_{short}"]+.5)/short /
                    ((result[f"tl_days_{long}"]+.5)/long))
            counts = result[[f"tl_weekday_{n}" for n in range(7)]].to_numpy()
            probabilities = (counts+.01)/(counts.sum(axis=1,keepdims=True)+.07)
            result["tl_weekday_entropy"] = -(probabilities*np.log(probabilities)).sum(axis=1)
            result["__position"] = positions
            output.append(result)
        if not output:
            raise ValueError("Nonempty timeline requests required")
        return pd.concat(output).sort_values("__position").drop(columns="__position").reset_index(drop=True).astype("float32")

    def close(self):
        self.connection.close()


def mature_label_state(history, requests, *, entity, timestamp, target, horizon,
                       phase_boundary, available_at=None):
    """Latest observed outcome, run lengths and EWMA from mature labels only."""
    history = history[[entity,timestamp,target,*([available_at] if available_at else [])]].copy()
    history[timestamp] = pd.to_datetime(history[timestamp])
    if history.duplicated([entity,timestamp]).any():
        raise ValueError("History requires unique entity/timestamp targets")
    history["__ready"] = pd.to_datetime(history[available_at]) if available_at else history[timestamp]+pd.Timedelta(horizon)
    # Explicit availability means observable BY that instant. Preserve the
    # old implicit strict contract for frozen benchmark/prototype callers.
    visible = (history.__ready.le(pd.Timestamp(phase_boundary)) if available_at
        else history.__ready.lt(pd.Timestamp(phase_boundary)))
    history = history.loc[visible]
    # Arrival order, not outcome-window order: an older result may arrive
    # after a newer result and must not contaminate earlier cumulative state.
    history = history.sort_values([entity,"__ready",timestamp],kind="stable")
    history["ls_last"] = pd.to_numeric(history[target]).astype(float)
    if not history.ls_last.isin([0,1]).all():
        raise ValueError("Binary outcome history required for this diagnostic")
    grouped = history.groupby(entity,sort=False)
    history["ls_count"] = grouped.cumcount()+1
    history["ls_rate"] = grouped.ls_last.expanding().mean().reset_index(level=0,drop=True)
    history["ls_recent_mean3"] = grouped.ls_last.rolling(3,min_periods=1).mean().reset_index(level=0,drop=True)
    history["ls_recent_mean5"] = grouped.ls_last.rolling(5,min_periods=1).mean().reset_index(level=0,drop=True)
    history["ls_ewma"] = grouped.ls_last.transform(lambda x:x.ewm(alpha=.5,adjust=False).mean())
    changed = history.ls_last.ne(grouped.ls_last.shift())
    run = changed.groupby(history[entity]).cumsum()
    history["ls_run_length"] = history.groupby([entity,run]).cumcount()+1
    history["ls_signed_run"] = history.ls_run_length*(2*history.ls_last-1)
    positives = history["__ready"].where(history.ls_last.eq(1))
    history["__last_positive"] = positives.groupby(history[entity]).ffill()
    states = []
    cutoffs = pd.to_datetime(requests["__event_cutoff"])
    if cutoffs.max()>pd.Timestamp(phase_boundary):
        raise ValueError("Label-state request exceeds the authorized phase")
    columns = [c for c in history if c.startswith("ls_")]
    for cutoff in sorted(cutoffs.unique()):
        positions = np.flatnonzero(cutoffs.eq(cutoff).to_numpy())
        visible = (history.__ready.le(pd.Timestamp(cutoff)) if available_at
            else history.__ready.lt(pd.Timestamp(cutoff)))
        available = history.loc[visible]
        last = available.groupby(entity,sort=False).tail(1).set_index(entity)
        last = last.reindex(requests.iloc[positions].__event_entity.to_numpy())
        result = last[columns].reset_index(drop=True)
        result["ls_days_since_observed"] = (pd.Timestamp(cutoff)-last.__ready).dt.total_seconds().to_numpy()/86400
        result["ls_days_since_positive"] = (pd.Timestamp(cutoff)-last.__last_positive).dt.total_seconds().to_numpy()/86400
        result["ls_count"] = result.ls_count.fillna(0)
        result["__position"] = positions
        states.append(result)
    return pd.concat(states).sort_values("__position").drop(columns="__position").reset_index(drop=True).astype("float32")
