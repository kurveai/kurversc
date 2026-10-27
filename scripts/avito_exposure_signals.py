"""One-off Avito exposure experiment; not a GraphReduce/RSC feature family.

Exact-ad and exact-ad/position response priors exclude ALL impressions from
the query user. Ordinary event rows must be strictly before the example
cutoff, including their SearchInfo ancestors. No predictive-task labels are
accepted. Aggregated priors join many-to-one, never impression-to-impression.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from kurversc.event_summaries import _compatible_frame


class ExposureAdjustedSignals:
    def __init__(self, search_info, search_stream, ads_info, *, horizon="4d",
                 memory_limit="32GB", threads=4, temp_directory=None):
        if search_info.SearchID.duplicated().any() or ads_info.AdID.duplicated().any():
            raise ValueError("Declared search/ad keys must be unique")
        if ads_info.AdID.lt(0).any():
            raise ValueError("RelBench ad keys must be nonnegative")
        self.horizon = pd.Timedelta(horizon)
        if self.horizon <= pd.Timedelta(0):
            raise ValueError("Positive horizon required")
        config = {"memory_limit": memory_limit, "threads": threads}
        if temp_directory is not None:
            config["temp_directory"] = str(temp_directory)
        self.con = duckdb.connect(config=config)
        self.con.register("search_info", _compatible_frame(search_info))
        self.con.register("search_stream", _compatible_frame(search_stream))
        self.availability = {name:"AvailableAt" in frame for name,frame in
            (("search_info",search_info),("search_stream",search_stream),("ads_info",ads_info))}
        self.con.register("ads_info", _compatible_frame(ads_info[["AdID",*(["AvailableAt"] if self.availability["ads_info"] else [])]]))
        self.last_audit = {}

    def close(self):
        self.con.close()

    def transform(self, ids, cutoff):
        cutoff = pd.Timestamp(cutoff)
        if pd.isna(cutoff) or cutoff.tz is not None:
            raise ValueError("Valid timezone-naive cutoff required")
        ids = pd.Index(ids)
        if ids.hasnans:
            raise ValueError("Missing user identities")
        self.con.register("seeds", _compatible_frame(pd.DataFrame({"entity": ids.unique()})))
        con = self.con
        object_visible = f" AND a.AvailableAt<=TIMESTAMP '{cutoff}'" if self.availability["ads_info"] else ""
        parent_visible = f" AND s.AvailableAt<=TIMESTAMP '{cutoff}'" if self.availability["search_info"] else ""
        event_visible = f" AND e.AvailableAt<=TIMESTAMP '{cutoff}'" if self.availability["search_stream"] else ""
        # Include the full eligible population in peer priors, not just the
        # users in this feature batch. Missing ad identities fall back to the
        # other-user global rate rather than silently dropping impressions.
        con.execute(f"""CREATE OR REPLACE TEMP TABLE events AS
            SELECT s.UserID AS entity, e.SearchID AS search_id,
                   coalesce(CAST(a.AdID AS BIGINT),-1) AS object_id,
                   coalesce(CAST(e.Position AS VARCHAR), '__missing__') AS context,
                   e.SearchDate AS event_time,
                   CAST(e.IsClick AS DOUBLE) AS response,
                   CASE WHEN isfinite(TRY_CAST(e.HistCTR AS DOUBLE))
                        THEN least(1., greatest(0., TRY_CAST(e.HistCTR AS DOUBLE)))
                        ELSE NULL END AS histctr
            FROM search_stream e JOIN search_info s ON e.SearchID = s.SearchID
            LEFT JOIN ads_info a ON e.AdID = a.AdID {object_visible}
            WHERE e.SearchDate < TIMESTAMP '{cutoff}'
              AND s.SearchDate < TIMESTAMP '{cutoff}' AND s.UserID IS NOT NULL
              {parent_visible} {event_visible}
              AND e.IsClick IS NOT NULL
        """)
        if con.execute("SELECT count(*) FROM events WHERE response NOT IN (0,1)").fetchone()[0]:
            raise ValueError("Expected binary historical click response")
        con.execute("""CREATE OR REPLACE TEMP TABLE totals AS
            SELECT count(*) AS n, coalesce(sum(response),0) AS clicks FROM events""")
        con.execute("""CREATE OR REPLACE TEMP TABLE own_events AS
            SELECT e.* FROM events e JOIN seeds s ON e.entity=s.entity""")
        for name, group, table in (("user", "entity", "own_events"),
                            ("ad", "object_id", "events"),
                            ("user_ad", "entity, object_id", "own_events"),
                            ("ad_context", "object_id, context", "events"),
                            ("user_ad_context", "entity, object_id, context", "own_events")):
            con.execute(f"""CREATE OR REPLACE TEMP TABLE {name}_totals AS
                SELECT {group}, count(*) AS n, sum(response) AS clicks
                FROM {table} GROUP BY {group}""")
        # Subtract the query user's entire history, not just the current row.
        # The ad prior shrinks to the other-user global rate (20 pseudo-events),
        # and ad/position shrinks to that ad prior (10 pseudo-events).
        # Materialization barriers bound each lookup at user/ad grain rather
        # than relying on the optimizer's many-way nullable join ordering.
        con.execute("""CREATE OR REPLACE TEMP TABLE user_priors AS
            SELECT u.entity,
                   coalesce((t.clicks-u.clicks)/nullif(t.n-u.n,0),0.) AS prior
            FROM user_totals u CROSS JOIN totals t""")
        con.execute("""CREATE OR REPLACE TEMP TABLE peer_ads AS
            SELECT u.entity,u.object_id,a.n-u.n AS peer_n,
                   CASE WHEN u.object_id=-1 THEN p.prior
                        ELSE (a.clicks-u.clicks+20.*p.prior)/(a.n-u.n+20.)
                        END AS expected_ad
            FROM user_ad_totals u JOIN ad_totals a ON u.object_id=a.object_id
            JOIN user_priors p ON u.entity=p.entity""")
        con.execute("""CREATE OR REPLACE TEMP TABLE peer_ad_contexts AS
            SELECT u.entity,u.object_id,u.context,p.peer_n,p.expected_ad,
                   a.n-u.n AS peer_context_n,
                   CASE WHEN u.object_id=-1 THEN p.expected_ad
                        ELSE (a.clicks-u.clicks+10.*p.expected_ad)/(a.n-u.n+10.)
                        END AS expected_ad_position
            FROM user_ad_context_totals u JOIN ad_context_totals a
                ON u.object_id=a.object_id AND u.context=a.context
            JOIN peer_ads p ON u.entity=p.entity AND u.object_id=p.object_id""")
        con.execute("""CREATE OR REPLACE TEMP TABLE scored AS
            SELECT e.*,p.peer_n,p.expected_ad,p.peer_context_n,p.expected_ad_position
            FROM own_events e JOIN peer_ad_contexts p
                ON e.entity=p.entity AND e.object_id=p.object_id AND e.context=p.context""")
        own_count=con.execute("SELECT count(*) FROM own_events").fetchone()[0]
        scored_count=con.execute("SELECT count(*) FROM scored").fetchone()[0]
        if own_count!=scored_count:
            raise ValueError("Peer lookups changed impression grain")
        self.last_audit = dict(
            cutoff=str(cutoff), full_population_events=con.execute("SELECT n FROM totals").fetchone()[0],
            scored_events=scored_count,
            peer_exclusion="All events from query user excluded from global, ad and ad/position priors",
            availability="Both impression and search timestamps strictly before example/phase cutoff",
            path=["UserInfo", "SearchInfo", "SearchStream", "AdsInfo", "SearchStream(second instance)"],
            prior_strengths={"ad": 20, "ad_position": 10},
            labels_used=False, history_sampling=False,
        )
        # Every arm gets the same bounded windows; no cutoff-specific tuning.
        windows = [("all", "TRUE")] + [
            (f"w{i}", f"event_time >= TIMESTAMP '{cutoff}' - INTERVAL '{int(self.horizon.total_seconds()*fraction)} SECOND'")
            for i, fraction in enumerate((.25, .5, 1.))
        ]
        blocks = {arm: [] for arm in ("activity", "quality", "histctr", "ad", "ad_position")}
        for window, where in windows:
            expressions = {
                "activity": [
                    ("exposures", "count(*)"), ("clicks", "sum(response)"),
                    ("click_rate", "avg(response)"), ("distinct_ads", "count(DISTINCT object_id) FILTER (WHERE object_id<>-1)"),
                    ("distinct_searches", "count(DISTINCT search_id)"),
                    ("active_hours", "count(DISTINCT date_trunc('hour',event_time))"),
                    ("hours_since_click", f"date_diff('second',max(event_time) FILTER (WHERE response=1),TIMESTAMP '{cutoff}')/3600."),
                ],
                "quality": [
                    ("ad_expected_mean", "avg(expected_ad)"),
                    ("ad_expected_std", "stddev_pop(expected_ad)"),
                    ("ad_expected_min", "min(expected_ad)"), ("ad_expected_max", "max(expected_ad)"),
                    ("ad_peer_support", "avg(ln(1.+peer_n))"),
                    ("ad_peer_coverage", "avg(CAST(peer_n>0 AND object_id<>-1 AS DOUBLE))"),
                    ("ad_position_expected_mean", "avg(expected_ad_position)"),
                    ("ad_position_peer_support", "avg(ln(1.+peer_context_n))"),
                    ("histctr_expected_mean", "avg(histctr)"),
                    ("histctr_coverage", "avg(CAST(histctr IS NOT NULL AS DOUBLE))"),
                ],
            }
            for arm, expected in (("histctr", "histctr"), ("ad", "expected_ad"),
                                  ("ad_position", "expected_ad_position")):
                expressions[arm] = [
                    ("residual_mean", f"avg(response-{expected})"),
                    ("residual_sum", f"sum(response-{expected})"),
                    ("residual_positive_fraction", f"avg(CASE WHEN {expected} IS NOT NULL THEN CAST(response>{expected} AS DOUBLE) END)"),
                    ("residual_negative_fraction", f"avg(CASE WHEN {expected} IS NOT NULL THEN CAST(response<{expected} AS DOUBLE) END)"),
                    ("residual_z", f"sum(response-{expected})/sqrt(1.+sum({expected}*(1.-{expected})))"),
                    ("unexpected_clicks", f"sum(response*(1.-{expected}))"),
                ]
            for arm, exprs in expressions.items():
                sql = ", ".join(f'{expr} AS "{name}"' for name, expr in exprs)
                frame = con.execute(f"SELECT entity, {sql} FROM scored WHERE {where} GROUP BY entity").df().set_index("entity").reindex(ids)
                zero = [c for c in frame if c in ("exposures", "clicks", "distinct_ads", "distinct_searches", "active_hours", "residual_sum", "unexpected_clicks")]
                frame[zero] = frame[zero].fillna(0)
                blocks[arm].append(frame.add_prefix(f"exposure_{arm}_{window}_"))
        return {arm: pd.concat(parts, axis=1).reset_index(drop=True).astype("float32")
                for arm, parts in blocks.items()}
