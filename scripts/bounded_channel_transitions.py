"""Equivalent channel-set transitions without correlated UNNEST expansion.

The admitted first version stays frozen. This isolated v2 uses channel bitsets,
computes predecessor timestamps before applying the window, and materializes
only edges that can contribute to the fixed H and 4H features. It does not
change the signal, cutoff semantics, feature dimensions, or any defaults.
"""

import numpy as np
import pandas as pd

from scripts.activity_timeline_challenger import literal
from scripts.engagement_hypothesis_features import EventView


class BoundedChannelTransitions(EventView):
    def transform(self, requests, *, phase_boundary, horizon):
        times = self.request_times(requests, phase_boundary)
        days = pd.Timedelta(horizon).total_seconds() / 86400
        if not np.isfinite(days) or days <= 0:
            raise ValueError("Positive finite horizon required")
        output = []
        for cutoff in sorted(times.unique()):
            positions = np.flatnonzero(times.eq(cutoff).to_numpy())
            ids = requests.iloc[positions]["__event_entity"].astype(str).to_numpy()
            self.connection.register("transition_roots", pd.DataFrame({"entity": ids}).drop_duplicates())
            stamp = f"TIMESTAMP {literal(pd.Timestamp(cutoff))}"
            earliest = f"TIMESTAMP {literal(pd.Timestamp(cutoff) - pd.Timedelta(days=4*days))}"
            # Lag MUST precede the recent-edge filter: the predecessor of the
            # first recent event can be arbitrarily far in the past. Repeated
            # rows and tied channels at one instant form a set, not fake edges.
            self.connection.execute(f"""CREATE OR REPLACE TEMP TABLE transition_edges AS
                WITH instants AS (
                    SELECT entity, event_time, BIT_OR(1::UBIGINT << channel) mask
                    FROM event_rows JOIN transition_roots USING(entity)
                    WHERE event_time < {stamp}
                    GROUP BY entity, event_time
                ), ordered AS (
                    SELECT *, LAG(mask) OVER(PARTITION BY entity ORDER BY event_time) previous_mask,
                        LAG(event_time) OVER(PARTITION BY entity ORDER BY event_time) previous_time
                    FROM instants
                ) SELECT entity, mask, previous_mask,
                    1.0 / (BIT_COUNT(mask) * BIT_COUNT(previous_mask)) mass,
                    EPOCH(event_time - previous_time) / 86400.0 gap,
                    EPOCH({stamp} - event_time) / 86400.0 age
                FROM ordered WHERE previous_time IS NOT NULL AND event_time > {earliest}
            """)
            fields = []
            for w, window in enumerate((days, 4*days)):
                for previous in range(len(self.channels)):
                    for current in range(len(self.channels)):
                        condition = (f"(previous_mask & {1 << previous}) != 0 "
                            f"AND (mask & {1 << current}) != 0 AND age < {window!r}")
                        prefix = f"ct_w{w}_p{previous}_c{current}"
                        count = f"SUM(mass) FILTER(WHERE {condition})"
                        fields += [f"{count} {prefix}_count",
                            f"SUM(mass*gap) FILTER(WHERE {condition}) / NULLIF({count}, 0) {prefix}_gap"]
            block = (self.connection.execute(
                f"SELECT entity, {','.join(fields)} FROM transition_edges GROUP BY entity")
                .df().set_index("entity").reindex(ids).reset_index(drop=True))
            counts = [c for c in block if c.endswith("_count")]
            block[counts] = block[counts].fillna(0)
            block["__position"] = positions
            output.append(block)
        return (pd.concat(output).sort_values("__position").drop(columns="__position")
            .reset_index(drop=True).astype("float32"))
