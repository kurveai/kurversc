"""Schema-discovered adapter for the validated exposure-quality prototype.

No dataset/task/table/column-name lookup. Only declared foreign keys, unique
parent keys, timestamps and ordinary binary event responses are eligible.
Missing paths are recorded as inapplicable. Predictive label tables are never
supplied to discovery or the feature provider.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from kurversc.signal_challengers import ChallengerUnavailable


@dataclass(frozen=True)
class ExposurePath:
    parent: str
    parent_root_fk: str
    event: str
    event_parent_fk: str
    object_table: str
    object_fk: str
    response: str
    context: str | None = None
    prior: str | None = None


def payload_columns(table):
    excluded = {table.pkey_col, table.time_col, *table.fkey_col_to_pkey_table}
    return [name for name in table.df if name not in excluded]


def numeric_roles(table):
    binary, context, prior = [], [], []
    for name in payload_columns(table):
        values = table.df[name]
        if not pd.api.types.is_numeric_dtype(values.dtype):
            continue
        # Bounded deterministic schema profiling, never target-driven ranking.
        if len(values) > 20000:
            values = values.iloc[np.linspace(0, len(values) - 1, 20000, dtype=int)]
        values = pd.to_numeric(values, errors="coerce").dropna()
        unique = values.unique()
        if len(unique) == 2 and set(unique) == {0, 1}:
            binary.append(name)
        elif 2 <= len(unique) <= 32:
            context.append((len(unique), name))
        elif len(unique) > 32 and values.between(0, 1).all():
            prior.append(name)
    # A richer low-cardinality context wins ties over constant/binary flags.
    return sorted(binary), [n for _, n in sorted(context, reverse=True)], sorted(prior)


def discover_exposure_path(tables, root):
    paths = []
    for parent_name, parent in sorted(tables.items()):
        if not parent.pkey_col or not parent.time_col:
            continue
        if parent.df[parent.pkey_col].duplicated().any():
            continue
        for root_fk, root_table in sorted(parent.fkey_col_to_pkey_table.items()):
            if root_table != root:
                continue
            for event_name, event in sorted(tables.items()):
                if not event.time_col or event_name in {root, parent_name}:
                    continue
                responses, contexts, priors = numeric_roles(event)
                if not responses:
                    continue
                for parent_fk, referenced in sorted(event.fkey_col_to_pkey_table.items()):
                    if referenced != parent_name:
                        continue
                    for object_fk, object_name in sorted(event.fkey_col_to_pkey_table.items()):
                        if object_name in {root, parent_name, event_name}:
                            continue
                        dimension = tables[object_name]
                        if not dimension.pkey_col or dimension.df[dimension.pkey_col].duplicated().any():
                            continue
                        paths.append(ExposurePath(parent_name, root_fk, event_name,
                            parent_fk, object_name, object_fk, responses[0],
                            contexts[0] if contexts else None, priors[0] if priors else None))
    # One path, one response, one context, one prior: no graph-space expansion.
    return sorted(paths, key=lambda p: tuple(str(v) for v in asdict(p).values()))[0] if paths else None


class SchemaExposureQuality:
    def __init__(self, tables, root, *, horizon, threads=4, temp_directory=None,
                 path=None, memory_limit="32GB"):
        from scripts.avito_exposure_signals import ExposureAdjustedSignals
        self.path = path or discover_exposure_path(tables, root)
        if self.path is None:
            raise ChallengerUnavailable("No timestamped two-hop exposure/object path with a binary event response")
        p = self.path
        parent, events, objects = tables[p.parent], tables[p.event], tables[p.object_table]
        parents = parent.df[[parent.pkey_col, p.parent_root_fk, parent.time_col]].copy()
        parents.columns = ["SearchID", "UserID", "SearchDate"]
        data = events.df[[p.event_parent_fk, p.object_fk, events.time_col, p.response]].copy()
        data.columns = ["SearchID", "AdID", "SearchDate", "IsClick"]
        data["Position"] = events.df[p.context].to_numpy() if p.context else "__none__"
        data["HistCTR"] = events.df[p.prior].to_numpy() if p.prior else np.nan
        dimensions = objects.df[[objects.pkey_col]].rename(columns={objects.pkey_col: "AdID"})
        for original,canonical in ((parent,parents),(events,data),(objects,dimensions)):
            availability = getattr(original,"available_col",None)
            if availability:
                canonical["AvailableAt"] = pd.to_datetime(original.df[availability]).to_numpy()
        self.provider = ExposureAdjustedSignals(parents, data, dimensions, horizon=str(horizon),
            threads=threads, memory_limit=memory_limit, temp_directory=temp_directory)
        self.last_audit = {}

    def transform(self, frame, *, phase_boundary):
        parts, audits = [], []
        for cutoff, rows in frame.groupby("__event_cutoff", sort=True):
            result = self.provider.transform(rows["__event_entity"],
                min(pd.Timestamp(cutoff), pd.Timestamp(phase_boundary)))
            extra = pd.concat([result["activity"], result["quality"]], axis=1)
            extra.index = rows.index
            parts.append(extra)
            audits.append({**self.provider.last_audit, "schema_path": asdict(self.path)})
        self.last_audit = dict(schema_path=asdict(self.path), windows=4, features=68,
            phases=audits, task_labels_used=False)
        if not parts:
            raise ChallengerUnavailable("No exposure query rows")
        return pd.concat(parts).loc[frame.index]

    def close(self):
        self.provider.close()
