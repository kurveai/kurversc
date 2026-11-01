"""Explicit, benchmark-independent inputs for a staged prediction procedure.

This module has no RelBench/RelArena registry dependency and never owns held-out
answers. Source adapters supply schema and availability semantics; the context
creates independent phase-censored snapshots. Structural key canonicalization
is shared by tables, labels and requests, not inferred from benchmark names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pandas as pd


def key_parts(key):
    return (key,) if isinstance(key, str) else tuple(key)


def _clock(values):
    # DuckDB/GraphReduce use naive UTC timestamps, not the machine's timezone.
    return pd.to_datetime(values, utc=True, errors="raise").dt.tz_localize(None)


def _stamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("A complete phase boundary is required")
    return result.tz_convert("UTC").tz_localize(None) if result.tzinfo else result


@dataclass(frozen=True)
class SourceTable:
    frame: pd.DataFrame
    primary_key: str | tuple[str, ...] | None = None
    foreign_keys: Mapping[str | tuple[str, ...], str] = field(default_factory=dict)
    timestamp: str | None = None
    available_at: str | None = None
    timeless: bool = False
    cutoff_boundary: str = "strict"

    def __post_init__(self):
        if self.timestamp is None and self.available_at is None and not self.timeless:
            raise ValueError("Declare a source timestamp/availability clock or explicitly timeless data")
        if self.timeless and (self.timestamp or self.available_at):
            raise ValueError("Timeless data cannot also declare an availability clock")
        if self.cutoff_boundary not in {"strict","inclusive"}:
            raise ValueError("Declare strict or inclusive event-time semantics")
        required = set(key_parts(self.primary_key)) if self.primary_key else set()
        for key in self.foreign_keys:
            required.update(key_parts(key))
        required.update(c for c in (self.timestamp, self.available_at) if c)
        if required - set(self.frame):
            raise ValueError(f"Missing declared source columns: {sorted(required-set(self.frame))}")
        if self.primary_key:
            keys = list(key_parts(self.primary_key))
            if self.frame[keys].isna().any().any() or self.frame.duplicated(keys).any():
                raise ValueError("Primary keys must be complete and unique")
        if not self.frame.columns.is_unique:
            raise ValueError("Source column names must be unique")


@dataclass(frozen=True)
class PredictionTask:
    entity_table: str
    entity_key: str | tuple[str, ...]
    timestamp: str
    target: str
    horizon: str | pd.Timedelta
    kind: str = "classification"
    label_available_at: str | None = None
    name: str = "prediction"

    def __post_init__(self):
        if self.kind not in {"classification", "regression"}:
            raise ValueError("The complete staged recipe currently supports binary classification and regression")
        if pd.isna(pd.Timedelta(self.horizon)) or pd.Timedelta(self.horizon) <= pd.Timedelta(0):
            raise ValueError("A positive prediction/label horizon is required")
        if self.timestamp in {*key_parts(self.entity_key), self.target}:
            raise ValueError("Entity, timestamp and target roles must be distinct")


@dataclass
class ContextTable:
    df: pd.DataFrame
    pkey_col: str | None = None
    fkey_col_to_pkey_table: dict[str, str] = field(default_factory=dict)
    time_col: str | None = None
    available_col: str | None = None
    cutoff_boundary: str | None = "strict"


@dataclass
class ContextDatabase:
    table_dict: dict[str, ContextTable]
    exclude_structural_aliases: bool = True

    def upto(self, cutoff):
        boundary = _stamp(cutoff)
        tables = {}
        for name, table in self.table_dict.items():
            mask = pd.Series(True, index=table.df.index)
            if table.time_col:
                clock = _clock(table.df[table.time_col])
                mask &= clock.lt(boundary) if table.cutoff_boundary == "strict" else clock.le(boundary)
            if table.available_col:
                mask &= _clock(table.df[table.available_col]).le(boundary)
            tables[name] = ContextTable(table.df.loc[mask].copy(), table.pkey_col,
                dict(table.fkey_col_to_pkey_table), table.time_col, table.available_col,table.cutoff_boundary)
        result = ContextDatabase(tables, exclude_structural_aliases=self.exclude_structural_aliases)
        result.validate_and_correct_db(result)
        return result

    @staticmethod
    def validate_and_correct_db(db):
        # Keep original structural IDs. Filtering a parent must not renumber it
        # or make a now-dangling child point at a different entity.
        changed = True
        while changed:
            changed = False
            for table in db.table_dict.values():
                keep = pd.Series(True, index=table.df.index)
                for foreign, parent in table.fkey_col_to_pkey_table.items():
                    target = db.table_dict[parent]
                    keep &= table.df[foreign].isna() | table.df[foreign].isin(target.df[target.pkey_col])
                if not keep.all():
                    table.df = table.df.loc[keep].copy()
                    changed = True


def _key_tuples(frame, columns):
    return list(frame[list(key_parts(columns))].itertuples(index=False, name=None))


def _routing_id(parts):
    """Stable positive int64 routing, independent of source row order/future IDs."""
    def scalar(value):
        if isinstance(value,np.generic):
            value = value.item()
        if isinstance(value,pd.Timestamp):
            return ("timestamp",str(_stamp(value)))
        if isinstance(value,bytes):
            return ("bytes",value.hex())
        return (type(value).__name__,str(value))
    encoded = json.dumps([scalar(v) for v in parts],ensure_ascii=False,separators=(",",":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8],"big") & ((1<<63)-1)


def _canonical_tables(tables):
    maps, names = {}, {}
    physical = {name:f"table{i}" for i,name in enumerate(tables)}
    for name, spec in tables.items():
        if spec.primary_key is not None:
            keys = _key_tuples(spec.frame, spec.primary_key)
            # Preserve native dense integer keys when their identity is already
            # stable. UUID/composite surrogates must not shift when a future
            # entity arrives or an input query returns rows in another order.
            def identity(key):
                value = key[0]
                # This decision is per key, not based on the column maximum
                # or dtype: a future uint64 ID must not remap existing IDs.
                if (len(key)==1 and isinstance(value,(int,np.integer)) and not isinstance(value,(bool,np.bool_))
                        and 0<=int(value)<=np.iinfo("int64").max):
                    return int(value)
                return _routing_id(key)
            maps[name] = {key:identity(key) for key in keys}
            if len(set(maps[name].values()))!=len(keys):
                raise ValueError("Structural routing hash collision; use an explicit stable key adapter")
            names[name] = spec.primary_key if isinstance(spec.primary_key,str) else "__kurversc_primary_key"
    result = {}
    for name, spec in tables.items():
        frame = spec.frame.copy()
        structural = set(key_parts(spec.primary_key)) if spec.primary_key else set()
        replacements, foreign = {}, {}
        if spec.primary_key:
            replacements[names[name]] = pd.Series([maps[name][v] for v in _key_tuples(frame,spec.primary_key)],
                index=frame.index,dtype="int64")
        for number, (key, parent) in enumerate(spec.foreign_keys.items()):
            if parent not in maps:
                raise ValueError(f"Foreign key references a missing primary-key table: {parent}")
            parts = key_parts(key)
            if len(parts) != len(key_parts(tables[parent].primary_key)):
                raise ValueError("Foreign and primary key arity differ")
            structural.update(parts)
            column = key if isinstance(key,str) else f"__kurversc_foreign_key_{number}"
            values = []
            for value in _key_tuples(frame,key):
                if any(pd.isna(part) for part in value):
                    values.append(pd.NA)
                elif value not in maps[parent]:
                    raise ValueError(f"Foreign key in {name} references an unknown {parent} entity")
                else:
                    values.append(maps[parent][value])
            replacement = pd.Series(values,index=frame.index,dtype="Int64")
            if column in replacements and not replacement.equals(replacements[column].astype("Int64")):
                raise ValueError("A primary/foreign key column has conflicting structural meanings")
            replacements[column], foreign[column] = replacement, physical[parent]
        generated = set(replacements)-structural
        if generated & set(frame):
            raise ValueError("Generated structural key collides with a source payload column")
        frame = frame.drop(columns=list(structural))
        for column, values in replacements.items():
            frame[column] = values
        for column in dict.fromkeys(c for c in (spec.timestamp,spec.available_at) if c):
            frame[column] = _clock(frame[column])
        ordering = [names[name]] if spec.primary_key else [*([spec.timestamp] if spec.timestamp else []),*foreign]
        if ordering:
            frame = frame.sort_values(ordering,kind="stable").reset_index(drop=True)
        result[physical[name]] = ContextTable(frame,names.get(name),foreign,spec.timestamp,
            spec.available_at or spec.timestamp,spec.cutoff_boundary)
    return ContextDatabase(result), maps, physical


@dataclass
class TaskContext:
    """A task definition and authorized inputs, never a held-out target table.

    The original request keys are retained for output alignment. Integer
    surrogates are routing state only, shared across every phase. Mutable/SCD
    source snapshots need an adapter with correct historical availability;
    declaring a current mutable table timeless does not make it leak-safe.
    """
    task: PredictionTask
    database: ContextDatabase
    training: pd.DataFrame
    validation: pd.DataFrame
    requests: pd.DataFrame
    request_keys: pd.DataFrame
    inner_cutoff: pd.Timestamp
    outer_cutoff: pd.Timestamp
    dataset_name: str = "database"
    preprocessing_seconds: float = 0.
    schema_aliases: dict[str,str] = field(default_factory=dict)

    @classmethod
    def from_protocol_splits(cls, *, task, inner_split, outer_split, dataset_name="database"):
        """Preserve already-censored protocol snapshots without renaming inputs.

        ``task`` is a PredictionTask; splits expose database/observed-label
        tables and target-free forecast requests. This compatibility boundary
        accepts no registry or held-out answer table. For raw databases, use
        from_tables, which also canonicalizes arbitrary/composite keys.
        """
        from .protocol_context import ProtocolTaskContext
        return ProtocolTaskContext(task=task, inner_split=inner_split,
            outer_split=outer_split, dataset_name=dataset_name)

    @classmethod
    def from_tables(cls, *, tables: Mapping[str,SourceTable], task: PredictionTask,
                    training: pd.DataFrame, validation: pd.DataFrame, requests: pd.DataFrame,
                    inner_cutoff, outer_cutoff, dataset_name="database"):
        started = time.monotonic()
        if task.target in requests:
            raise ValueError("Forecast requests must not contain held-out targets, even an all-null target column")
        if task.entity_table not in tables or tables[task.entity_table].primary_key is None:
            raise ValueError("The declared predictive entity requires a primary-key table")
        if len(key_parts(task.entity_key)) != len(key_parts(tables[task.entity_table].primary_key)):
            raise ValueError("Predictive entity key arity differs from the declared root")
        db, maps, physical = _canonical_tables(tables)
        entity = task.entity_key if isinstance(task.entity_key,str) else "__kurversc_entity_key"
        if entity == task.target or task.target in key_parts(task.entity_key):
            raise ValueError("Structural keys cannot be targets")
        originals = requests[[*key_parts(task.entity_key),task.timestamp]].copy().reset_index(drop=True)
        converted = []
        for labeled, frame in ((True,training),(True,validation),(False,requests)):
            required = [*key_parts(task.entity_key),task.timestamp,*([task.target] if labeled else []),
                *([task.label_available_at] if labeled and task.label_available_at else [])]
            if set(required)-set(frame) or frame[required].isna().any().any():
                raise ValueError("Complete declared entity/time/observed-target columns required")
            ids = []
            for value in _key_tuples(frame,task.entity_key):
                if value not in maps[task.entity_table]:
                    raise ValueError("Unknown prediction entity")
                ids.append(maps[task.entity_table][value])
            payload = pd.DataFrame({entity:ids,task.timestamp:_clock(frame[task.timestamp]).to_numpy()})
            if labeled:
                payload[task.target] = frame[task.target].to_numpy()
                available = (_clock(frame[task.label_available_at]).to_numpy() if task.label_available_at
                    else payload[task.timestamp]+pd.Timedelta(task.horizon))
                payload["__kurversc_label_available_at"] = available
                if (payload["__kurversc_label_available_at"] < payload[task.timestamp]+pd.Timedelta(task.horizon)).any():
                    raise ValueError("A forward-window target cannot be available before its horizon completes")
                if task.kind=="classification" and not payload[task.target].isin([0,1]).all():
                    raise ValueError("Binary targets must use declared 0/1 outcomes")
                if task.kind=="regression":
                    payload[task.target] = pd.to_numeric(payload[task.target],errors="raise")
                    if not np.isfinite(payload[task.target]).all():
                        raise ValueError("Observed regression targets must be finite")
            if payload.duplicated([entity,task.timestamp]).any():
                raise ValueError("Entity/time examples must be unique within a split")
            if labeled:
                payload = payload.sort_values([task.timestamp,entity],kind="stable").reset_index(drop=True)
            converted.append(payload)
        train,val,query = converted
        inner,outer = _stamp(inner_cutoff),_stamp(outer_cutoff)
        if inner>=outer or train.empty or val.empty or query.empty:
            raise ValueError("Nonempty train/validation/forecast phases and increasing boundaries required")
        if ((train["__kurversc_label_available_at"]>inner).any()
                or (val[task.timestamp]<inner).any()
                or (val["__kurversc_label_available_at"]>outer).any()
                or (query[task.timestamp]<outer).any()):
            raise ValueError("Observed labels or forecast rows violate the authorized phase boundaries")
        if train[task.timestamp].max()>=val[task.timestamp].min() or val[task.timestamp].max()>=query[task.timestamp].min():
            raise ValueError("Train, validation and forecast cohorts must be chronologically disjoint")
        definition = PredictionTask(physical[task.entity_table],entity,task.timestamp,task.target,task.horizon,
            task.kind,"__kurversc_label_available_at",task.name)
        return cls(definition,db,train,val,query,originals,inner,outer,dataset_name,time.monotonic()-started,
            {physical[name]:name for name in tables})

    def observed_before(self, frame, cutoff):
        available = self.task.label_available_at
        return frame.loc[_clock(frame[available]).le(_stamp(cutoff))].copy()

    def split(self, phase):
        if phase not in {"inner","outer"}:
            raise ValueError("Unknown prediction phase")
        cutoff = self.inner_cutoff if phase=="inner" else self.outer_cutoff
        db = self.database.upto(cutoff)
        root = db.table_dict[self.task.entity_table]
        def table(frame, *, maturity=False):
            frame = self.observed_before(frame,cutoff) if maturity else frame.copy()
            frame = frame.loc[frame[self.task.entity_key].isin(root.df[root.pkey_col])].reset_index(drop=True)
            return ContextTable(frame,None,{self.task.entity_key:self.task.entity_table},self.task.timestamp)
        result = dict(db_state=db,cutoff=cutoff,train_table=table(self.training,maturity=True))
        if phase=="inner":
            result.update(eval_table=table(self.validation),eval_target=table(self.validation))
        else:
            result.update(eval_table=table(self.requests),val_table=table(self.validation,maturity=True))
            if len(result["eval_table"].df)!=len(self.requests):
                raise ValueError("A forecast entity is unavailable at the supplied database boundary")
        return SimpleNamespace(**result)

    def export(self, directory):
        """Create a checksummed, label-masked bundle for isolated stage workers."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        started = time.monotonic()
        root = Path(directory).resolve()
        root.mkdir(parents=True,exist_ok=True)
        if (root/"context.json").exists():
            raise FileExistsError("Never overwrite an existing immutable task context")
        files = {}
        def write(path, table):
            destination = root/path
            destination.parent.mkdir(parents=True,exist_ok=True)
            arrow = pa.Table.from_pandas(table.df,preserve_index=False)
            schema = dict(pkey_col=table.pkey_col,time_col=table.time_col,
                fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,available_col=table.available_col,
                cutoff_boundary=table.cutoff_boundary)
            pq.write_table(arrow.replace_schema_metadata({k.encode():json.dumps(v).encode() for k,v in schema.items()}),destination)
            files[str(path)] = _file_digest(destination)
        for phase in ("inner","outer"):
            split = self.split(phase)
            for i,(name,table) in enumerate(split.db_state.table_dict.items()):
                # Names live in metadata, not paths: arbitrary schema names are safe.
                write(Path(phase)/"db"/f"table{i}.parquet",table)
            for role in ("train_table","eval_table",*(["val_table"] if phase=="outer" else [])):
                write(Path(phase)/(role+".parquet"),getattr(split,role))
        write(Path("request-keys.parquet"),ContextTable(self.request_keys))
        manifest = dict(version=1,dataset=self.dataset_name,task=self.task.name,
            definition={**self.task.__dict__,"horizon":str(pd.Timedelta(self.task.horizon))},
            tables=list(self.database.table_dict),inner_cutoff=str(self.inner_cutoff),outer_cutoff=str(self.outer_cutoff),
            files=files,heldout_targets_present=False)
        manifest["schema_aliases"] = self.schema_aliases
        manifest["preprocessing_seconds"] = self.preprocessing_seconds+time.monotonic()-started
        (root/"context.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
        return root/"context.json"


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()
