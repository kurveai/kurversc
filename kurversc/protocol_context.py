"""Lossless handoff of caller-censored protocol snapshots to staged workers.

The adapter preserves table names, source order, declared key roles and legacy
clock metadata. It does not recanonicalize an existing benchmark input or
reload a registry. Version-1 raw-database contexts retain their strict public
semantics; this explicitly marked version-2 bundle preserves supplied metadata.
"""

import json
from pathlib import Path
import time

import pandas as pd

from .task_context import ContextTable, TaskContext, _clock, _file_digest, _stamp


class ProtocolTaskContext(TaskContext):
    def __init__(self, *, task, inner_split, outer_split, dataset_name):
        began = time.monotonic()
        if not isinstance(task.entity_key, str):
            raise ValueError("Protocol keys must already be scalar/aligned; use from_tables for composite keys")
        self.task, self.dataset_name = task, dataset_name
        self.inner_cutoff, self.outer_cutoff = _stamp(inner_split.cutoff), _stamp(outer_split.cutoff)
        if self.inner_cutoff >= self.outer_cutoff:
            raise ValueError("Increasing caller-owned protocol boundaries required")
        self._splits = dict(inner=inner_split, outer=outer_split)
        names = list(inner_split.db_state.table_dict)
        if names != list(outer_split.db_state.table_dict) or task.entity_table not in names:
            raise ValueError("Aligned source relation names/order and declared root required across phases")
        if any(not name or Path(name).name != name or name in {".", ".."} for name in names):
            raise ValueError("Protocol names must be simple path components; use from_tables for arbitrary schema names")
        self.database = outer_split.db_state
        self.training, self.validation = inner_split.train_table.df, inner_split.eval_table.df
        self.requests = outer_split.eval_table.df
        if task.target in self.requests:
            raise ValueError("Protocol forecast requests must contain no held-out target column")
        self.request_keys = self.requests[[task.entity_key, task.timestamp]].copy().reset_index(drop=True)
        self.schema_aliases = {}
        for phase, split in self._splits.items():
            cutoff = self.inner_cutoff if phase == "inner" else self.outer_cutoff
            for table in split.db_state.table_dict.values():
                for name in dict.fromkeys(c for c in (table.time_col, getattr(table, "available_col", None)) if c):
                    if _clock(table.df[name]).gt(cutoff).any():
                        raise ValueError("Protocol database contains source rows beyond its supplied boundary")
            observed = (split.train_table, split.eval_table) if phase == "inner" else (split.train_table, split.val_table)
            for table in observed:
                fields = [task.entity_key, task.timestamp, task.target]
                if task.label_available_at:
                    fields.append(task.label_available_at)
                frame = table.df
                if set(fields)-set(frame) or frame[fields].isna().any().any():
                    raise ValueError("Complete observed protocol entity/time/target clocks required")
                available = (_clock(frame[task.label_available_at]) if task.label_available_at
                    else _clock(frame[task.timestamp])+pd.Timedelta(task.horizon))
                # Inner eval labels are observed selection answers at the outer
                # boundary, not materialization inputs at the inner boundary.
                ceiling = self.outer_cutoff if phase == "inner" and table is split.eval_table else cutoff
                if available.gt(ceiling).any():
                    raise ValueError("An observed protocol target is not mature by its authorized phase")
        self.preprocessing_seconds = time.monotonic()-began

    def split(self, phase):
        if phase not in self._splits:
            raise ValueError("Unknown protocol phase")
        return self._splits[phase]

    def export(self, directory):
        import pyarrow as pa
        import pyarrow.parquet as pq

        began = time.monotonic()
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if (root / "context.json").exists():
            raise FileExistsError("Never overwrite an immutable protocol bundle")
        files = {}
        def write(relative, table):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            fields = dict(pkey_col=table.pkey_col, time_col=table.time_col,
                fkey_col_to_pkey_table=table.fkey_col_to_pkey_table or {},
                available_col=getattr(table, "available_col", None),
                cutoff_boundary=getattr(table, "cutoff_boundary", None),
                protocol_datetime_dtypes={name: str(table.df[name].dtype) for name in table.df
                    if pd.api.types.is_datetime64_any_dtype(table.df[name].dtype)})
            data = pa.Table.from_pandas(table.df, preserve_index=False)
            metadata = {**(data.schema.metadata or {}), **{k.encode(): json.dumps(v).encode() for k, v in fields.items()}}
            pq.write_table(data.replace_schema_metadata(metadata), destination)
            files[str(relative)] = _file_digest(destination)
        names = list(self._splits["inner"].db_state.table_dict)
        for phase, split in self._splits.items():
            for i, name in enumerate(names):
                write(Path(phase) / "db" / f"table{i}.parquet", split.db_state.table_dict[name])
            for role in ("train_table", "eval_table", *(["val_table"] if phase == "outer" else [])):
                write(Path(phase) / (role+".parquet"), getattr(split, role))
        write(Path("request-keys.parquet"), ContextTable(self.request_keys))
        manifest = dict(version=2, protocol_snapshots_preserved=True, dataset=self.dataset_name,
            task=self.task.name, definition={**self.task.__dict__, "horizon": str(pd.Timedelta(self.task.horizon))},
            tables=names, inner_cutoff=str(self.inner_cutoff), outer_cutoff=str(self.outer_cutoff),
            files=files, heldout_targets_present=False, exclude_structural_aliases=False,
            preprocessing_seconds=self.preprocessing_seconds+time.monotonic()-began)
        (root / "context.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
        return root / "context.json"
