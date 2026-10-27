"""Portable task-bundle adapter used by every stage of the complete pipeline.

The no-bundle branch is the unchanged RelArena adapter. Explicit bundles are
never resolved through a benchmark registry, even when their names happen to
match a benchmark. The bundle has no held-out answers; production prediction
stages therefore cannot report a test score for it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .task_context import ContextDatabase, ContextTable, PredictionTask, _clock, _file_digest, _stamp


CONTEXT_ENV = "KURVERSC_TASK_CONTEXT"


@dataclass(frozen=True)
class ContextIdentity:
    dataset: str
    dataset_fingerprint: str
    task: str
    task_fingerprint: str
    phase: str | None = None
    data_version: str | None = None

    def for_phase(self, phase):
        return replace(self,phase=phase)


class ContextTask:
    def __init__(self, definition, source):
        self.definition, self.source = definition, source
        self.entity_table, self.entity_col = definition.entity_table, definition.entity_key
        self.time_col, self.target_col = definition.timestamp, definition.target
        self.timedelta, self.label_available_at = pd.Timedelta(definition.horizon), definition.label_available_at
        self.task_type = "binary_classification" if definition.kind=="classification" else "regression"
        # The compatibility adapter can expose the public task enum without
        # making the data contract depend on RelBench or its task registry.
        try:
            from relbench.base import TaskType
            self.task_type = TaskType.BINARY_CLASSIFICATION if definition.kind=="classification" else TaskType.REGRESSION
        except ImportError:
            pass
        self.metrics = [source.metric]
        self.has_hidden_targets = False

    def evaluate(self, prediction, target_table=None, metrics=None):
        if target_table is None:
            # Existing stage report schemas carry a score field. NaN denotes
            # explicitly unscored intermediate reports, never a benchmark row.
            # The public portable result exposes predictions, not this field.
            return {metric.__name__:float("nan") for metric in (metrics or self.metrics)}
        target = target_table.df[self.target_col].to_numpy()
        return {metric.__name__:float(metric(target,prediction)) for metric in (metrics or self.metrics)}


def roc_auc(target,prediction):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(target,prediction)


def mae(target,prediction):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(target,prediction)


class ContextTaskSource:
    def __init__(self, path, *, verify=True):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        self.manifest = json.loads(self.path.read_text())
        if (self.manifest.get("version") not in {1, 2} or self.manifest.get("heldout_targets_present") is not False
                or self.manifest.get("version") == 2 and self.manifest.get("protocol_snapshots_preserved") is not True):
            raise ValueError("A supported, heldout-target-free task context is required")
        self.identity_digest = _file_digest(self.path)
        if verify:
            for name,wanted in self.manifest["files"].items():
                file = (self.root/name).resolve()
                if not file.is_relative_to(self.root) or _file_digest(file)!=wanted:
                    raise ValueError(f"Task-context input changed or escaped its bundle: {name}")
        definition = PredictionTask(**self.manifest["definition"])
        self.metric = roc_auc if definition.kind=="classification" else mae
        self.task = ContextTask(definition,self)
        self._dataset = ContextDatabase({})

    def _table(self, path):
        import pyarrow.parquet as pq
        path = self.root/path
        metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
        fields = {name:json.loads(metadata.get(name.encode(),b"null"))
            for name in ("pkey_col","fkey_col_to_pkey_table","time_col","available_col","cutoff_boundary")}
        boundary = fields["cutoff_boundary"] if self.manifest["version"] == 2 else fields["cutoff_boundary"] or "strict"
        frame = pd.read_parquet(path)
        if self.manifest["version"] == 2:
            # Parquet promotes second-resolution timestamps to milliseconds;
            # restore the caller's exact original protocol input dtypes.
            dtypes = json.loads(metadata.get(b"protocol_datetime_dtypes", b"{}"))
            frame = frame.astype(dtypes)
        return ContextTable(frame,fields["pkey_col"],fields["fkey_col_to_pkey_table"] or {},
            fields["time_col"],fields["available_col"],boundary)

    def _split(self, phase):
        cutoff = _stamp(self.manifest[phase+"_cutoff"])
        db = ContextDatabase({name:self._table(Path(phase)/"db"/f"table{i}.parquet")
            for i,name in enumerate(self.manifest["tables"])},
            exclude_structural_aliases=self.manifest.get("exclude_structural_aliases", True))
        # Reapply the boundary defensively, even though exports are censored.
        db = db.upto(cutoff)
        values = dict(db_state=db,cutoff=cutoff,
            train_table=self._table(Path(phase)/"train_table.parquet"),
            eval_table=self._table(Path(phase)/"eval_table.parquet"))
        if phase=="inner":
            values["eval_target"] = values["eval_table"]
        else:
            values["val_table"] = self._table(Path(phase)/"val_table.parquet")
            if self.task.target_col in values["eval_table"].df:
                raise ValueError("Held-out targets reached a portable forecast split")
        return SimpleNamespace(**values)

    def inner_split(self):
        return self._split("inner")

    def outer_split(self):
        return self._split("outer")

    def run_identity(self, phase=None):
        return ContextIdentity(self.manifest["dataset"],self.identity_digest,
            self.manifest["task"],self.identity_digest,phase)

    def mature_labels(self, frame, cutoff):
        clock = self.task.label_available_at
        available = _clock(frame[clock]) if clock else _clock(frame[self.task.time_col])+self.task.timedelta
        return frame.loc[available.le(_stamp(cutoff))].copy()


def load_task_source(dataset, task, *, context=None, **kwargs):
    """Resolve an explicit source or use the benchmark adapter at its boundary."""
    path = context or os.environ.get(CONTEXT_ENV)
    if path:
        source = ContextTaskSource(path)
        if (dataset,task)!=(source.manifest["dataset"],source.manifest["task"]):
            raise ValueError("Task name does not match the explicitly supplied context")
        return source
    from relarena.dataset import RelBenchDatasetTask
    return RelBenchDatasetTask(dataset,task,**kwargs)


def run_context_system(system_cls,dataset_name,task_name,*,seed=0,cache_dir=None,
                       cache_predictions=True,time_limit=None,**kwargs):
    """RelArena-shaped adapter call with no benchmark loader or test scoring."""
    from .staged_adapter import resolve_cache_config
    import time

    source = load_task_source(dataset_name,task_name)
    if not isinstance(source,ContextTaskSource):
        raise ValueError("An explicit task context is required")
    inner,outer = source.inner_split(),source.outer_split()
    started = time.monotonic()
    system = system_cls(cache=resolve_cache_config(cache_dir,on_miss="raise"),run_identity=source.run_identity())
    prediction = np.asarray(system.run(source.task,inner_split=inner,outer_split=outer,
        seed=seed,time_limit=time_limit))
    if prediction.shape!=(len(outer.eval_table.df),) or not np.isfinite(prediction).all():
        raise ValueError("Invalid portable forecast output")
    return SimpleNamespace(system_name=system_cls.name,dataset=dataset_name,
        task_name=task_name,task_type=source.task.task_type,metric_name=source.metric.__name__,seed=seed,
        result=SimpleNamespace(test_score=None,test_metrics={},time_total=time.monotonic()-started,
            test_pred=prediction if cache_predictions else None))


def phase_source_root(dataset, *, phase="inner", directory=None):
    """Export authorized source relations for legacy file-backed providers.

    An explicit portable bundle is authoritative; there is no cache/registry
    fallback. Source files for inner and outer readouts live separately.
    """
    context = os.environ.get(CONTEXT_ENV)
    if not context:
        from relbench.utils import get_relbench_cache_dir
        return Path(get_relbench_cache_dir())/dataset
    source = ContextTaskSource(context)
    if dataset!=source.manifest["dataset"] or phase not in {"inner","outer"}:
        raise ValueError("Unexpected task source phase")
    destination = Path(directory) if directory else source.root/"provider-views"/phase
    destination.mkdir(parents=True,exist_ok=True)
    # Providers often run in separate processes. A directory's existence is
    # not evidence that its phase snapshot is complete or still unmodified.
    import fcntl
    import pyarrow as pa
    import pyarrow.parquet as pq
    split = source.inner_split() if phase=="inner" else source.outer_split()
    files = {}
    def write(path,table):
        path.parent.mkdir(parents=True,exist_ok=True)
        arrow = pa.Table.from_pandas(table.df,preserve_index=False)
        schema = dict(pkey_col=table.pkey_col,time_col=table.time_col,
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
            available_at=table.available_col,label_available_at=source.task.label_available_at,
            cutoff_boundary=table.cutoff_boundary)
        arrow = arrow.replace_schema_metadata({k.encode():json.dumps(v).encode() for k,v in schema.items()})
        pq.write_table(arrow,path)
        files[str(path.relative_to(destination))] = _file_digest(path)
    with (destination/".export.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest = destination/"provider-context.json"
        if manifest.exists():
            existing = json.loads(manifest.read_text())
            if existing.get("context_sha256")!=source.identity_digest or existing.get("phase")!=phase:
                raise ValueError("Provider snapshot belongs to a different task context/phase")
            for name,wanted in existing["files"].items():
                file = (destination/name).resolve()
                if not file.is_relative_to(destination.resolve()) or _file_digest(file)!=wanted:
                    raise ValueError(f"Authorized provider snapshot changed: {name}")
            return destination
        if list(destination.glob("**/*.parquet")):
            raise ValueError("Incomplete provider snapshot; choose a new export directory")
        for name,table in split.db_state.table_dict.items():
            write(destination/"db"/(name+".parquet"),table)
        # Inner providers may score observed validation outcomes, but every
        # feature path still enforces per-origin label availability.
        train = split.train_table
        val = split.eval_table if phase=="inner" else split.val_table
        write(destination/"tasks"/source.manifest["task"]/"train.parquet",train)
        write(destination/"tasks"/source.manifest["task"]/"val.parquet",val)
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(context_sha256=source.identity_digest,phase=phase,files=files),sort_keys=True)+"\n")
        temporary.replace(manifest)
    return destination
