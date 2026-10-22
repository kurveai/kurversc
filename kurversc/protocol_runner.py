"""Thin harness bridge to the same packaged, target-unseen staged procedure.

The caller owns protocol splitting and final scoring. This module never loads a
task registry, chooses a graph by name, or receives hidden target values.
"""

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import tempfile

from .staged import run_staged_task
from .staged_runtime import staged_source_identity
from .task_context import PredictionTask, TaskContext


def run_protocol_task(task, *, inner_split, outer_split, dataset_name,
                      task_name="prediction", seed=0, time_limit=None,
                      cache_directory=None, run_identity=None):
    if seed != 0:
        raise ValueError("The frozen complete staged recipe uses deployment seed 0; other seeds are not silently ignored")
    kind = str(task.task_type).lower()
    if "binary" in kind or kind == "classification":
        kind = "classification"
    elif "regression" in kind:
        kind = "regression"
    else:
        raise ValueError("The complete staged procedure supports binary classification and regression")
    hours = 23.75 if time_limit is None else min(23.75, float(time_limit)/3600)
    if hours <= 0:
        raise TimeoutError("The supplied complete-procedure time budget is exhausted")
    definition = PredictionTask(task.entity_table, task.entity_col, task.time_col, task.target_col,
        task.timedelta, kind=kind, label_available_at=getattr(task, "label_available_at", None), name=task_name)
    context = TaskContext.from_protocol_splits(task=definition, inner_split=inner_split,
        outer_split=outer_split, dataset_name=dataset_name)
    identity = asdict(run_identity) if is_dataclass(run_identity) else None
    if (cache_directory is not None and identity and identity.get("dataset_fingerprint")
            and identity.get("task_fingerprint")):
        # Persistent reuse follows the caller's explicit content identity. Raw
        # database callers must version changed data, as with other caches.
        binding = dict(source=identity, definition={**asdict(definition), "horizon": str(definition.horizon)},
            code=staged_source_identity(Path(__file__).resolve().parents[1]))
        key = hashlib.sha256(json.dumps(binding, sort_keys=True, default=str).encode()).hexdigest()
        output = Path(cache_directory) / "kurversc-complete" / key
        output.mkdir(parents=True, exist_ok=True)
        ledger = output / "protocol-binding.json"
        if ledger.exists() and json.loads(ledger.read_text()) != binding:
            raise ValueError("Protocol source identity changed; refusing mixed-input resume")
        if not ledger.exists():
            ledger.write_text(json.dumps(binding, sort_keys=True, default=str)+"\n")
    else:
        output = Path(tempfile.mkdtemp(prefix="kurversc-complete-"))
    bundle = output / "input-context/context.json"
    if not bundle.exists():
        bundle = context.export(bundle.parent)
    return run_staged_task(bundle, output=output, task_hours=hours,
        minimum_available_memory_gib=8.)
