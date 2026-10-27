"""Reuse a frozen cohort, rather than independently recomputing its aggregates.

This cache is scoped to the control/support materialization pair. Every source,
plan, label row, clock and provider must match before a cohort can be replayed.
It does not accept approximate numeric equality or reuse another task's cache.
"""

from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import inspect
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa

from kurversc import core
from kurversc.modeling import _jsonable, prepare_prediction_features
from scripts.run_gap_followup_queue import digest, dump


def _canonical_objects(frame):
    """Normalize only container storage and scalar missing-value sentinels."""
    result = frame.reset_index(drop=True).copy(deep=False)
    for name in result:
        if not pd.api.types.is_object_dtype(result[name].dtype):
            continue

        def canonical(value):
            if isinstance(value, (list, tuple, dict, set, frozenset, np.ndarray)):
                return json.dumps(_jsonable(value), sort_keys=True, default=str)
            return None if pd.isna(value) else value

        result[name] = pd.Series([canonical(v) for v in result[name]], dtype=object)
    return result


def assert_cache_equivalent(actual, reference, *, columns, categorical, datetime):
    """Exact raw values (modulo storage) AND exact prepared learner values."""
    pd.testing.assert_frame_equal(_canonical_objects(actual), _canonical_objects(reference),
        check_dtype=False, check_categorical=False, check_exact=True)
    prepared = [prepare_prediction_features(frame, columns=list(columns),
        categorical_columns=set(categorical), datetime_columns=set(datetime)).reset_index(drop=True)
        for frame in (actual, reference)]
    pd.testing.assert_frame_equal(*prepared, check_dtype=False,
        check_categorical=False, check_exact=True)


def _frame_digest(frame):
    """Bounded Arrow batches avoid hashing a second full wide-frame copy."""
    result = hashlib.sha256()
    result.update(json.dumps(dict(columns=list(frame.columns), rows=len(frame),
        dtypes=list(map(str, frame.dtypes)))).encode())
    for start in range(0, len(frame), 8192):
        table = pa.Table.from_pandas(frame.iloc[start:start+8192], preserve_index=False)
        with pa.BufferOutputStream() as output:
            with pa.ipc.new_stream(output, table.schema.remove_metadata()) as writer:
                writer.write_table(table.replace_schema_metadata(None))
            result.update(output.getvalue())
    return result.hexdigest()


class FrozenMaterializationCache:
    def __init__(self, output, *, columns, categorical, datetime):
        Path(output).mkdir(parents=True, exist_ok=True)
        self.directory = TemporaryDirectory(dir=output, prefix="canonical-control-")
        self.output = Path(self.directory.name)
        self.schema = dict(columns=columns, categorical=categorical, datetime=datetime)
        self.records = {}
        self.sources = {}
        self.hits = 0

    def close(self):
        self.directory.cleanup()
        self.sources.clear()

    def _source(self, source):
        if source is None:
            return None
        if isinstance(source, pd.DataFrame):
            key = id(source)
            if key not in self.sources:
                self.sources[key] = (source, _frame_digest(source))
            previous, sha = self.sources[key]
            if previous is not source:
                raise ValueError("Frozen source object identity changed")
            return dict(kind="frame", sha256=sha)
        path = Path(source)
        if not path.is_file():
            raise ValueError("Frozen replay requires explicit DataFrame or file sources, not mutable SQL views")
        path = path.resolve()
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        key = str(path)
        if key not in self.sources:
            self.sources[key] = (signature, digest(path))
        previous, sha = self.sources[key]
        if previous != signature:
            raise ValueError("Frozen materialization source changed during the fit")
        return dict(kind="file", sha256=sha)

    def _contract(self, arguments):
        def definition(value, excluded=()):
            if not is_dataclass(value):
                raise TypeError("Frozen materialization requires declarative table/label specifications")
            return {f.name: getattr(value, f.name) for f in fields(value) if f.name not in excluded}

        table_contract = {}
        for name, table in arguments["tables"].items():
            sources = {key: self._source(getattr(table, key))
                for key in ("source", "search_source", "fidelity_source")}
            table_contract[name] = {**definition(table, sources), **sources}
        spec = arguments["label_spec"]
        labels = definition(spec, ("source", "search_source", "fidelity_source", "history_source", "peer_context"))
        labels["peer_context_object"] = id(spec.peer_context) if spec.peer_context is not None else None
        contract = dict(tables=table_contract, relationships=arguments["relationships"],
            root=arguments["root_name"], config=arguments["config"], labels=labels,
            cohort_sha256=_frame_digest(arguments["labels"]),
            execution_plan=arguments["execution_plan"], cut_date=str(pd.Timestamp(arguments["cut_date"])),
            compute_period_days=arguments["compute_period_days"], train=arguments["train"],
            infer_ts_periods=arguments["infer_ts_periods"])
        return hashlib.sha256(json.dumps(contract, sort_keys=True, default=str).encode()).hexdigest()

    @contextmanager
    def install(self, *, replay=False):
        if replay:
            # DataFrames are mutable even inside frozen Table specifications.
            # Revalidate once at the phase boundary, not once per cohort.
            for key, (source, sha) in self.sources.items():
                if isinstance(key, int) and _frame_digest(source) != sha:
                    raise ValueError("Frozen DataFrame source changed between control and support fits")
        original = core._materialize_at_cutoff
        signature = inspect.signature(original)
        active = 0

        def materialize(*args, **kwargs):
            nonlocal active
            # Root batching recursively calls this hook. Cache complete cohorts,
            # never independently record or replay their component batches.
            if active:
                return original(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            if arguments["execution_plan"] is None or arguments["frozen_plan_sink"] is not None:
                raise ValueError("Frozen cohort reuse requires an already frozen plan with no learning side effects")
            contract = self._contract(arguments)
            cutoff = str(pd.Timestamp(arguments["cut_date"]))
            if replay and cutoff in self.records:
                record = self.records[cutoff]
                if record["contract"] != contract:
                    raise ValueError("An overlapping cohort changed sources, labels, graph or availability contract")
                if digest(record["path"]) != record["sha256"]:
                    raise ValueError("Canonical frozen cohort changed on disk")
                self.hits += 1
                return pd.read_parquet(record["path"])
            active += 1
            try:
                frame = original(*args, **kwargs)
            finally:
                active -= 1
            if not replay:
                if cutoff in self.records:
                    raise ValueError("Ambiguous duplicate control cohort")
                path = self.output / f"cohort-{len(self.records):04d}.parquet"
                if path.exists():
                    raise ValueError("A fresh control/support materialization pair is required")
                with NamedTemporaryFile(dir=self.output, suffix=".partial", delete=False) as handle:
                    temporary = Path(handle.name)
                frame.to_parquet(temporary, index=True)
                restored = pd.read_parquet(temporary)
                assert_cache_equivalent(restored, frame, **self.schema)
                temporary.replace(path)
                record = dict(path=str(path), sha256=digest(path), contract=contract)
                self.records[cutoff] = record
                dump(path.with_suffix(".capture.json"), record)
            return frame

        core._materialize_at_cutoff = materialize
        try:
            yield self
        finally:
            core._materialize_at_cutoff = original
