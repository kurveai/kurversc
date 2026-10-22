"""Disk-backed capture of unchanged full GraphReduce feature frames.

The frame returned to the ordinary fitter is untouched. Only the diagnostic
capture's extra copies move to disk; concatenation is performed in bounded
Arrow batches rather than retaining every historical full-width DataFrame.
"""

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import kurversc.core as core
from scripts.run_gap_followup_queue import digest, dump


class FrameSink:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.parts = []

    def append(self, frame, cutoff):
        path = self.output / f"frame{len(self.parts):04d}.parquet"
        proof_path = path.with_suffix(path.suffix+".capture.json")
        identity = dict(cutoff=str(pd.Timestamp(cutoff)), rows=len(frame))
        if path.exists():
            # An interrupted driver may revisit a completed cohort. Preserve
            # it only after full value/order equivalence, never overwrite it.
            if (not proof_path.exists() or json.loads(proof_path.read_text()) !=
                    dict(identity, output_sha256=digest(path))):
                raise ValueError("Existing frame capture identity is missing or changed")
            pd.testing.assert_frame_equal(pd.read_parquet(path), frame,
                check_dtype=False, check_categorical=False, check_exact=True)
        else:
            frame.to_parquet(path, index=False)
            dump(proof_path, dict(identity, output_sha256=digest(path)))
        self.parts.append(dict(path=path, cutoff=str(pd.Timestamp(cutoff)), rows=len(frame)))

    def concatenate(self, destination, *, cutoffs, batch_rows=2048):
        if batch_rows < 1:
            raise ValueError("Positive bounded Arrow batch size required")
        wanted = set(map(pd.Timestamp, cutoffs))
        selected = [r for r in self.parts if pd.Timestamp(r["cutoff"]) in wanted]
        if not selected:
            raise ValueError("At least one captured complete cohort is required")
        schemas = [pq.ParquetFile(r["path"]).schema_arrow.remove_metadata() for r in selected]
        if any(s.names != schemas[0].names for s in schemas):
            raise ValueError("Frozen ordered cache columns changed across cohorts")
        # Normalize dictionary storage, not values. pandas concat can promote
        # categoricals with different per-cohort dictionaries to ordinary text.
        schemas = [pa.schema([pa.field(f.name,
            f.type.value_type if pa.types.is_dictionary(f.type) else f.type, nullable=f.nullable)
            for f in s]) for s in schemas]
        schema = pa.unify_schemas(schemas, promote_options="permissive")
        destination = Path(destination)
        proof_path = destination.with_suffix(destination.suffix+".capture.json")
        identity = dict(parts=[dict(cutoff=r["cutoff"], rows=r["rows"], sha256=digest(r["path"])) for r in selected],
            columns=schema.names, batch_rows=batch_rows)
        if destination.exists():
            if not proof_path.exists():
                raise ValueError("Never overwrite a completed feature-cache file without its capture proof")
            proof = json.loads(proof_path.read_text())
            if proof["identity"] != identity or proof["output_sha256"] != digest(destination):
                raise ValueError("Completed streamed cache identity changed")
            return proof["report"]
        with NamedTemporaryFile(dir=destination.parent, prefix=destination.name+".",
                suffix=".partial", delete=False) as handle:
            temporary = Path(handle.name)
        with pq.ParquetWriter(temporary, schema) as writer:
            for record in selected:
                for batch in pq.ParquetFile(record["path"]).iter_batches(batch_size=batch_rows):
                    table = pa.Table.from_batches([batch]).cast(schema, safe=True)
                    writer.write_table(table)
        report = dict(rows=sum(r["rows"] for r in selected), columns=len(schema),
            cutoffs=[r["cutoff"] for r in selected], batch_rows=batch_rows,
            fitter_frames_unchanged=True, source_rows_sampled=False)
        dump(proof_path, dict(identity=identity, output_sha256=digest(temporary), report=report))
        temporary.replace(destination)
        return report


@contextmanager
def capture_to_disk(columns, target, entity, root, output):
    original = core._materialize_at_cutoff
    sink = FrameSink(output)
    active = 0

    def materialize(*args, **kwargs):
        nonlocal active
        active += 1
        try:
            frame = original(*args, **kwargs)
            # Nested root batching must not be recorded as duplicate cohorts.
            if active != 1:
                return frame
            key = entity if entity in frame else f"{root}_{entity}"
            if key not in frame:
                raise ValueError("Frozen frame lacks its entity key")
            missing = set([*columns, target])-set(frame.columns)
            if missing:
                raise ValueError(f"Frozen cache is missing {len(missing)} declared columns")
            captured = frame.loc[:, [*columns, target]].copy()
            spec = args[6] if len(args)>6 else kwargs.get("label_spec")
            if spec is not None and getattr(spec, "available_at", None):
                captured["__kurversc_label_available_at"] = frame[spec.available_at].to_numpy()
            captured["__event_entity"] = frame[key].to_numpy()
            captured["__event_cutoff"] = str(pd.Timestamp(kwargs["cut_date"]))
            sink.append(captured, kwargs["cut_date"])
            return frame
        finally:
            active -= 1

    core._materialize_at_cutoff = materialize
    try:
        yield sink
    finally:
        core._materialize_at_cutoff = original
