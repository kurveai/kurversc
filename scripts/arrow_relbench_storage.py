"""Opt-in Arrow string storage for isolated replay workers, never a default.

Keep numeric/date loading unchanged. Arrow-backed text can share buffers across
ordinary DataFrame copies; avoid immediately expanding it back to Python objects
at DuckDB's registration boundary. Schema and cell-value parity are mandatory.
"""

from contextlib import contextmanager
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_arrow_strings(cls, path):
    table = pq.read_table(path, memory_map=True, use_threads=False, pre_buffer=False)

    def dtype(kind):
        return pd.ArrowDtype(kind) if pa.types.is_string(kind) or pa.types.is_large_string(kind) else None

    frame = table.to_pandas(types_mapper=dtype, use_threads=False)
    metadata = {name: json.loads(table.schema.metadata[name.encode()])
                for name in ("fkey_col_to_pkey_table", "pkey_col", "time_col")}
    return cls(frame, **metadata)


def arrow_compatible_frame(frame, original):
    has_arrow_text = any(isinstance(dtype, pd.ArrowDtype) and
        (pa.types.is_string(dtype.pyarrow_dtype) or pa.types.is_large_string(dtype.pyarrow_dtype))
        for dtype in frame.dtypes)
    if not has_arrow_text:
        return original(frame)
    result = frame.copy(deep=False)
    for name in result:
        if isinstance(result[name].dtype, pd.StringDtype):
            result[name] = result[name].astype(object)
    return result


@contextmanager
def arrow_relbench_storage(audit):
    from relbench.base import Table
    import kurversc.core as core

    original_load = Table.__dict__["load"]
    original_compatible = core._duckdb_compatible_frame

    def load(cls, path):
        if Path(path).parent.name != "db":
            return original_load.__func__(cls, path)
        table = read_arrow_strings(cls, path)
        audit.append(dict(path=str(path), rows=len(table.df),
                          frame_bytes=int(table.df.memory_usage(deep=True).sum()),
                          arrow_string_columns=[name for name in table.df
                              if isinstance(table.df[name].dtype, pd.ArrowDtype)]))
        return table

    Table.load = classmethod(load)
    core._duckdb_compatible_frame = lambda frame: arrow_compatible_frame(frame, original_compatible)
    try:
        yield
    finally:
        Table.load = original_load
        core._duckdb_compatible_frame = original_compatible
