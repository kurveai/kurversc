"""Experimental schema-generic, as-of joint summaries of ordinary fact rows.

No labels or learned messages are accepted. This adapter projects bounded
payloads along existing FK paths, then uses GraphReduce joint-distribution SQL.
Unique dimension joins preserve event grain; each path is reduced separately.
"""

from dataclasses import dataclass

import duckdb
import pandas as pd

from graphreduce.joint_distribution import JointDistributionSketch
from .event_summaries import EventSummaryPlan, _compatible_frame, _payloads, _q


@dataclass
class JointEventDistributionPlan:
    max_depth: int = 2
    max_paths: int = 8
    memory_limit: str = "4GB"
    threads: int = 2

    def _connection(self, tables):
        con = duckdb.connect(config={
            "memory_limit": self.memory_limit, "threads": self.threads,
            "max_temp_directory_size": "16GB",
        })
        for i, (name, table) in enumerate(sorted(tables.items())):
            if table.key and table.frame[table.key].duplicated().any():
                con.close()
                raise ValueError(f"Declared source key is not unique: {name}")
            con.register(f"t{i}", _compatible_frame(table.frame))
        return con, {name: f"t{i}" for i, name in enumerate(sorted(tables))}

    def _relation_query(self, tables, refs, path, payloads, cutoff):
        stamp = pd.Timestamp(cutoff).isoformat(sep=" ")
        names = [self.root_, *(step[0] for step in path.steps)]
        joins = f"seeds s JOIN {refs[self.root_]} a0 ON s.entity = a0.{_q(tables[self.root_].key)}"
        for i, (name, fk) in enumerate(path.steps, 1):
            joins += f" JOIN {refs[name]} a{i} ON a{i}.{_q(fk)} = a{i - 1}.{_q(tables[names[i - 1]].key)}"
        for j, (fk, name, _) in enumerate(path.dimensions):
            dimension = tables[name]
            clause = f"d{j}.{_q(dimension.key)} = a{len(path.steps)}.{_q(fk)}"
            if dimension.timestamp:
                clause += f" AND d{j}.{_q(dimension.timestamp)} < TIMESTAMP '{stamp}'"
            joins += f" LEFT JOIN {refs[name]} d{j} ON {clause}"
        where = [
            f"a{i}.{_q(tables[name].timestamp)} < TIMESTAMP '{stamp}'"
            for i, name in enumerate(names) if tables[name].timestamp
        ]
        projected = ["s.entity AS __entity"]
        projected += [f"{expression} AS {_q(name)}" for name, expression in payloads.items()]
        terminal = tables[names[-1]]
        key = f"a{len(path.steps)}.{_q(terminal.key)}" if terminal.key else "s.entity"
        projected.append(f"{key} AS __fact")
        return f"SELECT {', '.join(projected)} FROM {joins}" + (
            " WHERE " + " AND ".join(where) if where else ""
        )

    def fit(self, tables, root, ids, cutoff):
        self.root_ = root
        paths = EventSummaryPlan.fit(
            tables, root, cutoff, max_depth=self.max_depth, max_paths=self.max_paths,
            max_numeric=4, max_categories=2, max_values=3, max_dimensions=2,
        )
        con, refs = self._connection(tables)
        self.paths_, self.audit_ = [], []
        try:
            con.register("seeds", _compatible_frame(pd.DataFrame({"entity": pd.Index(ids).unique()})))
            for path in paths.paths:
                terminal_alias = f"a{len(path.steps)}"
                payloads = {f"r_{c}": f"{terminal_alias}.{_q(c)}" for c in path.numeric}
                payloads.update({f"r_{c}": f"{terminal_alias}.{_q(c)}" for c, _ in path.categories})
                for j, (_, name, numeric) in enumerate(path.dimensions):
                    payloads.update({f"d{j}_{c}": f"d{j}.{_q(c)}" for c in numeric})
                    _, categories = _payloads(tables[name], cutoff, 4, 2, 3)
                    payloads.update({f"d{j}_{c}": f"d{j}.{_q(c)}" for c, _ in categories})
                if len(payloads) < 2:
                    continue
                query = self._relation_query(tables, refs, path, payloads, cutoff)
                rows = con.execute(
                    f"SELECT * EXCLUDE (__entity, __fact) FROM ({query}) "
                    "ORDER BY hash(__entity, __fact) LIMIT 20000"
                ).df()
                if rows.empty:
                    continue
                sketch = JointDistributionSketch().fit(rows)
                marginal, joint = sketch.expressions({c: _q(c) for c in rows})
                if not joint:
                    continue
                self.paths_.append((path, payloads, sketch))
                self.audit_.append(dict(
                    steps=list(path.steps), dimensions=list(path.dimensions),
                    input_columns=list(rows), marginal_features=len(marginal),
                    joint_features=len(joint), **sketch.audit_,
                ))
        finally:
            con.close()
        self.calibration_cutoff_ = str(pd.Timestamp(cutoff))
        return self

    def transform(self, tables, ids, cutoff, *, training=False):
        if not hasattr(self, "paths_"):
            raise ValueError("Fit joint event plan before inference")
        if pd.Timestamp(cutoff) < pd.Timestamp(self.calibration_cutoff_) and not training:
            raise ValueError("Query cutoff precedes encoder calibration; only training prefixes may do this")
        ids = pd.Index(ids)
        if ids.hasnans:
            raise ValueError("Root identities cannot be missing")
        con, refs = self._connection(tables)
        marginal_blocks, joint_blocks = [], []
        try:
            con.register("seeds", _compatible_frame(pd.DataFrame({"entity": ids.unique()})))
            for number, (path, payloads, sketch) in enumerate(self.paths_):
                query = self._relation_query(tables, refs, path, payloads, cutoff)
                marginal, joint = sketch.expressions({c: _q(c) for c in payloads})
                aliases = [name for name, _ in (*marginal, *joint)]
                operations = ", ".join(f"{expr} AS {_q(name)}" for name, expr in (*marginal, *joint))
                result = con.execute(
                    f"SELECT __entity, {operations} FROM ({query}) GROUP BY __entity"
                ).df().set_index("__entity")
                result = result.reindex(ids)
                for name in aliases:
                    if name.endswith("_support"):
                        result[name] = result[name].fillna(0)
                marginal_blocks.append(result[[c for c, _ in marginal]].add_prefix(f"jd_p{number}_"))
                joint_blocks.append(result[[c for c, _ in joint]].add_prefix(f"jd_p{number}_"))
        finally:
            con.close()
        def joined(blocks):
            result = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=ids)
            return result.reset_index(drop=True).astype("float32")
        return joined(marginal_blocks), joined(joint_blocks)
