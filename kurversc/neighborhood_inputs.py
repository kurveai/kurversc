"""Frozen numeric inputs for schema-generic, bounded relational learners.

Reuses GraphReduce extraction/context and the existing lexical payload encoder.
No targets are read; representation fitting is restricted to training examples.
Mutable source attributes must already have correct point-in-time source views.
"""

from dataclasses import dataclass, field, replace

from .relational_inputs import (
    PayloadEncoder,
    key_parts,
    member_payload,
    memberships,
    payload_columns,
    schema_paths,
    structural_columns,
)


@dataclass
class BoundedNeighborhoodInputs:
    max_depth: int = 2
    max_paths: int = 12
    max_neighbors: int = 128
    max_payload_columns: int = 128
    text_components: int = 64
    context_depth: int = 1
    context_columns: int = 8
    memory_limit: str = "4GB"
    random_state: int = 0
    audit: list = field(default_factory=list, init=False)

    def _rows(self, examples, tables, relationships, path, columns, plan):
        if not examples.index.is_unique or examples[self.timestamp_].isna().any():
            raise ValueError("Examples need unique indexes and complete times")
        identifiers = examples[[self.entity_, self.timestamp_]]
        terminal = path[-1] if path else self.root_
        if not path:
            links = memberships(
                identifiers,
                tables,
                relationships,
                root=self.root_,
                entity=self.entity_,
                timestamp=self.timestamp_,
                path=(),
                max_neighbors=1,
                seed=self.random_state,
                memory_limit=self.memory_limit,
            )
            values = member_payload(tables[terminal], links, columns)
            result = values.reindex(links["__neighbor_key_0"]).reset_index(drop=True)
            result.insert(0, "__example_id", links["__example_id"].to_numpy())
            return result
        from graphreduce.learned_messages import extract_path

        selected = dict(tables)
        selected[terminal] = replace(tables[terminal], columns=columns)
        return extract_path(
            identifiers,
            selected,
            relationships,
            root_name=self.root_,
            entity_key=self.entity_,
            timestamp=self.timestamp_,
            path=path,
            max_neighbors=self.max_neighbors,
            max_columns=self.max_payload_columns,
            seed=self.random_state,
            memory_limit=self.memory_limit,
            context_depth=self.context_depth,
            context_max_columns=self.context_columns,
            context_max_relations=12,
            context_plan_cache=plan,
        )

    def fit_transform(
        self, examples, *, tables, relationships, root_name, entity_key, timestamp
    ):
        """Fit lexical encoders on training rows and return their numeric bags."""
        if (
            min(
                self.max_depth,
                self.max_paths,
                self.max_neighbors,
                self.max_payload_columns,
                self.text_components,
            )
            < 1
        ):
            raise ValueError("Neighborhood budgets must be positive")
        if examples.empty:
            raise ValueError("Training examples must be nonempty")
        self.root_, self.entity_, self.timestamp_ = root_name, entity_key, timestamp
        structural = structural_columns(tables, relationships)
        paths = [
            (),
            *[p for p, _ in schema_paths(root_name, relationships, self.max_depth)],
        ]
        paths = sorted(set(paths), key=lambda p: (len(p), p))
        self.plans_, self.encoders_, self.columns_, self.paths_, self.audit = (
            {},
            {},
            {},
            {},
            [],
        )
        output = {}
        for path in paths:
            terminal = path[-1] if path else root_name
            if len(key_parts(tables[terminal].key)) != 1:
                continue
            columns = payload_columns(tables[terminal], structural[terminal])[
                : self.max_payload_columns
            ]
            if not columns:
                continue
            plan = {}
            rows = self._rows(examples, tables, relationships, path, columns, plan)
            if rows.empty:
                continue
            encoder = PayloadEncoder(
                components=self.text_components,
                max_numeric_columns=128,
                random_state=self.random_state,
            ).fit(rows.drop(columns="__example_id"))
            if not encoder.feature_names_:
                continue
            name = f"p{len(output)}"
            numeric = encoder.transform(rows.drop(columns="__example_id"))
            numeric.insert(0, "__example_id", rows["__example_id"].to_numpy())
            output[name] = numeric
            (
                self.plans_[name],
                self.encoders_[name],
                self.columns_[name],
                self.paths_[name],
            ) = plan, encoder, columns, path
            self.audit.append(
                dict(
                    name=name,
                    path=list(path),
                    neighbors=len(rows),
                    covered_roots=int(rows["__example_id"].nunique()),
                    payload_columns=list(columns),
                    **encoder.audit_,
                )
            )
            if len(output) >= self.max_paths:
                break
        if not output:
            raise ValueError("No eligible, informative neighborhood inputs")
        return output

    def transform(self, examples, *, tables, relationships):
        if not hasattr(self, "paths_"):
            raise ValueError("Fit neighborhood inputs before transforming")
        output = {}
        for name, path in self.paths_.items():
            rows = self._rows(
                examples,
                tables,
                relationships,
                path,
                self.columns_[name],
                self.plans_[name],
            )
            numeric = self.encoders_[name].transform(rows.drop(columns="__example_id"))
            numeric.insert(0, "__example_id", rows["__example_id"].to_numpy())
            output[name] = numeric
        return output
