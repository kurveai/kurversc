"""Research-only exact acceleration of the existing entity-cluster bootstrap.

The incumbent selector/rule, random draws, bootstrap positions, and scores stay
identical. Only constructing entity blocks and flattening sampled blocks are
accelerated. This does NOT edit or replace production temporal_policy defaults.
"""

import ast
import inspect

import numpy as np

from kurversc.temporal_policy import select_temporal_policy


class EntityBlocks:
    def __init__(self, codes, count):
        codes = np.asarray(codes, dtype=np.int64)
        if codes.ndim != 1 or count < 1 or np.any(codes < 0) or np.any(codes >= count):
            raise ValueError("Complete bounded entity codes required")
        self.positions = np.argsort(codes, kind="stable")
        self.lengths = np.bincount(codes, minlength=count)
        if np.any(self.lengths == 0):
            raise ValueError("Entity enumeration contains empty blocks")
        self.starts = np.cumsum(np.r_[0, self.lengths[:-1]])

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        start = self.starts[index]
        return self.positions[start:start+self.lengths[index]]


def sample_positions(blocks, indices):
    """Bit-identical to concatenate([flatnonzero(codes==i) for i in draws])."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(blocks)):
        raise ValueError("Bounded bootstrap entity draws required")
    if np.all(blocks.lengths == 1):
        return blocks.positions[indices]
    sizes = blocks.lengths[indices]
    output_starts = np.cumsum(np.r_[0, sizes[:-1]])
    offsets = np.repeat(blocks.starts[indices]-output_starts, sizes)
    return blocks.positions[offsets+np.arange(sizes.sum())]


def make_accelerated_selector():
    """Translate exactly two reviewed expressions in the installed selector.

    Fail closed if its implementation changes. Executed code is trusted local
    module source, never data, prompts, schemas, or artifact-provided Python.
    """
    source = inspect.getsource(select_temporal_policy)
    replacements = {
        "[np.flatnonzero(codes == index) for index in range(len(unique))]":
            "EntityBlocks(codes, len(unique))",
        "np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])":
            "sample_positions(blocks, rng.integers(0, len(blocks), len(blocks)))",
    }
    expected = {key:ast.dump(ast.parse(key,mode="eval").body,include_attributes=False)
        for key in replacements}
    replaced = {key:0 for key in replacements}

    class Rewrite(ast.NodeTransformer):
        def generic_visit(self,node):
            fingerprint = ast.dump(node,include_attributes=False)
            for key,value in expected.items():
                if fingerprint == value:
                    replaced[key] += 1
                    return ast.copy_location(ast.parse(replacements[key],mode="eval").body,node)
            return super().generic_visit(node)

    tree = ast.fix_missing_locations(Rewrite().visit(ast.parse(source)))
    if set(replaced.values()) != {1}:
        raise ValueError("Installed bootstrap changed; exact acceleration not applicable")
    namespace = dict(select_temporal_policy.__globals__, EntityBlocks=EntityBlocks,
        sample_positions=sample_positions)
    exec(compile(tree,"<reviewed-exact-bootstrap-acceleration>","exec"),namespace)
    return namespace["select_temporal_policy"]


accelerated_select_temporal_policy = make_accelerated_selector()
