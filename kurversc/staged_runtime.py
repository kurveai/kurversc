"""Content identities for the portable staged implementation and its runtime."""

import ast
from pathlib import Path
import re

from .task_context import _file_digest


def staged_source_identity(root):
    """Hash the transitive stage code, not unrelated research or old results."""
    root = Path(root)
    stage_root = root/"scripts"
    if not stage_root.is_dir():
        stage_root = root/"kurversc"/"_staged"
    pending = ["run_portable_staged"]
    visited = set()
    files = set((root/"kurversc").rglob("*.py"))
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        path = stage_root/(name+".py")
        if not path.exists():
            continue
        files.add(path)
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.ImportFrom):
                if node.module in {"scripts","kurversc._staged"}:
                    pending.extend(alias.name for alias in node.names)
                elif node.module and node.module.startswith(("scripts.","kurversc._staged.")):
                    pending.append(node.module.rsplit(".",1)[-1])
            elif isinstance(node,ast.Import):
                pending.extend(alias.name.rsplit(".",1)[-1] for alias in node.names if alias.name.startswith(("scripts.","kurversc._staged.")))
        pending.extend(re.findall(r"\b(?:scripts|kurversc\._staged)\.([A-Za-z_]\w*)",source))
        pending.extend(re.findall(r"\b(?:scripts/|kurversc/_staged/)([A-Za-z_]\w*)\.py",source))
    result = {str(p.relative_to(root)):_file_digest(p) for p in sorted(files)}
    # The existing execution adapter remains an explicit optional dependency;
    # its source must be pinned too, including when installed outside the repo.
    import os
    from .task_source import CONTEXT_ENV
    if not os.environ.get(CONTEXT_ENV):
        try:
            import relarena
        except ImportError:
            pass
        else:
            package = Path(relarena.__file__).parent
            result.update({"relarena/"+str(p.relative_to(package)):_file_digest(p) for p in sorted(package.rglob("*.py"))})
    return result
