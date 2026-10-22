"""Package the complete staged dependency closure without the research archive.

Only import/module paths and repository-root locations are relocated. No
selection, fit, feature, budget or gating logic is changed in wheel builds.
This module intentionally imports only the build backend and stdlib.
"""

import ast
from pathlib import Path
import re

from setuptools.command.build_py import build_py


def stage_sources(root):
    root = Path(root)
    pending,visited,result = ["run_portable_staged"],set(),[]
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        file = root/"scripts"/(name+".py")
        if not file.exists():
            continue
        result.append(file)
        text = file.read_text()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node,ast.ImportFrom):
                if node.module=="scripts":
                    pending.extend(alias.name for alias in node.names)
                elif node.module and node.module.startswith("scripts."):
                    pending.append(node.module.split(".")[1])
            elif isinstance(node,ast.Import):
                pending.extend(a.name.split(".")[1] for a in node.names if a.name.startswith("scripts."))
        pending.extend(re.findall(r"\bscripts\.([A-Za-z_]\w*)",text))
        pending.extend(re.findall(r"\bscripts/([A-Za-z_]\w*)\.py",text))
    return sorted(result)


def relocated_stage(text):
    # Qualified modules also occur in subprocess CLI arguments and provenance
    # paths. Relocate those too; avoid import-time aliases or sys.modules hacks.
    text = text.replace("from scripts import","from kurversc._staged import")
    text = text.replace("scripts.","kurversc._staged.")
    text = text.replace("scripts/","kurversc/_staged/")
    return text.replace("Path(__file__).resolve().parents[1]","Path(__file__).resolve().parents[2]")


class BoundedStagedBuild(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib)/"kurversc"/"_staged"
        destination.mkdir(parents=True,exist_ok=True)
        (destination/"__init__.py").write_text('"""Private complete staged execution runtime."""\n')
        for file in stage_sources(Path.cwd()):
            text = relocated_stage(file.read_text())
            ast.parse(text)  # A broken relocation must fail the wheel build.
            (destination/file.name).write_text(text)

    def get_source_files(self):
        return [*super().get_source_files(),"staged_build.py",*[str(p.relative_to(Path.cwd())) for p in stage_sources(Path.cwd())]]

    def get_outputs(self,include_bytecode=1):
        result = list(super().get_outputs(include_bytecode))
        destination = Path(self.build_lib)/"kurversc"/"_staged"
        return result+[str(destination/"__init__.py"),*[str(destination/p.name) for p in stage_sources(Path.cwd())]]
