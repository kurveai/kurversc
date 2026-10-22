"""Verify wheel/source completeness without loading any model checkpoints.

Run manually after ``uv build`` from this repository. This checks distribution
contents and source parity, not benchmark performance or full qualification.
"""

import argparse
import ast
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import tarfile
from zipfile import ZipFile

from staged_build import relocated_stage, stage_sources


def check(root):
    root = Path(root).resolve()
    wheel, = (root/'dist').glob('kurversc-*.whl')
    source, = (root/'dist').glob('kurversc-*.tar.gz')
    stages = stage_sources(root)
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name, = (name for name in names if name.endswith('.dist-info/METADATA'))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        assert metadata['Version'] == '0.2.0rc1'
        for file in (root/'kurversc').rglob('*.py'):
            if '_staged' in file.parts:
                continue
            assert archive.read(str(file.relative_to(root))) == file.read_bytes(), file
        for file in stages:
            assert archive.read('kurversc/_staged/'+file.name) == relocated_stage(file.read_text()).encode(), file
        assert sum(name.startswith('kurversc/_staged/') and name.endswith('.py') for name in names) == len(stages)+1
        assert not any(name.startswith(('results/', 'data/', 'tests/', 'scripts/')) for name in names)
        for name in names:
            if name.endswith('.py'):
                ast.parse(archive.read(name))
    with tarfile.open(source, 'r:gz') as archive:
        members = archive.getnames()
        prefix = members[0].split('/')[0]+'/'
        names = {name.removeprefix(prefix) for name in members}
        assert 'staged_build.py' in names
        assert all(str(file.relative_to(root)) in names for file in stages)
        assert not any(name.startswith(('results/', 'data/', 'catboost_info/')) for name in names)
    return dict(status='passed', version=metadata['Version'], stage_modules=len(stages),
        wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        sdist_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_and_wheel_exact=True, no_models_or_benchmark_data=True,
        full_21_task_qualification=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(check(parser.parse_args().root), indent=2))
