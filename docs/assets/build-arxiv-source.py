"""Package the current manuscript and vector figures for arXiv's XeLaTeX processor."""

from argparse import ArgumentParser
from pathlib import Path
import shutil
import subprocess
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


DOCS = Path(__file__).resolve().parent.parent
FIGURES = (
    "kurversc-staged-architecture.pdf",
    "kurversc-keyed-reduction.pdf",
    "kurversc-reduction-example.pdf",
    "kurversc-relarena-elo-20261105.pdf",
)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DOCS / "kurversc-0.2.0-arxiv-source.zip",
    )
    args = parser.parse_args()
    pandoc = shutil.which("pandoc")
    if not pandoc:
        parser.error("pandoc must be installed to expand the manuscript and references")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kurversc-arxiv-") as directory:
        tex = Path(directory) / "kurversc.tex"
        subprocess.run(
            [
                pandoc,
                "kurversc-staged-technical-report.md",
                "--from=markdown+tex_math_dollars+raw_tex",
                "--citeproc",
                "--lua-filter=assets/report-links.lua",
                "--standalone",
                "--to=latex",
                "--highlight-style=tango",
                "--resource-path=.:assets",
                f"--output={tex}",
            ],
            cwd=DOCS,
            check=True,
        )
        contents = {"kurversc.tex": tex.read_bytes()}
        contents.update(
            (f"assets/{name}", (DOCS / "assets" / name).read_bytes())
            for name in FIGURES
        )
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
            for name, data in contents.items():
                info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, data)
    print(f"Created {output}; main file: kurversc.tex; processor: XeLaTeX")


if __name__ == "__main__":
    main()
