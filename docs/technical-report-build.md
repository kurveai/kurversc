# Building the current technical report

The KurveRSC 0.2.0 technical report describes published KurveRSC 0.2.0 and
GraphReduce 1.10.18. Its source and PDF are the canonical report for this release.

Use Pandoc with citeproc and XeLaTeX (including the LaTeX packages named in the
manuscript header), librsvg, and Python with Matplotlib and NumPy. From the
`docs/` directory, regenerate the vector figures and build the report:

```bash
python assets/build-relarena-elo-figure.py
rsvg-convert --format pdf \
  --output assets/kurversc-staged-architecture.pdf \
  assets/kurversc-staged-architecture.svg

pandoc kurversc-staged-technical-report.md \
  --from=markdown+tex_math_dollars+raw_tex \
  --citeproc \
  --lua-filter=assets/report-links.lua \
  --pdf-engine=xelatex \
  --highlight-style=tango \
  --resource-path=.:assets \
  --output=kurversc-staged-technical-report.pdf
```

The source resolves references from `kurversc-staged-references.bib` and selects
`assets/ieee.csl` in its metadata. The vendored [IEEE CSL style](https://github.com/citation-style-language/styles/blob/master/ieee.csl)
is licensed under CC BY-SA 3.0; its upstream attribution and license are retained
in the style file. It numbers citations/references and repeats author names.

Run `assets/report-links.lua` **after citeproc**, as above. It shortens only
bibliography link labels (for example, `GitHub source` or an arXiv identifier),
preserving every original destination, including commit-pinned sources. Body
links and in-text citation links are unchanged.

Figures are editable vector SVGs under `assets/`, accompanied by vector PDFs
for publication. The filter selects these PDFs for LaTeX output and leaves SVG
links intact for other formats. The Elo figure generator reads the frozen,
published leaderboard extract rather than recomputing ratings or mixing
comparison panels.

The PDF was built with Pandoc 3.1.8 and XeTeX 0.999994 (TeX Live 2022).

Before publishing, check local links and citation keys, render every page with
Poppler, and inspect headings, tables, formulas, code and bibliography. Do not
report benchmark scores without a matching completed evaluation of the
released procedure.
