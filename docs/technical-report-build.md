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
`assets/kurversc-full-authors.csl` in its metadata. This local adaptation of the
[IEEE CSL style](https://github.com/citation-style-language/styles/blob/master/ieee.csl)
preserves numbered citations and prints complete author lists with full given
names, including long author lists. The upstream attribution, template link and
CC BY-SA 3.0 license are retained. The unmodified upstream style remains in
`assets/ieee.csl` for provenance.

The BibTeX file must also contain every author: a complete-list style cannot
recover names omitted from source metadata. The RelArena, TabPFN-3 and
TabPFN-3.5 entries use complete author lists from their cited arXiv versions.
Before publishing, verify the rendered reference list against the parsed
BibTeX author arrays, including the last author of each long entry.

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
