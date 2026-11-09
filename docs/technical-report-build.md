# Building the current technical report

The KurveRSC 0.2.0 technical report describes published KurveRSC 0.2.0 and
GraphReduce 1.10.18. Its source and PDF are the canonical report for this release.

Use Pandoc with citeproc and XeLaTeX (including the LaTeX packages named in the
manuscript header), librsvg, and Python with Matplotlib and NumPy. From the
`docs/` directory, regenerate the vector figures and build the report:

```bash
python assets/build-relarena-elo-figure.py
python assets/build-method-figures.py

pandoc kurversc-staged-technical-report.md \
  --from=markdown+tex_math_dollars+raw_tex \
  --citeproc \
  --lua-filter=assets/report-links.lua \
  --pdf-engine=xelatex \
  --highlight-style=tango \
  --resource-path=.:assets \
  --output=kurversc-staged-technical-report.pdf
```

The title page uses the manuscript's YAML `abstract` field for the short,
unnumbered abstract. The default article abstract environment centers the
heading and sets the paragraph in a narrower, justified block. The results
panel follows on that page. Automatic `toc` is disabled because a manual LaTeX
`\tableofcontents` places Contents on the next page, before Section 1,
Introduction. Keep that ordering when changing the front matter; enabling
automatic `toc` would put it ahead of the results panel.

The results heading's `results-at-glance` identifier is handled by
`report-links.lua`, which renders it as a centered 11-point heading in LaTeX.
Other headings retain their normal section styling.

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

`build-method-figures.py` renders the editable six-panel architecture SVG to
its vector PDF and regenerates the keyed-reduction and worked-example figures.
The architecture SVG is the canonical artwork; rebuilding preserves its
composition. The script uses only the Python standard library and
`rsvg-convert`. Optional `--preview-dir /path/to/previews` writes PNGs and a
dimension receipt outside the source assets. At the report's 6.94-inch text
width the architecture is approximately 4.63 inches tall; the other two figures
are 3.22 and 3.02 inches tall. The dense overview's smallest main labels are
approximately 5.6 points, excluding mathematical subscripts; the two explanatory
figures use minimum labels of 9.3 points. All three remain vector artwork.

The paragraphs immediately after Figure 1 explain each stage, distinguish
historical examples from feature columns and frame ensembles, and define the
S4 baseline and optional procedures. They also explain why S4 is postponed to
control cost. Section 13 records the resulting limitation: graph selection and
optional-procedure selection are sequential and can miss interactions. Keep
these explanations adjacent to the overview when changing its layout.

Both algorithms use native LaTeX `algorithm` and `algpseudocode` environments,
with numbered steps, explicit inputs/outputs, indented control flow and
9.5-point body text. They are searchable text rather than screenshots or
verbatim blocks. The corresponding packages must be available to XeLaTeX.

The PDF was built with Pandoc 3.1.8 and XeTeX 0.999994 (TeX Live 2022).

Before publishing, check local links and citation keys, render every page with
Poppler, and inspect headings, tables, formulas, code and bibliography. Do not
report benchmark scores without a matching completed evaluation of the
released procedure.
