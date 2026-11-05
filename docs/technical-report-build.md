# Building the current technical report

The current report describes published KurveRSC 0.2.0 and GraphReduce 1.10.18.
The October 27 manuscript, PDF and bibliography are preserved under `archive/`
and describe a superseded procedure.

Use Pandoc with citeproc and XeLaTeX (including the LaTeX packages named in the
manuscript header). From the `docs/` directory:

```bash
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

The PDF was built with Pandoc 3.1.8 and XeTeX 0.999994 (TeX Live 2022).

Before publishing, check local links and citation keys, render every page with
Poppler, and inspect headings, tables, formulas, code and bibliography. Do not
copy historical benchmark scores into the current report without a matching
completed evaluation of the released procedure.
