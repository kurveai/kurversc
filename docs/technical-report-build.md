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
  --pdf-engine=xelatex \
  --highlight-style=tango \
  --resource-path=.:assets \
  --output=kurversc-staged-technical-report.pdf
```

The source resolves references from `kurversc-staged-references.bib`. The current
PDF was built with Pandoc 3.1.8 and XeTeX 0.999994 (TeX Live 2022).

Before publishing, check local links and citation keys, render every page with
Poppler, and inspect headings, tables, formulas, code and bibliography. Do not
copy historical benchmark scores into the current report without a matching
completed evaluation of the released procedure.
