# Bioinformatics LaTeX Submission Draft

This directory contains a Bioinformatics-oriented LaTeX manuscript prepared with the Oxford University Press `oup-authoring-template` class.

## Template Source

- Official/OUP template page used for journal guidance: https://academic.oup.com/bioinformatics/pages/author-guidelines
- Overleaf OUP General Template page checked: https://www.overleaf.com/latex/templates/oup-general-template/ybpypwncdxyb
- Local template files downloaded from the CTAN OUP authoring template bundle: https://ctan.org/pkg/oup-authoring-template
- Download date: 2026-08-03

## Main Files

- `main.tex`: current P5.7 Bioinformatics-style manuscript draft using the P4.8 warm-start RegionAdapterMoETCN main model.
- `regionawaretcn_refs.bib`: BibTeX references cited by `main.tex`.
- `oup-authoring-template.cls`: OUP LaTeX class.
- `oup-abbrvnat.bst`: OUP author-year bibliography style.
- `oup-plain.bst`: OUP numbered bibliography style, included for completeness.
- `Fig/`: copied main figure PDFs.

## Compile Command Used In This Workspace

The current P5.7 PDF in this package was generated with a user-level Tectonic environment:

```bash
/data8T/IDPs_DM3000Train/.conda/tectonic/bin/tectonic --keep-logs --keep-intermediates main.tex
```

Tectonic automatically downloaded the required TeX packages and ran BibTeX during compilation.

The compiled P5.7 manuscript is also copied to:

```bash
manuscript/latex/RegionAdapterMoETCN_Bioinformatics_P5_7_manuscript.pdf
```

If using a conventional TeX Live environment instead, use:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

or:

```bash
latexmk -pdf main.tex
```

## Required Manual Replacements Before Submission

- Replace all `\placeholder{...}` fields with real authors, affiliations, funding, conflicts and acknowledgements.
- Confirm that `https://github.com/MedBio07/IDP_RegionAware` is public and accessible.
- Insert the Zenodo or institutional archive DOI for weights, predictions and large derived artifacts.
- Re-check all BibTeX entries with a reference manager.
- Confirm whether Bioinformatics prefers author-year or numbered references for the final upload. This draft currently uses `namedate` and `oup-abbrvnat`.
