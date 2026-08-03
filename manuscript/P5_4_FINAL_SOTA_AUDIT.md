# P5.4 Final SOTA Audit

Date checked: 2026-08-03.

## Direct Target-Set Comparator Status

| dataset | current direct comparator kept for manuscript | comparator AUC | RegionAwareTCN full-DM3000 AUC | status |
| --- | --- | ---: | ---: | --- |
| SL329 | IDP-EDL, Briefings in Bioinformatics 2025 | 0.915000 | 0.919327 | keep AUC-level point-comparison claim |
| MXD494 | FusionEncoder, Bioinformatics 2025 | 0.842000 | 0.850637 | keep AUC-level point-comparison claim |
| DISORDER723 | IDP-EDL, Briefings in Bioinformatics 2025 | 0.943000 | 0.944611 | keep only conservative AUC-level point-comparison claim; MCC remains lower than IDP-EDL |

## 2026 Literature Risks Checked

- CAID3 and flDPnn3 are important 2026 context but report CAID-style benchmarks rather than direct SL329/MXD494/DISORDER723 tables.
- ESMDisPred is a 2026 bioRxiv structure-aware method reporting CAID3-style metrics, not direct target-set results for this manuscript's three external tests.
- D2MOE arXiv 2603.06292 was inspected because search results suggested broad benchmark improvements. The paper reports TS115, CASP12 and CB513 comparisons, not SL329/MXD494/DISORDER723 intrinsic-disorder target-set results, so it should not replace the direct comparator table.

## Claim Boundary After Audit

The main text can retain: "AUC-level state-of-the-art point performance on SL329, MXD494 and DISORDER723 under the full DM3000 training protocol, relative to curated direct target-set aggregate metrics."

The main text should not claim:

- statistically significant superiority over external SOTA methods;
- uniformly SOTA NR25 performance;
- overall SOTA across all metrics;
- direct superiority over CAID3-only or preprint methods that do not report the same target sets.

## Sources To Cite Or Mention

- Bioinformatics author guidelines: https://academic.oup.com/bioinformatics/pages/author-guidelines
- IDP-EDL: https://academic.oup.com/bib/article/26/2/bbaf182/8116687
- FusionEncoder: https://academic.oup.com/bioinformatics/article/41/7/btaf362/8169326
- CAID3 PubMed: https://pubmed.ncbi.nlm.nih.gov/40859602/
- CAID3 full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/
- flDPnn3 DOI: https://doi.org/10.1016/j.jmb.2026.169629
- ESMDisPred DOI: https://doi.org/10.64898/2026.01.22.701204
- D2MOE arXiv: https://arxiv.org/abs/2603.06292
