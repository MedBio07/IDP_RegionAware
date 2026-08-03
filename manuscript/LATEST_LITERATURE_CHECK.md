# Latest Literature Check Draft

Date checked: 2026-08-03

## Sources Checked

- IDP-EDL, Briefings in Bioinformatics 2025: https://academic.oup.com/bib/article/26/2/bbaf182/8116687
- FusionEncoder, Bioinformatics 2025: https://academic.oup.com/bioinformatics/article/41/7/btaf362/8169326
- CAID3, Proteins 2026 / PubMed: https://pubmed.ncbi.nlm.nih.gov/40859602/
- CAID3 full text / PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/
- 2026 review, Modern resources for intrinsic disorder predictions: https://link.springer.com/article/10.1007/s00018-026-06087-3

## Current Direct Benchmark Competitors

- IDP-EDL remains the direct local comparator for SL329 and DISORDER723 in the curated project tables.
- FusionEncoder remains the direct local comparator for MXD494 in the curated project tables.
- FusionEncoder also reports DISORDER723 and MXD494 benchmark results and CAID3 results, reinforcing that PLM-based semantic features and fusion remain current in 2025 methods.

## 2026 Context

- A 2026 review emphasizes that intrinsic disorder prediction is now shaped by protein language models, deep learning, meta-servers, and curated disorder/function databases.
- CAID3 confirms that current community evaluation increasingly emphasizes PLM-era predictors, challenge-style assessment, AUC/Fmax/average precision, and statistical comparisons such as DeLong tests.
- CAID3 also introduced a binding-region-in-IDR subchallenge, which supports treating disorder function prediction as an important future extension rather than a current main claim for this project.

## Manuscript Impact on This Project

The manuscript should position itself against PLM-era methods rather than older profile-only predictors. The direct benchmark claim should remain tied to SL329/MXD494/DISORDER723 point estimates from the curated local comparison table, while CAID3 and recent reviews should be used to motivate:

1. sequence-only and PLM-based representation,
2. transparent low-homology evaluation,
3. AUPR/Fmax reporting under class imbalance,
4. probability calibration and uncertainty as practical reliability features.

## Required Manual Citation Audit Before Submission

Before journal submission, manually verify whether any 2026 method has reported direct SL329, MXD494, or DISORDER723 results exceeding the current local SOTA table. If none is found, keep IDP-EDL and FusionEncoder as the direct benchmark comparators.
