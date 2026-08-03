# Reviewer Risk Prebuttal

## Risk 1: The method is not architecturally novel enough

Response strategy: frame the contribution as a calibrated, leakage-aware and hard-case-stratified PLM-era benchmark framework, not as a radically new neural architecture. Emphasize the sequence-only input design, NR25 split, protein-level statistics, hard-case analysis and calibration.

## Risk 2: AUC gains over SOTA are small

Response strategy: acknowledge this explicitly. The manuscript claims AUC-level SOTA point performance, not broad statistical dominance over external predictors. The strongest statistical evidence is t33 versus t12 on DISORDER723.

## Risk 3: NR25 is not uniformly SOTA

Response strategy: present NR25 as robustness evidence and a leakage-control boundary. State that the model remains competitive but does not uniformly beat SOTA under NR25.

## Risk 4: DISORDER723 MCC remains below IDP-EDL

Response strategy: keep MCC in the main table and state the limitation. Do not hide it. Use AUPR, Fmax, calibration and hard-case analysis to show additional value beyond a single MCC number.

## Risk 5: External SOTA predictions are unavailable for paired testing

Response strategy: explain that paired tests are only possible for local t12 versus t33 predictions. External methods are compared by curated aggregate metrics. Avoid claiming statistical superiority over external SOTA.

## Risk 6: Structure/function features were not used

Response strategy: explain the local audit. Structure features were not used because local coverage was zero and PDB-chain-like benchmark identifiers introduce leakage risk. Function heads are future work pending curated residue-level labels.

## Risk 7: Data and code availability

Response strategy: before submission, package scripts, environment, model weights, prediction files, evaluation scripts, calibration scripts and instructions for ESM2 embedding extraction or download.
