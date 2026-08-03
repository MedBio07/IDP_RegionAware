# Table 1. Dataset and NR25 leakage-control summary

| dataset | role | proteins | known_residues | disorder_residues | unknown_residues | disorder_fraction_known | sdr_segments | ldr_segments | terminal_segments | internal_segments | nr25_removed_train_proteins | nr25_kept_train_proteins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DM3000 train | train | 3000 | 730804 | 74170 | 0 | 0.101491 | 5427 | 392 | 4183 | 1636 | NA | NA |
| DM1229 validation | validation | 1229 | 305830 | 29082 | 0 | 0.095092 | 2372 | 166 | 1751 | 787 | NA | NA |
| SL329 test | test | 329 | 90836 | 39544 | 89582 | 0.435334 | 464 | 274 | 328 | 410 | 176 | 2824 |
| MXD494 test | test | 494 | 196501 | 44087 | 0 | 0.224360 | 577 | 271 | 517 | 331 | 323 | 2677 |
| DISORDER723 test | test | 723 | 215229 | 13526 | 0 | 0.062845 | 1363 | 60 | 1017 | 406 | 424 | 2576 |
| NR25 train vs SL329 | nr25_train | 2824 | 686451 | 64364 | 0 | 0.093763 | 5144 | 325 | 3941 | 1528 | 176 | 2824 |
| NR25 train vs MXD494 | nr25_train | 2677 | 639506 | 54830 | 0 | 0.085738 | 4904 | 287 | 3729 | 1462 | 323 | 2677 |
| NR25 train vs DISORDER723 | nr25_train | 2576 | 604856 | 59658 | 0 | 0.098632 | 4584 | 292 | 3568 | 1308 | 424 | 2576 |

Note: SL329 contains unknown labels and all `-1` residues are masked during evaluation.
