# Current Method Results On SL329, MXD494, And DISORDER723

Source scope: IDP-Fusion main paper and supplementary Tables S2-S4, updated with direct target-dataset results from IDP-EDL (2025) and FusionEncoder (2025). Metrics are residue-level disorder prediction metrics. For SL329, residues labeled `-1` in the local FASTA are unannotated/ignored; known residues match the supplement counts.

## Dataset Label Counts

| Dataset | Proteins | Known residues | Disordered | Ordered | Unknown/masked |
|---|---:|---:|---:|---:|---:|
| MXD494 | 494 | 196501 | 44087 | 152414 | 0 |
| SL329 | 329 | 90836 | 39544 | 51292 | 89582 |
| DISORDER723 | 723 | 215229 | 13526 | 201703 | 0 |

## Best AUC By Dataset

| Dataset | Best AUC method | Year | AUC | Sn | Sp | BACC | MCC | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MXD494 | FusionEncoder | 2025 | 0.842 | 0.742 | 0.806 | 0.774 | 0.492 | AUC top in the collected direct target-dataset table. |
| SL329 | IDP-EDL | 2025 | 0.915 | 0.690 | 0.970 | 0.828 | 0.700 | AUC top in the collected direct target-dataset table. |
| DISORDER723 | IDP-EDL | 2025 | 0.943 | 0.603 | 0.984 | 0.793 | 0.636 | AUC top in the collected direct target-dataset table. |

## Full Results

### MXD494

| Method | Sn | Sp | BACC | MCC | AUC | AUC rank | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| FusionEncoder | 0.742 | 0.806 | 0.774 | 0.492 | 0.842 | 1 | FusionEncoder Table 3 |
| IDP-EDL | 0.679 | 0.843 | 0.761 | 0.488 | 0.837 | 2 | IDP-EDL Table 4 |
| IDP-Fusion | 0.712 | 0.808 | 0.760 | 0.470 | 0.834 | 3 | IDP-Fusion main paper Table 1 |
| IDP-Seq2Seq(13) | 0.743 | 0.791 | 0.767 | 0.475 | 0.825 | 4 | IDP-Fusion supplementary Table S2 |
| DeepIDP-2L(18) | 0.737 | 0.776 | 0.757 | 0.452 | 0.825 | 4 | IDP-Fusion supplementary Table S2 |
| MFDp(9) | 0.746 | 0.768 | 0.757 | 0.451 | 0.821 | 5 | IDP-Fusion supplementary Table S2 |
| RFPR-IDP(17) | 0.749 | 0.758 | 0.754 | 0.442 | 0.821 | 5 | IDP-Fusion supplementary Table S2 |
| MD(16) | 0.673 | 0.813 | 0.743 | 0.444 | 0.821 | 5 | IDP-Fusion supplementary Table S2 |
| SPOT-Disorder(7) | 0.626 | 0.851 | 0.739 | 0.457 | 0.813 | 6 | IDP-Fusion supplementary Table S2 |
| SPINE-D(15) | 0.787 | 0.698 | 0.742 | 0.411 | 0.803 | 7 | IDP-Fusion supplementary Table S2 |
| DISOPRED3(6) | 0.622 | 0.820 | 0.721 | 0.410 | 0.800 | 8 | IDP-Fusion supplementary Table S2 |
| AUCpreD(8) | 0.521 | 0.881 | 0.701 | 0.411 | 0.800 | 8 | IDP-Fusion supplementary Table S2 |
| IDP-FSP(20) | 0.670 | 0.831 | 0.751 | 0.465 | 0.794 | 9 | IDP-Fusion supplementary Table S2 |
| PONDR-FIT(21) | 0.631 | 0.821 | 0.726 | 0.419 | 0.790 | 10 | IDP-Fusion supplementary Table S2 |
| IUPred-long(12) | 0.581 | 0.841 | 0.711 | 0.405 | 0.784 | 11 | IDP-Fusion supplementary Table S2 |
| DISOPRED2(10) | 0.647 | 0.800 | 0.724 | 0.406 | 0.781 | 12 | IDP-Fusion supplementary Table S2 |
| IUPred-short(12) | 0.522 | 0.866 | 0.694 | 0.389 | 0.781 | 12 | IDP-Fusion supplementary Table S2 |
| DISpro(19) | 0.303 | 0.940 | 0.622 | 0.318 | 0.775 | 13 | IDP-Fusion supplementary Table S2 |
| RONN(22) | 0.664 | 0.754 | 0.709 | 0.368 | 0.764 | 14 | IDP-Fusion supplementary Table S2 |
| Ucon(23) | 0.554 | 0.787 | 0.671 | 0.313 | 0.741 | 15 | IDP-Fusion supplementary Table S2 |
| NORSnet(24) | 0.532 | 0.829 | 0.681 | 0.347 | 0.738 | 16 | IDP-Fusion supplementary Table S2 |
| PROFbval(25) | 0.835 | 0.387 | 0.611 | 0.196 | 0.697 | 17 | IDP-Fusion supplementary Table S2 |

### SL329

| Method | Sn | Sp | BACC | MCC | AUC | AUC rank | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| IDP-EDL | 0.690 | 0.970 | 0.828 | 0.700 | 0.915 | 1 | IDP-EDL Table 5 |
| IDP-Fusion | 0.729 | 0.933 | 0.831 | 0.685 | 0.908 | 2 | IDP-Fusion main paper Table 1 |
| DeepIDP-2L(18) | 0.730 | 0.930 | 0.828 | 0.680 | 0.904 | 3 | IDP-Fusion supplementary Table S3 |
| SPOT-Disorder(7) | 0.650 | 0.960 | 0.805 | 0.650 | 0.901 | 4 | IDP-Fusion supplementary Table S3 |
| IDP-Seq2Seq(13) | 0.710 | 0.920 | 0.822 | 0.670 | 0.899 | 5 | IDP-Fusion supplementary Table S3 |
| AUCpreD(8) | 0.630 | 0.960 | 0.795 | 0.640 | 0.887 | 6 | IDP-Fusion supplementary Table S3 |
| SPINE-D(15) | 0.820 | 0.800 | 0.815 | 0.610 | 0.886 | 7 | IDP-Fusion supplementary Table S3 |
| DISOPRED3(6) | 0.670 | 0.920 | 0.796 | 0.620 | 0.880 | 8 | IDP-Fusion supplementary Table S3 |
| RFPR-IDP(17) | 0.780 | 0.840 | 0.809 | 0.620 | 0.879 | 9 | IDP-Fusion supplementary Table S3 |
| MFDp(9) | 0.880 | 0.620 | 0.750 | 0.510 | 0.873 | 10 | IDP-Fusion supplementary Table S3 |
| IDP-FSP(20) | 0.750 | 0.890 | 0.821 | 0.650 | 0.864 | 11 | IDP-Fusion supplementary Table S3 |
| MD(16) | 0.660 | 0.890 | 0.775 | 0.580 | 0.864 | 11 | IDP-Fusion supplementary Table S3 |
| DISOPRED2(10) | 0.690 | 0.900 | 0.795 | 0.590 | 0.858 | 12 | IDP-Fusion supplementary Table S3 |
| DISOClust(11) | 0.810 | 0.700 | 0.755 | 0.510 | 0.846 | 13 | IDP-Fusion supplementary Table S3 |
| PONDR-FIT(21) | 0.610 | 0.910 | 0.760 | 0.550 | 0.843 | 14 | IDP-Fusion supplementary Table S3 |
| IUpred-long(12) | 0.600 | 0.920 | 0.760 | 0.550 | 0.839 | 15 | IDP-Fusion supplementary Table S3 |
| IUpred-short(12) | 0.500 | 0.940 | 0.720 | 0.500 | 0.829 | 16 | IDP-Fusion supplementary Table S3 |
| NORSnet(24) | 0.540 | 0.920 | 0.730 | 0.510 | 0.815 | 17 | IDP-Fusion supplementary Table S3 |
| Ucon(23) | 0.590 | 0.810 | 0.700 | 0.420 | 0.779 | 18 | IDP-Fusion supplementary Table S3 |
| PONDERVL-XT(26) | 0.590 | 0.780 | 0.685 | 0.380 | 0.755 | 19 | IDP-Fusion supplementary Table S3 |

### DISORDER723

| Method | Sn | Sp | BACC | MCC | AUC | AUC rank | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| IDP-EDL | 0.603 | 0.984 | 0.793 | 0.636 | 0.943 | 1 | IDP-EDL Table 6 |
| FusionEncoder | 0.695 | 0.955 | 0.825 | 0.564 | 0.932 | 2 | FusionEncoder Table 2 |
| IDP-Fusion | 0.625 | 0.962 | 0.793 | 0.539 | 0.917 | 3 | IDP-Fusion main paper Table 1 |
| DeepIDP-2L(18) | 0.615 | 0.962 | 0.789 | 0.529 | 0.914 | 4 | IDP-Fusion supplementary Table S4 |
| AUCpreD(8) | 0.580 | 0.974 | 0.777 | 0.564 | 0.914 | 4 | IDP-Fusion supplementary Table S4 |
| IDP-Seq2Seq(13) | 0.618 | 0.955 | 0.787 | 0.511 | 0.906 | 5 | IDP-Fusion supplementary Table S4 |
| DISOPRED3(6) | 0.452 | 0.986 | 0.719 | 0.536 | 0.899 | 6 | IDP-Fusion supplementary Table S4 |
| RFPR-IDP(17) | 0.522 | 0.974 | 0.748 | 0.517 | 0.898 | 7 | IDP-Fusion supplementary Table S4 |
| SPOT-Disorder(7) | 0.470 | 0.983 | 0.726 | 0.531 | 0.898 | 7 | IDP-Fusion supplementary Table S4 |
| SPINE-D(15) | 0.779 | 0.840 | 0.810 | 0.376 | 0.891 | 8 | IDP-Fusion supplementary Table S4 |
| IUPred-Short(12) | 0.495 | 0.943 | 0.719 | 0.382 | 0.810 | 9 | IDP-Fusion supplementary Table S4 |
| IUPred-Long(12) | 0.298 | 0.949 | 0.623 | 0.247 | 0.721 | 10 | IDP-Fusion supplementary Table S4 |

## Files

- Full TSV: `/data8T/IDPs_DM3000Train/results/literature_test_results/current_methods_on_SL329_MXD494_DISORDER723.tsv`
- Method-year TSV: `/data8T/IDPs_DM3000Train/results/literature_test_results/current_methods_with_year_results.tsv`
- Method-year Markdown: `/data8T/IDPs_DM3000Train/results/literature_test_results/current_methods_with_year_results.md`
- Recent method coverage TSV: `/data8T/IDPs_DM3000Train/results/literature_test_results/recent_2023_2026_methods_scope.tsv`
- Recent method coverage Markdown: `/data8T/IDPs_DM3000Train/results/literature_test_results/recent_2023_2026_methods_scope.md`
- Supplementary DOCX: `/data8T/IDPs_DM3000Train/references/IDP-Fusion_2023_BMC_Biology_supplementary_tables.docx`
- Three-test-set total TSV: `/data8T/IDPs_DM3000Train/results/literature_test_results/three_test_sets_total_with_latest_methods.tsv`
- Three-test-set total Markdown: `/data8T/IDPs_DM3000Train/results/literature_test_results/three_test_sets_total_with_latest_methods.md`
