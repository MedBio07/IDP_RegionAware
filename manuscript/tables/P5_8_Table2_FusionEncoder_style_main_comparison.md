# P5.8 Table 2. FusionEncoder-style main benchmark comparison

## SL329

| Predictor | Sn | Sp | BACC | MCC | AUC | Rank AUC | Rank BACC | Rank MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **RegionAdapterMoETCN (this work)** | 0.781 | 0.936 | **0.858** | **0.733** | **0.921** | **1** | **1** | **1** |
| IDP-EDL | 0.690 | 0.970 | 0.828 | 0.700 | 0.915 | 2 | 3 | 2 |
| IDP-Fusion | 0.729 | 0.933 | 0.831 | 0.685 | 0.908 | 3 | 2 | 3 |
| DeepIDP-2L | 0.730 | 0.930 | 0.828 | 0.680 | 0.904 | 4 | 3 | 4 |
| SPOT-Disorder | 0.650 | 0.960 | 0.805 | 0.650 | 0.901 | 5 | 9 | 6 |

## MXD494

| Predictor | Sn | Sp | BACC | MCC | AUC | Rank AUC | Rank BACC | Rank MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **RegionAdapterMoETCN (this work)** | 0.752 | 0.808 | **0.780** | **0.502** | **0.850** | **1** | **1** | **1** |
| FusionEncoder | 0.742 | 0.806 | 0.774 | 0.492 | 0.842 | 2 | 2 | 2 |
| IDP-EDL | 0.679 | 0.843 | 0.761 | 0.488 | 0.837 | 3 | 4 | 3 |
| IDP-Fusion | 0.712 | 0.808 | 0.760 | 0.470 | 0.834 | 4 | 5 | 5 |
| IDP-Seq2Seq | 0.743 | 0.791 | 0.767 | 0.475 | 0.825 | 5 | 3 | 4 |
| DeepIDP-2L | 0.737 | 0.776 | 0.757 | 0.452 | 0.825 | 5 | 6 | 8 |

## DISORDER723

| Predictor | Sn | Sp | BACC | MCC | AUC | Rank AUC | Rank BACC | Rank MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **RegionAdapterMoETCN (this work)** | 0.659 | 0.973 | 0.816 | 0.613 | **0.945** | **1** | 2 | 2 |
| IDP-EDL | 0.603 | 0.984 | 0.793 | **0.636** | 0.943 | 2 | 4 | **1** |
| FusionEncoder | 0.695 | 0.955 | **0.825** | 0.564 | 0.932 | 3 | **1** | 3 |
| IDP-Fusion | 0.625 | 0.962 | 0.793 | 0.539 | 0.917 | 4 | 4 | 5 |
| DeepIDP-2L | 0.615 | 0.962 | 0.789 | 0.529 | 0.914 | 5 | 6 | 8 |
| AUCpreD | 0.580 | 0.974 | 0.777 | 0.564 | 0.914 | 5 | 8 | 3 |

Note: ranks follow the FusionEncoder convention and are computed within each dataset after adding this work to the curated local direct-benchmark table. Higher is better for AUC, BACC and MCC.
