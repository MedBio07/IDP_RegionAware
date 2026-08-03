# Supplementary Material Draft

## Supplementary Methods

### Region annotation

Short disordered regions were defined as continuous disordered segments shorter than 30 residues; long disordered regions were defined as segments of at least 30 residues. Terminal IDRs were segments near the N- or C-terminus using the project annotation rule, while the remaining disordered segments were treated as internal IDRs. Stratified analyses also included residue zones, protein length bins and disorder-content bins.

### Protein-level bootstrap and paired permutation

Protein-level resampling was used to avoid treating residues from the same protein as independent. Each bootstrap replicate sampled proteins with replacement and recomputed AUC for both the ESM2-t12 and ESM2-t33 RegionAwareTCN ensembles. Paired permutation randomly swapped t12 and t33 predictions within proteins under the null hypothesis of no paired model difference.

## Supplementary Table S1. Key paper-level ablation deltas

# Table 4. Paper-level key ablation deltas

| axis | dataset | comparison | auc_delta | aupr_delta | mcc_delta | ece_delta | brier_delta | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | SL329 | RegionAwareTCN_aux_t12_3seed minus GenericTCN_t12_3seed | -0.001532 | -0.003458 | -0.012859 | NA | NA | Region-aware supervision helps MXD494/DISORDER723 but is not uniformly beneficial for SL329. |
| representation | SL329 | ESM2-t33 RegionAwareTCN minus ESM2-t12 RegionAwareTCN | 0.004055 | 0.002627 | 0.031063 | NA | NA | ESM2-t33 is the decisive ranking-performance upgrade, especially on DISORDER723. |
| calibration | SL329 | Platt-calibrated t33 minus raw t33 | 0.000000 | 0.000000 | 0.000000 | -0.009123 | -0.015659 | Platt preserves ranking metrics and improves probability quality. |
| homology | SL329 | NR25-vs-SL329 t33 minus full-train t33 | -0.002266 | -0.005537 | -0.003297 | NA | NA | NR25 remains competitive but is not uniformly SOTA. |
| architecture | MXD494 | RegionAwareTCN_aux_t12_3seed minus GenericTCN_t12_3seed | 0.006270 | 0.052391 | 0.010291 | NA | NA | Region-aware supervision helps MXD494/DISORDER723 but is not uniformly beneficial for SL329. |
| representation | MXD494 | ESM2-t33 RegionAwareTCN minus ESM2-t12 RegionAwareTCN | 0.004781 | 0.004664 | 0.002013 | NA | NA | ESM2-t33 is the decisive ranking-performance upgrade, especially on DISORDER723. |
| calibration | MXD494 | Platt-calibrated t33 minus raw t33 | 0.000000 | 0.000000 | 0.000000 | -0.156019 | -0.072682 | Platt preserves ranking metrics and improves probability quality. |
| homology | MXD494 | NR25-vs-MXD494 t33 minus full-train t33 | -0.016548 | -0.018969 | -0.015936 | NA | NA | NR25 remains competitive but is not uniformly SOTA. |
| architecture | DISORDER723 | RegionAwareTCN_aux_t12_3seed minus GenericTCN_t12_3seed | 0.001908 | 0.006036 | 0.002509 | NA | NA | Region-aware supervision helps MXD494/DISORDER723 but is not uniformly beneficial for SL329. |
| representation | DISORDER723 | ESM2-t33 RegionAwareTCN minus ESM2-t12 RegionAwareTCN | 0.021479 | 0.064740 | 0.050966 | NA | NA | ESM2-t33 is the decisive ranking-performance upgrade, especially on DISORDER723. |
| calibration | DISORDER723 | Platt-calibrated t33 minus raw t33 | 0.000000 | 0.000000 | 0.000000 | -0.170328 | -0.061230 | Platt preserves ranking metrics and improves probability quality. |
| homology | DISORDER723 | NR25-vs-DISORDER723 t33 minus full-train t33 | -0.007736 | -0.038080 | -0.009357 | NA | NA | NR25 remains competitive but is not uniformly SOTA. |

## Supplementary Table S2. Protein-level AUC bootstrap and paired permutation

```tsv
dataset	proteins	known_residues	positives	negatives	reference_variant	candidate_variant	reference_auc	reference_auc_ci_low	reference_auc_ci_high	candidate_auc	candidate_auc_ci_low	candidate_auc_ci_high	auc_delta	auc_delta_ci_low	auc_delta_ci_high	paired_bootstrap_p_delta_le_0	paired_permutation_p_one_sided	sota_method	sota_auc	candidate_gap_vs_sota	candidate_auc_ci_low_gt_sota	bootstrap_p_candidate_le_sota	n_bootstrap	n_permutation
SL329	329	90836	39544	51292	p2_region_aware_tcn_3seed_ensemble	p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	0.915272	0.892487	0.933677	0.919327	0.895622	0.938032	0.004055	-0.002693	0.011198	0.115884	0.169661	IDP-EDL	0.915000	0.004327	0	0.363636	1000	500
MXD494	494	196501	44087	152414	p2_region_aware_tcn_3seed_ensemble	p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	0.845856	0.818083	0.871643	0.850637	0.823035	0.874115	0.004781	-0.003227	0.012471	0.121878	0.177645	FusionEncoder	0.842000	0.008637	0	0.280719	1000	500
DISORDER723	723	215229	13526	201703	p2_region_aware_tcn_3seed_ensemble	p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	0.923132	0.913105	0.933151	0.944611	0.933356	0.954116	0.021479	0.013868	0.028520	0.000999	0.001996	IDP-EDL	0.943000	0.001611	0	0.352647	1000	500
```

## Supplementary Table S3. Hard-case stratified t33 performance

```tsv
dataset	stratum_group	stratum	display_stratum	residues	positives	auc	aupr	mcc	fmax	sn	sp
SL329	overall	all_known	Overall	90836.000000	39544.000000	0.919327	0.918972	0.736481	0.842101	0.774150	0.942759
SL329	positive_region_length_type	SDR	SDR	56520.000000	5228.000000	0.847277	0.479613	0.441944	0.495770	0.513772	0.942759
SL329	positive_region_length_type	LDR	LDR	85608.000000	34316.000000	0.930304	0.923337	0.772210	0.857112	0.813819	0.942759
SL329	positive_region_location	terminal	Terminal IDR	78842.000000	27550.000000	0.932671	0.916824	0.784318	0.857968	0.828857	0.942759
SL329	positive_region_location	internal	Internal IDR	63286.000000	11994.000000	0.888676	0.733146	0.617910	0.685368	0.648491	0.942759
SL329	residue_zone	middle	Middle residues	72021.000000	29495.000000	0.913831	0.910175	0.729299	0.828574	0.749992	0.950665
SL329	protein_disorder_content_bin	0-5	0-5% disorder proteins	3249.000000	103.000000	0.909221	0.551004	0.523258	0.590674	0.592233	0.979975
SL329	protein_length_bin	>1000	>1000 aa proteins	21751.000000	11191.000000	0.950259	0.963444	0.790382	0.891871	0.830578	0.955966
MXD494	overall	all_known	Overall	196501.000000	44087.000000	0.850637	0.604872	0.499439	0.623957	0.742350	0.812379
MXD494	positive_region_length_type	SDR	SDR	159095.000000	6681.000000	0.802136	0.134532	0.215152	0.213516	0.621314	0.812379
MXD494	positive_region_length_type	LDR	LDR	189820.000000	37406.000000	0.859299	0.588997	0.499716	0.604460	0.763968	0.812379
MXD494	positive_region_location	terminal	Terminal IDR	187463.000000	35049.000000	0.864659	0.589699	0.505404	0.601573	0.780821	0.812379
MXD494	positive_region_location	internal	Internal IDR	161452.000000	9038.000000	0.796259	0.149714	0.228756	0.252527	0.593162	0.812379
MXD494	residue_zone	middle	Middle residues	156675.000000	31377.000000	0.855893	0.610984	0.497162	0.610754	0.706122	0.843286
MXD494	protein_disorder_content_bin	0-5	0-5% disorder proteins	40506.000000	1106.000000	0.831777	0.086302	0.208486	0.175725	0.666365	0.829442
MXD494	protein_length_bin	>1000	>1000 aa proteins	32767.000000	4687.000000	0.829200	0.417708	0.351999	0.482371	0.814807	0.676425
DISORDER723	overall	all_known	Overall	215229.000000	13526.000000	0.944611	0.690249	0.610462	0.639095	0.654000	0.972866
DISORDER723	positive_region_length_type	SDR	SDR	211949.000000	10246.000000	0.948319	0.649218	0.588243	0.618293	0.669042	0.972866
DISORDER723	positive_region_length_type	LDR	LDR	204983.000000	3280.000000	0.933028	0.498690	0.388454	0.502469	0.607012	0.972866
DISORDER723	positive_region_location	terminal	Terminal IDR	211281.000000	9578.000000	0.968186	0.761451	0.666365	0.707830	0.800793	0.972866
DISORDER723	positive_region_location	internal	Internal IDR	205651.000000	3948.000000	0.887416	0.152541	0.210033	0.226462	0.297872	0.972866
DISORDER723	residue_zone	middle	Middle residues	171407.000000	5297.000000	0.916962	0.443969	0.422864	0.440755	0.375307	0.988556
DISORDER723	protein_disorder_content_bin	0-5	0-5% disorder proteins	125275.000000	3301.000000	0.950739	0.549724	0.531958	0.558656	0.679188	0.976634
DISORDER723	protein_length_bin	>1000	>1000 aa proteins	5474.000000	124.000000	0.920080	0.332646	0.395774	0.422442	0.540323	0.972897
```

## Supplementary Table S4. Full Platt calibration metrics

```tsv
dataset	auc	aupr	mcc	raw_brier	platt_brier	brier_delta	raw_ece	platt_ece	ece_delta	raw_nll	platt_nll	nll_delta
DM1229_Validation	0.927369	0.712355	0.616901	0.099110	0.046325	-0.052785	0.175170	0.005851	-0.169319	0.331946	0.166137	-0.165809
SL329	0.919327	0.918972	0.736481	0.126544	0.110885	-0.015659	0.109384	0.100261	-0.009123	0.399929	0.401966	0.002037
MXD494	0.850637	0.604872	0.499439	0.219616	0.146934	-0.072682	0.256921	0.100902	-0.156019	0.786378	0.457715	-0.328663
DISORDER723	0.944611	0.690249	0.610462	0.093709	0.032479	-0.061230	0.187936	0.017608	-0.170328	0.317318	0.120267	-0.197051
```

## Supplementary Table S5. Platt uncertainty-error enrichment

```tsv
experiment_id	method	dataset	top_uncertain_fraction	selected_residues	overall_error_rate	top_uncertain_error_rate	error_enrichment	mean_uncertainty_top	mean_uncertainty_all
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	DM1229_Validation	0.010000	3059	0.063254	0.557372	8.811630	0.997884	0.239919
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	DM1229_Validation	0.050000	15292	0.063254	0.472992	7.477657	0.952817	0.239919
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	DM1229_Validation	0.100000	30583	0.063254	0.354020	5.596795	0.859785	0.239919
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	DM1229_Validation	0.200000	61166	0.063254	0.236144	3.733264	0.696964	0.239919
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	SL329	0.010000	909	0.130642	0.312431	2.391506	0.999262	0.412931
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	SL329	0.050000	4542	0.130642	0.339938	2.602060	0.982291	0.412931
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	SL329	0.100000	9084	0.130642	0.335205	2.565826	0.941887	0.412931
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	SL329	0.200000	18168	0.130642	0.276255	2.114595	0.845460	0.412931
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	MXD494	0.010000	1966	0.203332	0.593082	2.916814	0.999282	0.373414
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	MXD494	0.050000	9826	0.203332	0.589151	2.897480	0.981666	0.373414
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	MXD494	0.100000	19651	0.203332	0.511170	2.513963	0.935010	0.373414
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	MXD494	0.200000	39301	0.203332	0.466171	2.292658	0.828608	0.373414
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	DISORDER723	0.010000	2153	0.047173	0.605667	12.839259	0.997270	0.225927
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	DISORDER723	0.050000	10762	0.047173	0.480394	10.183662	0.941828	0.225927
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	DISORDER723	0.100000	21523	0.047173	0.328114	6.955547	0.836643	0.225927
p4_6_region_aware_tcn_esm2_t33_3seed_ensemble	platt	DISORDER723	0.200000	43046	0.047173	0.200414	4.248478	0.663742	0.225927
```

## Supplementary Figures

Supplementary Figure S1. Reliability diagrams for raw, temperature-scaled, Platt-calibrated and isotonic-calibrated t33 ensemble predictions. Source files: `figures/p4_6/calibration/reliability_p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_*.pdf`.
