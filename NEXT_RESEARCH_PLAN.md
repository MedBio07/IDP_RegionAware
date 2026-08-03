# DM3000 IDR 项目下一步具体研究计划

制定日期：2026-07-31  
依据文档：`RESEARCH_STRATEGY_HIGH_IMPACT.md`、`PROJECT_STATUS.md`、`results/literature_test_results/`

## 1. 总目标

围绕 DM3000/DM1229 和 SL329、MXD494、DISORDER723 三个独立测试集，建立一个可复现、低泄漏、可解释的 IDR 残基层面预测研究体系。论文目标不是只证明一个新网络在 AUC 上小幅提升，而是形成以下完整贡献：

1. 严格 NR25 去冗余 benchmark。
2. 面向 SDR/LDR、terminal/internal、低 disorder content、fully disordered 等困难类型的分层评估。
3. PLM 表征与结构/理化/位置先验的 gated fusion。
4. 概率校准与不确定性分析。
5. 可发布的软件、模型、预测文件和评估流程。

建议主线名称：

**Structure-aware, region-aware and uncertainty-calibrated PLM fusion for intrinsic disorder prediction**

## 2. 当前竞争目标

| 测试集 | 当前直接 SOTA | 年份 | AUC | MCC | BACC | 本项目最低目标 |
|---|---|---:|---:|---:|---:|---|
| MXD494 | FusionEncoder | 2025 | 0.842 | 0.492 | 0.774 | AUC > 0.842，MCC >= 0.492 |
| SL329 | IDP-EDL | 2025 | 0.915 | 0.700 | 0.828 | AUC > 0.915，MCC >= 0.700，mask `-1` |
| DISORDER723 | IDP-EDL | 2025 | 0.943 | 0.636 | 0.793 | AUC > 0.943，MCC >= 0.636，并补充 AUPR |

主文必须同时报告普通 DM3000 训练和 NR25 训练结果。若普通训练超过 SOTA 但 NR25 结果明显下降，需要把论文重点转向低同源泛化、困难样本和不确定性，而不是宣称简单性能领先。

## 3. 12 周里程碑

| 阶段 | 时间 | 核心问题 | 主要产出 | Go/No-Go |
|---|---|---|---|---|
| P0 | 第 1 周 | 实验骨架是否可靠 | 数据分层标注、统一指标、实验目录、配置模板 | 能复现数据统计和文献比较表 |
| P1 | 第 2-3 周 | PLM 轻量 baseline 是否接近 SOTA | frozen ESM-2/ProtT5 baseline、三测试集结果、NR25 初版结果 | 至少一个测试集 AUC 距 SOTA <= 0.005，MCC 不明显退步 |
| P2 | 第 4-6 周 | gated fusion 和 region experts 是否有效 | concat/gated/cross-attention 消融，SDR/LDR 与 terminal/internal 分层 | 分层困难样本至少一个维度稳定提升 |
| P3 | 第 7-8 周 | 校准和不确定性是否能形成独立贡献 | ECE、Brier、reliability diagram、uncertainty case study | 校准后 ECE/Brier 改善，AUC/MCC 不明显下降 |
| P4 | 第 9-10 周 | 功能/结构扩展是否值得进入主文 | pLDDT/PAE/功能辅助头可行性实验 | 若有显著收益则进入主模型，否则作为讨论或补充实验 |
| P5 | 第 11-12 周 | 论文结果是否闭环 | 最终结果表、消融图、分层图、runtime/coverage、论文大纲 | 达到投稿最低证据链 |

## 4. P0：实验骨架与数据分层

### 4.1 目标

先建立可靠评估和数据注释，不直接进入模型训练。这个阶段决定后续所有结果是否可信。

### 4.2 建议新增文件

| 路径 | 用途 |
|---|---|
| `scripts/annotate_disorder_regions.py` | 从三行 FASTA 生成 region 注释 |
| `scripts/summarize_disorder_splits.py` | 复现 train/validation/test 数据统计 |
| `scripts/evaluate_stratified_predictions.py` | 在现有评估脚本基础上增加分层指标 |
| `configs/data.yaml` | 统一记录数据路径和测试集名称 |
| `results/experiment_registry.tsv` | 登记每次实验配置、模型、数据、指标和 commit/日期 |
| `results/stratified/` | 保存 SDR/LDR、terminal/internal、disorder content 分层结果 |

### 4.3 Region annotation 规则

| 注释 | 规则 |
|---|---|
| Known residues | label 为 0 或 1；label 为 `-1` 的残基 mask |
| Disorder segment | 连续 label=1 的区间 |
| SDR | disorder segment 长度 < 30 |
| LDR | disorder segment 长度 >= 30 |
| Terminal IDR | disorder segment 与 N 端或 C 端距离 <= 10 aa，或位于序列前/后 10% |
| Internal IDR | 不属于 terminal IDR 的 disorder segment |
| Disorder content bin | 0-5%、5-20%、20-80%、80-100% |
| Length bin | <=200、201-500、501-1000、>1000 aa |

### 4.4 P0 交付物

1. `results/dataset_region_summary.tsv`
2. `results/testset_region_summary.tsv`
3. `results/literature_test_results/current_methods_with_year_results.md` 的复核说明
4. `results/experiment_registry.tsv` 初始模板

## 5. P1：MVP baseline

### 5.1 目标

用最小工程成本验证本项目是否有性能潜力。第一版不要同时做 LoRA、结构、功能和复杂融合，避免无法定位收益来源。

### 5.2 模型配置

| Baseline | 输入 | 模型 | 目的 |
|---|---|---|---|
| B0 | one-hot + 理化 + relative position | 轻量 CNN/BiGRU | 最低工程基线 |
| B1 | frozen ESM-2 embedding | CNN/Transformer head | 检验 ESM-2 表征上限 |
| B2 | frozen ProtT5 embedding | CNN/Transformer head | 对照 IDP-EDL/DisoFLAG/DisorderUnetLM |
| B3 | ESM-2 + terminal position | CNN/Transformer head | 检验 terminal-aware 是否有效 |
| B4 | ESM-2 + ProtT5 | projection + CNN/Transformer head | 检验双 PLM 互补性 |

### 5.3 训练设置

| 项目 | 建议 |
|---|---|
| 训练集 | `data/DM3000_Train.fasta` |
| 验证集 | `data/DM1229_Validation.fasta` |
| 测试集 | SL329、MXD494、DISORDER723 |
| 低同源训练 | 三套 `data/nr25_by_test/DM3000_Train_nr25_vs_*.fasta` |
| loss | weighted BCE 起步；若 recall 低再加 focal loss |
| 阈值 | 只在 DM1229 validation 上选择 |
| 早停指标 | validation MCC 或 BACC；同时记录 AUC |
| 预测输出 | 每个测试集输出 `id,position,score` 或 `id,scores` TSV |
| 评估脚本 | `scripts/evaluate_disorder_predictions.py` |

### 5.4 P1 交付物

| 文件/目录 | 内容 |
|---|---|
| `results/baselines/*.tsv` | 每个 baseline 的主指标 |
| `predictions/baselines/` | 每个模型在 validation/test 上的预测概率 |
| `results/stratified/baselines_*.tsv` | 分层评估结果 |
| `reports/P1_MVP_BASELINE_SUMMARY.md` | P1 实验结论 |

### 5.5 P1 判定标准

继续推进到 P2 的条件：

1. 至少一个目标测试集 AUC 距当前 SOTA 不超过 0.005。
2. 或者整体 AUC 未领先，但 internal IDR、LDR、低 disorder content 之一有稳定优势。
3. NR25 结果没有崩溃，AUC/MCC 下降幅度可以解释。

若都不满足，暂停复杂模型开发，先做错误分析和数据/标签清理。

## 6. P2：Region-aware gated fusion

### 6.1 目标

验证论文核心方法是否真实有效：不是简单拼接更多特征，而是用 region-aware gated fusion 提升困难样本。

### 6.2 模型版本

| 版本 | 结构 | 目的 |
|---|---|---|
| M0 | 最强 P1 baseline | 对照 |
| M1 | ESM-2 + ProtT5 simple concat | 检验简单融合 |
| M2 | gated fusion | 检验门控融合是否优于 concat |
| M3 | gated fusion + SDR/LDR experts | 检验长度专家 |
| M4 | gated fusion + SDR/LDR + terminal/internal experts | 检验区域专家 |
| M5 | M4 + class-balanced sampler/focal loss | 检验低 disorder content 和 DISORDER723 |

### 6.3 必做消融

1. 去掉 ESM-2。
2. 去掉 ProtT5。
3. 去掉 terminal relative position。
4. 去掉 gated fusion，改成 concat。
5. 去掉 SDR/LDR expert。
6. 去掉 terminal/internal expert。
7. 普通 DM3000 train vs NR25 train。

### 6.4 P2 交付物

1. 主结果表：SL329、MXD494、DISORDER723 的 Sn/Sp/BACC/MCC/AUC/AUPR/Fmax。
2. NR25 结果表。
3. 分层结果表：SDR/LDR、terminal/internal、disorder content、length bins。
4. 消融表。
5. 典型失败样本列表：false positive、false negative、边界错误。

### 6.5 P2 判定标准

优先选择最终模型时按以下顺序判断：

1. 三测试集平均 MCC 和 AUPR。
2. 是否超过或接近 SOTA AUC。
3. NR25 低同源稳定性。
4. hard-case 分层是否有可解释优势。
5. 模型复杂度和推理成本。

## 7. P3：概率校准与不确定性

### 7.1 目标

把预测结果从“排序分数”提升为“可解释概率”，形成区别于多数 IDR 预测器的论文贡献。

### 7.2 方法

| 方法 | 用途 |
|---|---|
| Temperature scaling | 首选，简单稳定 |
| Platt scaling | 适合二分类概率后处理 |
| Isotonic regression | 非参数校准，需防止过拟合 |
| MC dropout 或 deep ensemble | 估计 epistemic uncertainty |
| boundary uncertainty | 对 order/disorder 边界附近残基单独分析 |

### 7.3 指标

1. ECE
2. Brier score
3. Negative log likelihood
4. reliability diagram
5. uncertainty vs error enrichment
6. unknown `-1` 邻近区域的不确定性分布

### 7.4 P3 交付物

1. `results/calibration/calibration_metrics.tsv`
2. `figures/calibration/reliability_*.pdf`
3. `figures/calibration/uncertainty_error_enrichment.pdf`
4. `reports/P3_CALIBRATION_SUMMARY.md`

## 8. P4：结构感知与功能扩展

### 8.1 结构感知实验

优先级从低成本到高成本：

1. 使用可获得的 AlphaFold/ESMFold pLDDT。
2. 若有结构文件，再提取 PAE、SASA、二级结构或 missing residues。
3. 若结构特征不全，先只做覆盖率统计和可用子集实验，不强行作为主模型必需输入。

结构特征使用原则：

- pLDDT 低不等于 disorder，只作为辅助输入。
- 必须报告有结构特征时的 coverage。
- 必须有 “without structure feature” 的 light model。

### 8.2 功能扩展实验

可选功能标签来源：

1. DisProt function annotations。
2. CAID binding/linker 数据。
3. MoRF 数据集或已有 MoRF 预测器输出。
4. protein/DNA/RNA binding、flexible linker 标签。

功能头建议：

| 输出头 | 优先级 | 理由 |
|---|---|---|
| protein-binding IDR | 高 | 文献多、功能意义强 |
| DNA/RNA-binding IDR | 中高 | 生物学解释明确 |
| flexible linker | 中 | DisoFLAG/CAID2 已关注 |
| MoRF | 中 | 可做 case study |
| ion/lipid binding | 低 | 标签更稀疏 |

### 8.3 P4 判定标准

结构/功能扩展进入主文的条件：

1. 主指标或 hard-case 指标有稳定提升。
2. 或者指标提升小，但能解释具体生物学案例。
3. coverage 足够，且 full/light 两版模型逻辑清楚。

若不满足，放入补充材料或讨论，不作为标题创新点。

## 9. P5：最终论文证据链

### 9.1 主文结果表

| 表/图 | 内容 |
|---|---|
| Table 1 | 数据集统计和 NR25 去冗余说明 |
| Table 2 | 三测试集主结果，对比 IDP-EDL、FusionEncoder、IDP-Fusion |
| Table 3 | NR25 低同源结果 |
| Table 4 | 消融实验 |
| Figure 1 | 方法框架图 |
| Figure 2 | 三测试集 ROC/PR 曲线 |
| Figure 3 | SDR/LDR、terminal/internal、disorder content 分层结果 |
| Figure 4 | calibration reliability diagrams |
| Figure 5 | case study：功能区域或典型边界错误 |
| Supplementary | runtime/coverage、更多消融、失败样本 |

### 9.2 最终模型锁定规则

1. 测试集只用于最终报告，不用于调参。
2. 阈值只由 DM1229 validation 决定。
3. 若针对某个测试集使用 NR25 train，只能评估对应测试集。
4. 所有预测文件保留，禁止只保留指标。
5. 每个结果表必须能追溯到配置、模型权重和预测 TSV。

## 10. 实验命名与目录规范

建议目录结构：

```text
configs/
  data.yaml
  baseline_esm2.yaml
  fusion_region_aware.yaml
scripts/
  annotate_disorder_regions.py
  summarize_disorder_splits.py
  extract_plm_embeddings.py
  train_disorder_model.py
  predict_disorder_model.py
  evaluate_stratified_predictions.py
models/
  dataset.py
  features.py
  baseline_heads.py
  fusion_models.py
  calibration.py
predictions/
  baselines/
  fusion/
  final/
results/
  baselines/
  fusion/
  stratified/
  calibration/
  final_tables/
figures/
  benchmark/
  stratified/
  calibration/
  case_studies/
reports/
  P1_MVP_BASELINE_SUMMARY.md
  P2_FUSION_ABLATION_SUMMARY.md
  P3_CALIBRATION_SUMMARY.md
  FINAL_MANUSCRIPT_RESULTS.md
```

实验 ID 建议格式：

```text
YYYYMMDD_dataset_model_features_trainset_seed
```

示例：

```text
20260801_all_esm2_terminal_cnn_dm3000_seed1
20260805_sl329_esm2_prott5_gated_nr25_seed1
```

## 11. 每周具体任务

### 第 1 周

1. 编写 region annotation 脚本。
2. 生成所有 split 的 SDR/LDR、terminal/internal、disorder content、length bin 统计。
3. 扩展评估脚本支持 AUPR、Fmax、分层指标。
4. 建立 experiment registry。

### 第 2 周

1. 确认可用 GPU、磁盘和 PLM 权重缓存。
2. 提取 frozen ESM-2 embedding。
3. 训练 B0/B1/B3。
4. 输出 validation 和三测试集预测文件。

### 第 3 周

1. 提取或生成 ProtT5 embedding。
2. 训练 B2/B4。
3. 跑三套 NR25 baseline。
4. 写 `reports/P1_MVP_BASELINE_SUMMARY.md`，决定是否进入 P2。

### 第 4 周

1. 实现 simple concat fusion。
2. 实现 gated fusion。
3. 在 DM3000 上训练 M1/M2。
4. 比较三测试集主指标和分层指标。

### 第 5 周

1. 实现 SDR/LDR experts。
2. 实现 terminal/internal experts。
3. 训练 M3/M4。
4. 汇总 hard-case 分层收益。

### 第 6 周

1. 实验 focal loss、class-balanced sampler、boundary-aware loss。
2. 在 NR25 train 上复训最优 2 个模型。
3. 写 `reports/P2_FUSION_ABLATION_SUMMARY.md`。

### 第 7 周

1. 在 DM1229 validation 上做 temperature scaling、Platt scaling、isotonic regression。
2. 评估 ECE、Brier、NLL。
3. 生成 reliability diagram。

### 第 8 周

1. 加入 uncertainty 输出或 ensemble/dropout。
2. 分析错误富集与 uncertainty 的关系。
3. 分析 SL329 `-1` 邻近区域的 uncertainty。
4. 写 `reports/P3_CALIBRATION_SUMMARY.md`。

### 第 9 周

1. 检查本地是否已有 AlphaFold/ESMFold 结构或 pLDDT 资源。
2. 若有，加入结构置信度特征并做 coverage 统计。
3. 若没有，先做 light model 定稿，不阻塞主线。

### 第 10 周

1. 整理功能标签可用性。
2. 尝试 protein-binding/linker/MoRF 辅助头或外部预测器输出融合。
3. 选择 2-3 个生物学 case study。

### 第 11 周

1. 锁定最终模型、阈值和随机种子。
2. 生成最终普通训练和 NR25 结果。
3. 生成所有主图、主表和补充表。

### 第 12 周

1. 完成 `reports/FINAL_MANUSCRIPT_RESULTS.md`。
2. 写论文 Methods 和 Results 初稿。
3. 整理代码运行说明、环境文件、模型权重和预测输出。

## 12. 第一批应立即创建的脚本

优先顺序：

1. `scripts/annotate_disorder_regions.py`
2. `scripts/evaluate_stratified_predictions.py`
3. `scripts/extract_plm_embeddings.py`
4. `scripts/train_disorder_model.py`
5. `scripts/predict_disorder_model.py`
6. `models/dataset.py`
7. `models/baseline_heads.py`
8. `models/calibration.py`

其中前两个脚本不依赖 GPU，应先完成。这样即使训练尚未开始，也能立刻产出论文需要的数据统计、分层 benchmark 和评估框架。

## 13. 近期两周最小任务清单

| 优先级 | 任务 | 完成标准 |
|---|---|---|
| P0-1 | region annotation | 生成所有 split 的 segment-level TSV |
| P0-2 | stratified evaluation | 给定任意预测 TSV，能输出 overall + stratified metrics |
| P0-3 | experiment registry | 每个实验有 ID、配置、数据、指标、预测路径 |
| P1-1 | ESM-2 embedding | train/validation/test 可读取 embedding 缓存 |
| P1-2 | baseline model | frozen ESM-2 + terminal position 可训练、可预测 |
| P1-3 | literature comparison | 自动合并本项目结果与文献 SOTA 表 |

两周结束时必须回答三个问题：

1. 当前 baseline 与 IDP-EDL/FusionEncoder 的差距是多少？
2. 差距主要来自哪个测试集、哪类区域或哪种 disorder content？
3. 下一步是继续模型融合，还是先修数据/标签/评估？

## 14. 暂不建议立即做的事项

1. 不建议一开始就全量 LoRA fine-tuning；先用 frozen embedding 判断信号。
2. 不建议同时引入过多外部预测器；会导致贡献归因困难。
3. 不建议只追求测试集最高 AUC；阈值、MCC、AUPR、NR25 和分层结果更重要。
4. 不建议在没有功能标签整理的情况下把功能预测写进标题。
5. 不建议用测试集选模型或阈值；这会削弱投稿可信度。

## 15. 下一步执行建议

马上开始 P0：

1. 创建 `scripts/annotate_disorder_regions.py`，先完成 region annotations。
2. 创建 `scripts/evaluate_stratified_predictions.py`，把 overall 指标扩展到分层指标。
3. 生成 `results/dataset_region_summary.tsv` 和 `results/testset_region_summary.tsv`。
4. 再进入 ESM-2/ProtT5 embedding 和 baseline 训练。

这个顺序最稳：先把论文级评估框架搭好，再投入 GPU 训练，避免训练完成后才发现指标或分层定义不一致。
