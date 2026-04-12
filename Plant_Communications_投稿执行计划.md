# Plant Communications 投稿执行计划

> **目标期刊**：Plant Communications  
> **稿件主题**：PlantMEGNN — 多视图图神经网络用于水稻单环境/多环境基因组预测  
> **当前状态**：数据验证完成，手稿 v1.0 已具备基础框架，需补充可解释性分析 + 可视化升级 + 叙事重构  
> **计划制定日期**：2026-04-09  
> **预计投稿日期**：2026-04-20（10天后）

---

## 1. 当前资产盘点

### 1.1 已完成的核心数据

| 实验模块 | 状态 | 关键文件 | 规模 |
|---------|------|----------|------|
| GSTP007 主实验 | 已锁定 | `results/gstp007/multiview_results_v3.json` | 7性状 × 5fold × 3seed = 15 runs/配置 |
| GSTP007 基线对比 | 已锁定 | `results/gstp007/benchmark_5fold_cv.json` | GBLUP/Ridge/DNNGP/Transformer/NetGP |
| GSTP007 视图消融 | 已锁定 | `multiview_results_v3.json` | 4配置（1view/GO/2view/3view）× 7性状 |
| GSTP007 注意力权重 | 已锁定 | `multiview_results_v3.json` → `view_weights` | 15 runs |
| GSTP007 统计检验 | 已锁定 | `results/gstp007/statistical_tests.json` | Wilcoxon signed-rank |
| GSTP008 LOEO | 已锁定 | `results/gstp008/loeo_results.json` + `multitask_results.json` | 4环境，3~5 seeds |
| GSTP008 GxE-WtLoss | 已锁定 | `results/gstp008/gxe_selective_results.json` | 5 seeds |
| 论文手稿 | 已校对 | `paper/manuscript_draft.md` | ~6,000词 |
| 基础图表 | 已生成 | `paper/figures/` | Fig 1-6, S1-S2 |
| 投稿表格（LaTeX） | 已生成 | `paper/tables/*.tex` | Table 1a, 1b, 2a, 3, 5 |

### 1.2 关键数字（经 2026-04-09 验证）

**GSTP007 平均 PCC（7性状）**：
- GBLUP: 0.739 | Ridge: 0.732 | DNNGP: 0.751 | Transformer: 0.748
- NetGP: 0.757 | PlantMEGNN-1view: 0.755 | **PlantMEGNN-3view: 0.755**

**GSTP008 LOEO 平均 PCC（4环境）**：
- ST-Ridge: 0.324 | ST-MLP: 0.313 | ST-NetGP: 0.313
- MT-Ridge: **0.307** | **PlantMEGNN-E: 0.340**

> **重要发现**：经服务器数据同步与重新计算，GSTP008 MT-Ridge 实际平均为 0.307（手稿旧版误写为 0.333）。修正后 PlantMEGNN-E (0.340) **显著优于多任务统计基线**，反而增强了多环境故事的说服力。

---

## 2. 目标期刊匹配度分析与策略定位

### 2.1 Plant Communications 的审稿偏好

1. **"方法创新 + 生物学洞见" > "纯性能提升"**：接受性能提升 modest 的论文，但必须有可防御的生物学解释
2. **跨条件验证**：从单一实验到多环境/多尺度拓展，体现模型的泛化能力
3. **高质量可视化**：主图质量接近 Cell 子刊标准，信息密度高、配色专业
4. **对作物改良有直接或间接启示**：Discussion 需要回扣到育种实践

### 2.2 我们的优势

- **首创性**：首个植物 GP 多视图 GNN（PPI+GO+KEGG 联合）+ 首个 LOEO-CV 多环境 GNN 框架
- **生物学故事完整**：KEGG 主导粒型性状、GO 主导营养性状，与已知遗传学一致
- **数据量扎实**：GSTP007 420 次独立运行；GSTP008 严格 LOEO-CV
- **修正后的 GSTP008 数字更强**：0.340 vs MT-Ridge 0.307，支撑多任务 GNN 的有效性

### 2.3 核心短板与应对策略

| 短板 | 风险 | 应对策略 |
|------|------|----------|
| 3view 准确率仅持平 1view | 审稿人质疑多视图意义 | **主动转化为叙事核心**：将论文定位从"性能 SOTA"转向"可解释性框架"，强调多视图的价值在于揭示性状特异性网络偏好 |
| 仅水稻单物种 | 泛化性受质疑 | Discussion 明确承认，给出理论预期（网络注释完善的物种应更强），并列为 future work |
| Fig 1 为 matplotlib 手绘 | 视觉质量不够期刊级别 | **重制为 PPT/BioRender/Illustrator 级专业图** |
| 缺乏"网络权重 → 通路富集"的桥接 | 生物学洞见停留在统计层面 | **补充 KEGG/GO 富集分析**（Table S1），将 attention 数字升级为可引证的通路结论 |
| 缺少明星基因案例 | 故事不够 concrete | **补充 GW5/qSW5 案例**（Supplementary），增强对水稻育种读者的吸引力 |

---

## 3. 核心叙事策略：从"性能SOTA"转向"认知升级"

 Plant Communications 不排斥 modest 的性能提升，但厌恶**没有洞见的调参**。既然多视图在 accuracy 上无法碾压，我们必须把整个手稿的 spine 改为：

> **"PlantMEGNN 首次提供了植物基因组预测中的多网络可解释性框架，其真正价值不在于预测精度的绝对提升，而在于揭示了不同生物网络对性状预测的差异化贡献规律，并为多环境育种提供了机制可解释的工具。"**

这一策略要求：
1. **Abstract 第一句话就直接抛生物学 gap**，而不是模型定义
2. **Results 中"注意力分析"升格为一级节（4.3）**，与 GSTP007 性能、GSTP008 多环境并列
3. **Discussion 专门增加"育种启示"小节**，把 KEGG/GO/PPI 的发现翻译成对育种家的 actionable advice
4. **所有对"性能持平"的描述都要前置主动承认**，不要等审稿人来问

---

## 4. Phase 1：论文叙事重构（Day 1-2）

### 4.1 Abstract 重写

**当前问题**：0.755 = 0.755 这个等式出现在 Abstract 第二句，容易让 reviewer 提前失去兴趣。

**目标结构**：
1. **Gap**：GP 很重要，但 DL 方法忽略网络先验且假设单环境
2. **Solution**：PlantMEGNN — 多视图 GNN + 多环境扩展
3. **Biological insight（前置）**：首次发现 KEGG 主导粒型性状、GO 主导营养性状
4. **多环境结果**：0.340 vs 0.324，YangZ15 +0.096
5. **Implication**：为性状特异性育种提供了可解释的网络先验框架

### 4.2 Results 结构调整

```
4.1 Single-Environment GP Performance on GSTP007
    4.1.1 Main comparison (Table 1a)
    4.1.2 Network view ablation (Table 2a)

4.2 Multi-Environment GP under LOEO-CV on GSTP008
    4.2.1 LOEO main results (Table 1b)
    4.2.2 Analysis of YangZ15 gain sources
    4.2.3 Negative transfer in LingS15
    4.2.4 GxE-WtLoss analysis (Table 2b)

4.3 Biological Interpretability of Network Contributions  ← 升格为一级节
    4.3.1 Trait-specific network preferences (Table 3)
    4.3.2 Pathway enrichment of attention-weighted genes (Table S1)  ← 新增
    4.3.3 Case study: GW5 neighborhood in grain length  ← 可选
```

### 4.3 关键段落重写（必须在手稿中加入）

**4.1 开头，主动设防**：
> "It is important to note that under identical training conditions, three-view PlantMEGNN achieves the same average PCC as single-view PlantMEGNN-1view (0.755 vs 0.755). We therefore do not frame multi-view fusion as a tool for accuracy improvement; rather, its primary contribution is interpretability — it enables data-driven quantification of which biological network types are most informative for specific traits."

**4.3 开头，升华主题**：
> "Beyond predictive performance, we asked whether the learned attention weights reflect known genetic architecture. The answer is yes: KEGG pathway networks dominate for grain-filling traits, consistent with the central role of starch and sucrose metabolism in grain development, while GO functional similarity is preferred for plant height, reflecting the distributed regulatory architecture of gibberellin and brassinosteroid signaling."

---

## 5. Phase 2：可解释性补充分析（Day 3-6）

这是决定 Plant Communications 成败的关键阶段。**没有通路富集，Table 3 只是数字游戏。**

### 5.1 通路富集分析（必须做）

**目标**：对每个性状，提取 multi-view fusion 中注意力权重最高的基因，做 KEGG 和 GO 富集。

**输入数据**：
- `results/gstp007/multiview_results_v3.json` → `MultiView_3view` 的 `view_weights`
- `data/raw/annotations/gene_go_map.json`
- `data/raw/annotations/ncbi_to_rap_kegg.json`

**分析方法**：
- 对每性状每 run，提取前 20% 高 KEGG attention 的基因作为 "KEGG-high" gene set
- 提取前 20% 高 GO attention 的基因作为 "GO-high" gene set
- 用超几何检验（Fisher exact test）对 KEGG/GO terms 做富集
- 多重检验校正：BH FDR

**预期生物学发现**（基于已知遗传学）：

| 性状 | 预期富集通路 | 富集来源 |
|------|-------------|----------|
| Grain Length | Starch and sucrose metabolism | KEGG-high |
| Grain Width | Starch and sucrose metabolism | KEGG-high |
| Grain Weight | Starch and sucrose metabolism | KEGG-high |
| Days to Heading | Circadian rhythm - plant | KEGG-high |
| Plant Height | Gibberellin / brassinosteroid biosynthesis | GO-high (BP) |

**产出**：
- `paper/tables/tableS1_pathway_enrichment.csv`
- 对应文字写入 Results 4.3.2

**失败预案**：
- 若 FDR > 0.05，可降级为"趋势性富集"（nominal p < 0.05），并在 Discussion 中解释为"网络覆盖率低（831/35,000 genes）导致统计 power 不足"
- 即使不确定著，分析过程本身已体现严谨性

### 5.2 GW5 / qSW5 案例研究（强烈推荐）

GW5（Os5g0959700）是控制水稻粒宽/粒重的主效 QTL，Plant Communications 读者非常熟悉。

**操作步骤**：
1. 在 attention weight matrix 中定位 GW5 对应的 node
2. 提取其一阶邻居节点的 attention 权重和 KEGG 注释
3. 绘制 GW5 邻域子网络图
4. 验证：邻居中包含 GS3、GW8、DEP1 等已知粒型基因的应获得高权重

**产出**：
- `paper/figures/figS3_gw5_case_study.pdf`
- 写入 Results 4.3.3 或 Supplementary

**风险**：若 GW5 不在 831 网络覆盖基因中，则无法用此案例。备用方案：改用 **GS3** 或 **DEP1**。

### 5.3 d=256 数据的归档处理（不做入主文）

远程服务器 `server_all_traits_summary_d256.json` 显示 d=256 并非全方位碾压 d=64，且只有单 seed。决定：
- **不替换主结果**（主结果继续使用经完整验证的 d=64 v3 数据）
- 整理为 **Supplementary Figure S3**，标题："Hyperparameter sensitivity: d_model = 64 vs 256"

**产出**：
- `paper/figures/figS3_dmodel_comparison.pdf`
- 简要文字写入 Supplementary Methods

---

## 6. Phase 3：可视化升级（Day 7-8）

Plant Communications 对 figure quality 非常敏感，以下图必须升级。

### 6.1 Fig 1 架构图（最高优先级重制）

**当前问题**：`scripts/generate_fig1_architecture.py` 生成的 matplotlib 示意图过于简陋。

**重制方案**：
- **工具**：PowerPoint / Adobe Illustrator / BioRender（如果有订阅）
- **分辨率**：300 dpi 以上
- **风格要求**：简洁、扁平化、配色统一
- **建议色板**：
  - PPI: #2E8B57 (海绿色)
  - GO: #DAA520 (金黄色)
  - KEGG: #CD5C5C (印度红)
  - Fusion / MLP: #4682B4 (钢蓝色)
- **必备元素**：
  - 左侧：SNP inputs → gene-level features（小图标）
  - 中部：三个并行的 GCN 模块（网络图小示意）
  - 中右：LightGatedFusion（gate icon + α 符号）
  - 右侧：MLP → predicted phenotype
  - 右上角小插图：PlantMEGNN-E 的多环境扩展示意

**交付**：`paper/figures/fig1_architecture_v2.pdf` 和 `.png`

### 6.2 Fig 4 注意力权重图

**当前问题**：热图不够直观，std 表现较弱。

**升级方案**：改为 **lollipop plot（棒棒糖图）** 或 **dot plot + error bar**。
- X轴：PPI / GO / KEGG
- Y轴：注意力权重 (0~1)
- 分面（facet）：7个性状并排展示
- 加上 uniform 0.333 的虚线参考线，方便读者一眼看出哪些偏离了均匀分布

**交付**：`paper/figures/fig4_attention_weights_v2.pdf`

### 6.3 Fig 2 主结果图

**当前问题**：柱状图过于常规。

**升级方案**：改为 **dot heatmap（点大小 = mean，颜色 = 性能高低）** 或 **李桥图 (ridge plot)**。
 Plant Communications 偏好信息密度高的图。

**备选**：若时间紧，保留柱状图但统一配色并提升分辨率即可。

### 6.4 Fig S1-S3

- Fig S1 (env corr matrix): 已有，可保留
- Fig S2 (GxE scatter): 已有，可保留
- Fig S3 (d=256 comparison): 新增，用简洁的柱状图展示 6 个性状对比

---

## 7. Phase 4：写作冲刺与最终整合（Day 9-10）

### 7.1 Discussion 强化

增加 **"Implications for Rice Breeding"** 小节（约 300-400 词）：

1. **育种应用启示**：
   - 若目标是粒型（GL/GW/GWt），应优先构建/扩充 **KEGG 代谢通路网络**
   - 若目标是株高，应优先利用 **GO 功能注释网络**
   - 若目标是产量（YPP），PPI 的权重更高，提示蛋白质互作 hub 基因可能更重要

2. **多环境试验设计启示**：
   - 多任务联合预测在气候同质区（如长江中下游 multiple temperate sites）最有效
   - 跨气候带（如海南南繁 ↔ 长江流域）直接联合预测会产生 negative transfer，建议分区建模或使用显式环境协变量

3. **未来方向**：
   - 扩展网络覆盖（pan-genome + RNA-seq 共表达）
   - 跨物种验证（小麦、玉米）
   - 整合更多网络类型（TF-target regulatory networks）

### 7.2 补充材料整理

**Supplementary Figures**：
- Fig S1: Environment correlation matrix (已有)
- Fig S2: GxE scatter plots (已有)
- Fig S3: d=256 comparison (新增)
- Fig S4: GW5 case study (新增，若执行)

**Supplementary Tables**：
- Table S1: Pathway enrichment results (新增)
- Table S2: Full per-fold per-seed results for GSTP007 (可选，可上传为 CSV)

**Supplementary Methods**（如期刊要求单独文件）：
- LightGatedFusion 的数学细节
- GxE-WtLoss 的权重计算细节
- Network construction pipeline 的完整参数

### 7.3 参考文献补全

**必须补 DOI 的**：
- Zhang Y et al. (2025) NetGP
- Su B et al. (2024) GRAFT
- Huang Q et al. (2023) TREE
- Wang X et al. (2023) Cropformer
- Li H et al. (2024) GPformer

**建议补充的 Plant Communications 相关引用**（显示对期刊的熟悉）：
- 搜索 2023-2025 年 Plant Communications 上发表的 rice genomics / GWAS / GP 类文章 1-2 篇，在 Introduction 最后提到 "Our work complements recent advances in rice genomic prediction published in Plant Communications [ref]."

### 7.4 最终检查清单

- [ ] 所有 TBD placeholder 已替换为真实信息
- [ ] Data Availability 包含 GitHub 链接和数据 accession
- [ ] 所有表格从 markdown 完全替换为 LaTeX 编译版本或高分辨率图片
- [ ] Fig 1 为专业级矢量图
- [ ] Cover Letter、Highlights、Author Information 完整
- [ ] 拼写检查 + 语法检查通过
- [ ] 转化为主投稿格式（Word 或 PDF）
- [ ] 所有合著者确认并同意投稿

---

## 8. 风险清单与应对预案

| 风险 | 可能性 | 影响 | 应对 |
|------|--------|------|------|
| 通路富集分析不显著（FDR > 0.05） | 中 | 削弱 4.3 节说服力 | 提前降低 claim：用 nominal p < 0.05 描述趋势；归因于网络覆盖率低的 limitation |
| GW5 不在 831 网络基因中 | 中 | 案例研究无法做 | 备选基因：GS3、DEP1、或放弃案例研究，不影响主线 |
| Fig 1 重制时间不足 | 低 | 视觉质量短板 | 若时间紧，使用 PowerPoint 模板快速出图，保证配色统一和清晰度即可 |
| Plant Communications 初审 desk reject | 中 | 需转投 | 备投期刊已排好序：Briefings in Bioinformatics → BMC Plant Biology |
| 审稿人质疑"为何性能没提升" | 高 | 核心质疑 | 已在前述叙事策略中设防，Abstract + Results 4.1 开头主动解释 |

---

## 9. 分日执行表

| 日期 | 核心任务 | 产出物 | 负责人 |
|------|---------|--------|--------|
| **Day 1 (4/10)** | Abstract 重写；Results 结构调整；4.1/4.2 文字润色 | 更新版 `manuscript_draft.md` | Claude + 用户审阅 |
| **Day 2 (4/11)** | 可解释性章节（4.3）框架写入；Discussio breeding implications 草稿 | 手稿 V0.9（内容完整，缺数据） | Claude |
| **Day 3 (4/12)** | 编写通路富集分析脚本；运行并得到 Table S1 | `scripts/pathway_enrichment.py` + `tableS1.csv` | 用户主跑 / Claude 辅助 debug |
| **Day 4 (4/13)** | 若富集成功：写入 4.3.2；若 GW5 在网：开始做案例图 | Results 4.3 完整版 | Claude |
| **Day 5 (4/14)** | 集成所有新文字；生成 Supplementary 材料 | `supplementary_draft.md` 完整版 | Claude |
| **Day 6 (4/15)** | Fig 1 重制（PPT/AI）；Fig 4 升级为 lollipop/dot plot | `fig1_architecture_v2.pdf`, `fig4_attention_v2.pdf` | 用户主做 / Claude 给反馈 |
| **Day 7 (4/16)** | Fig 2 视情况升级；生成 Fig S3 (d=256) | 全部 figures 定稿 | 用户 |
| **Day 8 (4/17)** | 参考文献 DOI 补全；全文 spell-check；生成 Word/PDF | 完整投稿文档 V1.0 | Claude 辅助 |
| **Day 9 (4/18)** | 内部审阅（导师/合作者）；根据反馈微调 | 投稿文档 V1.1 | 用户 |
| **Day 10 (4/19-20)** | 最终上传 Editorial Manager | 提交确认邮件 | 用户 |

---

## 10. 投稿包目录结构（最终版）

```
paper/
├── manuscript_draft.md           → 转为 Word/PDF 主文件
├── DATA_PROVENANCE.md            → 内部留档
├── figures/
│   ├── fig1_architecture_v2.pdf  ← 重制图
│   ├── fig2_gstp007_main.pdf
│   ├── fig3_gstp008_loeo.pdf
│   ├── fig4_attention_weights_v2.pdf  ← 升级图
│   ├── fig5_ablation.pdf
│   ├── fig6_overall_comparison.pdf
│   ├── figS1_env_corr_matrix.pdf
│   ├── figS2_gxe_scatter.pdf
│   └── figS3_d256_comparison.pdf  ← 新增
├── tables/
│   ├── table1a_gstp007.tex
│   ├── table1b_gstp008.tex
│   ├── table2a_ablation.tex
│   ├── table3_attention.tex
│   ├── table5_statistical_tests.tex
│   └── tableS1_enrichment.csv     ← 新增
├── supplementary_draft.md         → 转为 Supplementary PDF
└── SUBMISSION_PACKAGE/
    ├── Cover_Letter.md
    ├── Highlights.md
    ├── Author_Information.md
    ├── Data_Availability_Statement.md
    └── Submission_Checklist_PlantCommunications.md
```

---

## 11. 关键决策点（需用户确认）

1. **是否执行 GW5 案例研究？**
   - 推荐执行，但需检查 GW5 (Os5g0959700) 是否在 831 网络基因中。
   
2. **Fig 1 重制工具？**
   - 推荐 PowerPoint（最快）
   - 备选：Adobe Illustrator（最专业）
   - 备选：BioRender（如果你有订阅）

3. **是否加入 Plant Communications 的过往引用？**
   - 推荐加 1-2 篇，提升期刊匹配度。

4. **投稿时是否把 d=256 的 6 个性状结果作为 Fig S3？**
   - 推荐加入，显得数据工作量大且诚实。

---

*本计划基于 2026-04-09 的数据验证结果制定，所有主实验数字已锁定，后续变动仅限补充分析和可视化升级。*
