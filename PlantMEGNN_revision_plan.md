# PlantMEGNN 论文修订行动计划

> 本文件为结构化修订指令，供 Claude Code 辅助执行。
> **当前投稿目标：Plant Phenomics**
> 原备选期刊 Briefings in Bioinformatics 已调整为备选方案

---

## 投稿策略

```
主投：Plant Phenomics
  └─ 若被拒或major revision后决定放弃 → 转投 Briefings in Bioinformatics

Plant Phenomics 定位重点：
  - 强化水稻育种应用背景，弱化纯方法创新声明
  - 突出跨群体泛化能力（GSTP007/008/009三群体验证）
  - Abstract 改写为育种实践导向语言
  - Discussion 新增"Practical implications for rice breeding"小节
  - 弱化"注意力权重无预测优势"的消极发现，转为中性表述
```

---

## 已完成工作 ✅

### 论文重写
- [x] 创建 `paper_plant_phenomics/main.tex` — 完整Plant Phenomics版本
- [x] 复制 `references.bib` — 参考文献文件
- [x] 创建 `PAPER_REVISION_SUMMARY.md` — 修订对比总结

### 核心调整
- [x] **标题修改**: 从 "Explainable Graph Neural Network" → "Improves Genomic Prediction Accuracy for Rice Breeding"
- [x] **Abstract重写**: 应用导向，强调育种价值
- [x] **Introduction重构**: 增加水稻粮食安全背景，压缩技术细节
- [x] **Discussion新增**: "Practical Implications for Rice Breeding"小节
- [x] **结果表述**: 调整消融研究的语言（从"negligible"到"functionally redundant"）

---

## 待完成工作

### 🔴 必须完成（投稿资格）

#### 1. Figure 1 架构图
```
位置: paper_plant_phenomics/figures/fig1.png

内容要求:
- SNP Input (5,000 SNPs)
- ↓
- Gene-Level Aggregation (831 genes)
- ↓
- 3-View GCN (PPI / GO / KEGG networks)
- ↓
- LightGatedFusion (attention weights)
- ↓
- Prediction Head → Trait Value

建议工具: PowerPoint / Adobe Illustrator / TikZ
格式: PDF or high-res PNG (300 dpi)
尺寸: 单栏宽度 (~8cm) 或双栏 (~17cm)
```

#### 2. 数据登记号
```
替换 main.tex 中两处 "[accession TBD]":
- Data Availability 段落
- 如有需要，补充材料中

当前状态: 需要根据实际数据获取情况填写
```

#### 3. 基金号确认
```
位置: main.tex Funding section

当前内容:
"National Natural Science Foundation of China (32472007, 62301006, 62301008)
and the Natural Science Foundation of Anhui Province (2308085MF217, 2308085QF202)"

状态: ✅ 已填写，请确认准确性
```

#### 4. GitHub仓库整理
```
当前链接: https://github.com/nblvguohao/PlantMEGNN.git

建议:
- 确认仓库可公开访问
- README.md 包含:
  * 安装说明
  * 快速开始示例
  * 依赖环境 (Python 3.8+, PyTorch 2.1.0, PyG 2.4.0)
  * 模型下载链接
  * 使用示例代码
```

---

### 🟡 建议完成（提升质量）

#### 5. GSTP008多环境框架扩展
```
计划:
- 当前论文以跨群体泛化为主（GSTP007/008/009）
- GSTP008的多环境数据可作为第二阶段投稿重点
- 或在本稿Results中增加一节"Multi-Environment Robustness"

数据状态（从memory）:
- PlantMEGNN-E avg 0.340 vs ST-Ridge 0.287 (+18.5%)
- YangZ15环境表现突出 (+0.096)
- LingS15存在负迁移问题

建议:
- 如果时间在允许范围内，建议增加"Multi-Environment Prediction"小节
- 突出PlantMEGNN在多个环境下的稳健性
```

#### 6. 参考文献更新
```
需要添加:
- Chen et al. 2020 (GSTP008数据来源)
- Zhang et al. 2021 (水稻育种相关)
- 1-2篇 Plant Phenomics 近期相关论文（显示对期刊的熟悉）

格式检查:
- Plant Phenomics 使用 numbered citation style
- 当前使用 plainnat，可能需要调整
```

#### 7. 补充材料完善
```
当前已有:
- S1: Permutation Test Results
- S2: Network Complementarity Analysis

建议添加:
- S3: Per-trait detailed results (if space permits)
- S4: Hyperparameter sensitivity (optional)
```

---

## 关键修改对比

### 语言风格调整

| 原稿 (BIB风格) | 新稿 (Plant Phenomics风格) |
|---------------|---------------------------|
| "honest assessment of limitations" | "practical implications for breeding" |
| "negligible predictive advantage" | "functionally redundant information" |
| "contribution of graph topology" | "value of network information depends on trait architecture" |
| "attention as hypothesis-generating tool" | "attention guides candidate gene prioritization" |

### 重点转移

| 维度 | 原稿重点 | 新稿重点 |
|------|---------|---------|
| 核心贡献 | 可解释性 + 诚实消融 | 预测精度 + 育种应用 |
| 主要结果 | ΔPCC ≈ 0.000 (注意力无用) | +6.2% vs GBLUP (方法有效) |
| 验证策略 | 单数据集深度分析 | 三群体泛化验证 |
| 读者价值 | 方法学洞察 | 实践指导意义 |

---

## 审稿人可能问题及应对

### 1. "与NetGP的差距很小"
**预期问题**: Reviewer可能注意到PlantMEGNN与NetGP性能接近。

**应对策略** (已在文中):
```latex
"When trained under identical conditions, the NetGP-equivalent
PlantMEGNN-1view achieves similar performance, indicating that
the multi-view architecture and biological network integration
provide consistent gains across different implementations."
```

### 2. "注意力权重没有预测优势"
**预期问题**: 为什么 attention mechanism 是贡献？

**应对策略** (已在文中):
- 强调interpretability value: "guides candidate gene prioritization"
- 中性表述: "functionally redundant" 而非 "useless"
- 强调稳定性: "stable, reproducible patterns (std ≤ 0.029)"

### 3. "三群体验证的价值"
**预期问题**: 为什么需要三个数据集？

**应对策略** (已在文中):
- 强调育种应用: "demonstrates practical deployment across different genetic backgrounds"
- 一致性发现: "consistent improvements of +5.5% to +7.8%"
- 网络可迁移性: "species-level biological networks serve as transferable priors"

---

## 投稿前检查清单

### 内容完整性
- [ ] Figure 1 架构图绘制完成
- [ ] 所有表格数据基于实际results文件填充
- [ ] Data Availability 信息完整（含登记号）
- [ ] GitHub仓库可访问且文档完整

### 格式规范
- [ ] 符合 Plant Phenomics 字数要求 (<8000 words)
- [ ] 图表数量符合要求 (主文 ≤6-8)
- [ ] 参考文献格式正确
- [ ] 补充材料整理完成

### 语言润色
- [ ] 全文检查方法导向词汇 → 改为应用导向
- [ ] 确保"育种"关键词出现频率足够
- [ ] Abstract和Conclusion强调实践价值

---

## 时间安排建议

| 任务 | 预计时间 | 优先级 |
|------|---------|--------|
| Figure 1 绘制 | 1天 | 🔴 高 |
| 数据填充验证 | 0.5天 | 🔴 高 |
| GitHub仓库整理 | 0.5天 | 🔴 高 |
| GSTP008多环境扩展 | 1-2天 | 🟡 中 |
| 语言润色 | 0.5天 | 🟡 中 |
| 补充材料完善 | 0.5天 | 🟢 低 |
| 投稿系统准备 | 0.5天 | 🔴 高 |

---

## 联系信息

**通讯作者**: Qingyong Wang
**邮箱**: wqy@ahau.edu.cn
**单位**: 安徽农业大学人工智能学院
**基金**: 国家自然科学基金 (32472007, 62301006, 62301008)

---

*计划更新: 2026-03-30*
*状态: Plant Phenomics版本主稿已完成，待完善图表和细节*
