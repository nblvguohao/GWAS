# PlantMEGNN 对比算法配置分析报告

**分析日期**: 2026-03-30
**目标期刊**: Plant Phenomics
**当前配置**: 6个对比算法 + 3个我们的方法

---

## 一、当前配置分析

### 1.1 论文中的对比算法 (Table 1)

| 序号 | 方法 | 类型 | 说明 |
|------|------|------|------|
| 1 | GBLUP | 统计方法 | 基因组预测金标准 |
| 2 | Ridge | 线性ML | 简单线性基线 |
| 3 | DNNGP | 深度学习 | CNN-based SOTA |
| 4 | Transformer | 深度学习 | 纯自注意力架构 |
| 5 | NetGP | GNN | GCN+PPI，直接竞争对手 |
| 6 | Gene-MLP | 消融基线 | 无图结构的MLP |

**我们的方法** (3个变体):
- PlantMEGNN-1view (单网络)
- PlantMEGNN-3view (多网络)
- PlantMEGNN-best (每性状最优配置)

**总计**: 9个方法（6对比 + 3 ours）

### 1.2 已有但未使用的数据

从 `results/gstp007/benchmark_5fold_cv.json`:
- ✅ GeneSeqGNN - 可用但未使用
- ✅ GeneSeqGNN_core - 可用但未使用

---

## 二、同类型期刊标准对比

### 2.1 Plant Phenomics 发表文章基准

根据近期发表的GP/GNN文章（2023-2026）:

**最低要求** (必须):
1. GBLUP - 统计金标准 ✅
2. 至少1个深度学习SOTA ✅ (DNNGP)
3. 至少1个GNN方法 ✅ (NetGP)

**推荐配置** (中等质量):
4. 传统ML方法 (XGBoost/RF) ❌ 缺失
5. 贝叶斯方法 (BayesB) ❌ 缺失
6. 额外的深度学习SOTA (2-3个) ⚠️ 部分满足

**高阶配置** (高质量):
7. 消融研究 (≥3个消融版本) ✅ (Gene-MLP, 1-view, Uniform)
8. 跨群体验证 ✅ (3个群体)
9. 可解释性分析 ✅ (注意力权重)

### 2.2 与近期论文对比

| 论文 | 期刊 | 基线数量 | 主要基线 |
|------|------|---------|---------|
| NetGP (2024) | Plant Biotech J | 5 | GBLUP, DNNGP, DeepGS, ML-GWAS, CNNGP |
| Cropformer (2024) | Plant Communications | 6 | GBLUP, BayesB, DNNGP, GP-Transformer, DL-GWAS, LSTM |
| GPformer (2024) | Briefings in Bioinf | 7 | GBLUP, RRBLUP, DNNGP, DeepGS, CNN, LSTM, Transformer |
| **PlantMEGNN (当前)** | **Plant Phenomics** | **6** | **GBLUP, Ridge, DNNGP, Transformer, NetGP, Gene-MLP** |

**评估**: 当前配置处于中等水平，与近期论文相当或略少1-2个方法。

---

## 三、缺失的关键基线

### 3.1 高优先级缺失（建议添加）

#### 1. LightGBM / XGBoost
**重要性**: ⭐⭐⭐⭐⭐
**原因**:
- 传统ML在GP中表现强劲
- Plant Phenomics读者期望看到与GBDT的比较
- 引用率高：LightGBM在农业基因组学广泛应用

**实现难度**: 低（sklearn/lightgbm已有实现）
**已有数据**: ❌ 需要重新运行

#### 2. BayesB / BayesC
**重要性**: ⭐⭐⭐⭐
**原因**:
- 贝叶斯方法是GP的标准基线
- 与GBLUP形成互补（贝叶斯 vs 线性混合模型）
- 审稿人常要求贝叶斯方法对比

**实现难度**: 中（可用BGLR包或自行实现）
**已有数据**: ❌ 需要运行

#### 3. GPformer 或 GP-Transformer
**重要性**: ⭐⭐⭐⭐
**原因**:
- 最新的Transformer-based GP方法
- 2024年发表于Briefings in Bioinformatics
- 与我们方法直接竞争（都是深度学习+注意力）

**实现难度**: 高（需要重新实现或使用公开代码）
**已有数据**: ❌ 需要重新实现

### 3.2 中优先级缺失（可选添加）

#### 4. Random Forest
**重要性**: ⭐⭐⭐
**原因**: 经典ML方法，但Ridge和LightGBM更常用

#### 5. CNN-based方法 (DeepGS)
**重要性**: ⭐⭐⭐
**原因**: 与DNNGP类似，但DNNGP已覆盖

#### 6. SoyDNGP
**重要性**: ⭐⭐
**原因**: 针对大豆优化，水稻上效果可能不佳

---

## 四、已有但未利用的数据

### 4.1 GeneSeqGNN
**状态**: 有数据但未使用
**潜力**: ⭐⭐⭐⭐
**建议**: 添加到Table 1

GeneSeqGNN是另一个GNN-based方法，添加它可以：
- 展示与另一个GNN方法的对比
- 证明多视图（PPI+GO+KEGG）优于单网络GNN
- 增加基线数量到7个

**数据状态**: ✅ 已在 `benchmark_5fold_cv.json`

---

## 五、详细建议

### 方案A: 最小改动（快速可行）

**行动**: 添加 GeneSeqGNN 到 Table 1

**修改内容**:
1. 在Table 1添加一行 "GeneSeqGNN"
2. 调整文字描述，提及与另一个GNN的比较
3. 不添加新的实验（利用已有数据）

**结果**: 7个对比算法 + 3个ours = 10个方法
**时间**: 1-2小时
**风险**: 低

### 方案B: 推荐配置（平衡质量和工作量）

**行动**: 添加 GeneSeqGNN + LightGBM

**修改内容**:
1. 添加 GeneSeqGNN（已有数据）
2. 运行 LightGBM 实验（需要1-2天）
3. 添加 BayesB（可选，如果时间允许）

**结果**: 8-9个对比算法
**时间**: 2-3天
**风险**: 中

### 方案C: 高阶配置（追求最佳）

**行动**: 添加 GeneSeqGNN + LightGBM + BayesB + GPformer

**修改内容**:
1. 添加 GeneSeqGNN（已有数据）
2. 运行 LightGBM
3. 运行 BayesB
4. 实现/运行 GPformer（可能需1周）

**结果**: 10个对比算法
**时间**: 1-2周
**风险**: 高（可能延迟投稿）

---

## 六、针对Plant Phenomics的专门建议

### Plant Phenomics 的审稿偏好

基于期刊scope和近期发表文章:

1. **应用导向**: 更看重实际育种价值，而非方法复杂性
2. **完整性**: 期望全面的对比，但不必穷尽所有SOTA
3. **可重复性**: 必须包含统计金标准（GBLUP）和至少1个DL方法

### 当前配置是否足够?

**结论**: ✅ 当前配置**基本足够**，但有改进空间。

**理由**:
- ✅ 包含GBLUP（必须）
- ✅ 包含DNNGP（重要SOTA）
- ✅ 包含NetGP（GNN直接竞争）
- ✅ 消融研究完整（Gene-MLP, 1-view, Uniform）
- ✅ 跨群体验证强（3个群体）
- ⚠️ 缺少传统ML方法（LightGBM/RF）
- ⚠️ 缺少贝叶斯方法（BayesB）

**审稿人可能的反馈**:
1. "为什么没有与XGBoost/LightGBM的比较?" - 可能问题
2. "贝叶斯方法如BayesB呢?" - 可能问题
3. "消融研究很全面" - 正面评价

---

## 七、我的建议

### 推荐方案: 方案B（添加 GeneSeqGNN + LightGBM）

**理由**:
1. GeneSeqGNN已有数据，只需添加到表格（1小时）
2. LightGBM运行快速（1-2天），显著提升基线全面性
3. 8-9个对比算法达到Plant Phenomics高质量标准
4. 不需要延迟投稿太久

**实施步骤**:

**Step 1**: 添加 GeneSeqGNN（立即执行）
```bash
# 提取GeneSeqGNN数据并更新表格
python scripts/update_table_with_geneseqgnn.py
```

**Step 2**: 运行 LightGBM（并行执行）
```bash
# 运行LightGBM基准
python scripts/run_lightgbm.py --dataset gstp007 --n_folds 5 --seeds 42 123 456
```

**Step 3**: 更新论文（2-3小时）
- 修改 Table 1
- 添加 LightGBM 到 Methods 2.5
- 调整 Results 第一段讨论
- 更新 Supplementary Materials

**时间预算**:
- GeneSeqGNN: 1小时
- LightGBM运行: 1-2天（可并行）
- 论文修改: 2-3小时
- **总计**: 2-3天

---

## 八、快速决策指南

**如果时间紧迫（1-2天内投稿）**:
→ 选择方案A：仅添加 GeneSeqGNN
→ 在Cover Letter中说明LightGBM结果将在Revision中添加

**如果时间充裕（3-5天）**:
→ 选择方案B：添加 GeneSeqGNN + LightGBM
→ 这是性价比最高的选择

**如果追求顶级质量（1-2周）**:
→ 选择方案C：添加所有建议基线
→ 但可能不值得延迟投稿

---

## 九、GeneSeqGNN 数据提取

基于已有数据，GeneSeqGNN的性能:

```python
# 从 benchmark_5fold_cv.json 提取
GeneSeqGNN_avg_PCC = ~0.755  # 估算，需精确计算
```

建议添加到Table 1的位置：NetGP之后，Gene-MLP之前

---

**总结**: 当前配置可以通过添加2个方法（GeneSeqGNN + LightGBM）显著提升，投入2-3天时间获得更高质量的投稿。

是否需要我立即开始实施**方案B**（添加 GeneSeqGNN + LightGBM）？
