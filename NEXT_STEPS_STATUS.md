# PlantMEGNN 论文升级执行状态

## 当前状态摘要

### 已完成的任务

1. **视图差异化实现完成**
   - PPI网络: 831 genes (基准)
   - GO网络: 1,712 genes (+881 exclusive, 59.8% overlap)
   - KEGG网络: 4,053 genes (+3,222 exclusive, 11.0% overlap)
   - 注意力权重方差提升 13,000× (10⁻⁶ → 0.013)

2. **论文内容更新**
   - Methods 2.2 新增 "View-Differentiated Network Design" 段落
   - Results 3.2 更新消融实验讨论
   - Discussion 4.2 新增视图差异化分析

3. **可视化图表生成**
   - `fig4_attention_comparison.png/pdf` - 注意力权重对比热力图
   - `fig_attention_variance.png/pdf` - 方差对比柱状图
   - `fig_ablation_results.png/pdf` - 消融实验结果图

4. **审稿报告完成**
   - `PAPER_REVIEW_REPORT.md` - 基于adjustment.md的完整审稿意见
   - 评分: 6/10 (Borderline Accept with Major Revisions)
   - 提供详细修改建议和行动指南

### 进行中任务

**CPU消融实验运行中** (Task: bgpklcgwa)
- 状态: PPI+GO配置，Run 3/5
- 当前结果:
  - PPI-only: 0.8486 ± 0.0077
  - PPI+GO: Run 1=0.8520, Run 2=0.8571
- 预计完成时间: 还需3-4小时
- 输出: `results/gstp007/ablation_diverse_views_complete.json`

## 关键发现

### 1. 注意力机制成功差异化

| 设计 | PPI权重 | GO权重 | KEGG权重 | 方差 |
|------|---------|--------|----------|------|
| 原版 | 0.333 | 0.333 | 0.333 | ~10⁻⁶ |
| 改进 | 0.364 | 0.431 | 0.206 | 0.013 |

- GO视图获得最高权重(0.43)，符合Grain Length生物学特性
- KEGG权重较低(0.21)，可能反映噪声

### 2. 审稿报告识别的关键问题

**必须解决 (Critical):**
1. 消融实验不完整 - PPI+GO+KEGG结果缺失
2. 跨群体迁移声明过度 - 需要降级为"探索性分析"
3. 注意力机制预测价值有限 - 需要明确区分"可解释性"vs"预测增益"

**建议解决 (Recommended):**
4. LightGBM对比需要更多背景说明
5. 网络覆盖限制需要明确讨论

## 下一步行动计划

### 短期 (1-2天)

1. **等待CPU消融实验完成**
   - 监控: `tail -f results/gstp007/ablation_cpu_run.log`
   - 完成后更新Table 3和Section 3.2

2. **根据审稿报告修改论文**
   - 修改Section 4.3跨群体迁移声明
   - 澄清Abstract和Introduction中的注意力机制价值主张
   - 在Discussion中添加LightGBM背景段落

### 中期 (3-5天)

3. **补充实验 (如消融实验结果支持)**
   - 在其他性状(Plant Height, Grain Weight)上验证视图差异化
   - 补充网络内外基因对比分析

4. **完善Rebuttal准备**
   - 完成 `PAPER_UPDATE_REBUTTAL.md`
   - 准备针对审稿意见的完整回复

### 长期 (投稿前)

5. **最终论文润色**
   - 语法检查和逻辑检查
   - 图表最终确认
   - 补充材料整理

## 文件清单

### 核心代码
- `scripts/build_diverse_view_networks.py` - 构建差异化网络
- `scripts/ablation_diverse_views.py` - GPU消融实验(失败)
- `scripts/ablation_diverse_views_cpu.py` - CPU消融实验(运行中)
- `scripts/generate_attention_visualizations.py` - 可视化生成

### 数据文件
- `data/processed/gstp007/graph_diverse_views/`
  - `ppi_adj.npz` (831 genes)
  - `go_adj.npz` (1,712 genes)
  - `kegg_adj.npz` (4,053 genes)

### 文档
- `IMPLEMENTATION_SUMMARY.md` - 实现总结
- `PAPER_UPDATE_REBUTTAL.md` - Rebuttal预备
- `PAPER_REVIEW_REPORT.md` - 审稿报告
- `NEXT_STEPS_STATUS.md` - 本文件

## 技术债务

1. **稀疏矩阵支持** - 当前实现部分支持，需要完整迁移
2. **内存优化** - KEGG网络(4,053节点)需要稀疏矩阵避免OOM
3. **CPU性能** - 消融实验在CPU上运行缓慢，考虑服务器资源

## 预期时间表

| 任务 | 预计完成 | 状态 |
|------|----------|------|
| CPU消融实验完成 | +4小时 | 进行中 |
| 论文修改(迁移声明) | +1天 | 待开始 |
| 论文修改(注意力价值) | +1天 | 待开始 |
| 补充实验(其他性状) | +3天 | 待开始 |
| Rebuttal最终版 | +5天 | 待开始 |
| 投稿准备 | +7天 | 待开始 |

---

*最后更新: 2026-03-31*
*运行任务: bgpklcgwa (CPU消融实验)*
