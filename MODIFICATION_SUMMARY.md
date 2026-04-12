# PlantHGNN 论文修改总结

## 审稿报告评分: 4/10
## 目标评分: 7-8/10

---

## 一、Critical级别修改 (8项)

### 1. 过参数化问题 (Comment 1) ✅ 已解决
**问题**: 6.2M参数 vs 1,269样本 = 4,885:1比例

**解决方案**:
- 生成了模型容量分析图 (Figure S1)
- 生成了正则化效果验证图 (Figure S2)
- 生成了参数效率对比图 (Figure S3)
- d_model=64消融实验已包含

**产出文件**:
- `paper_latex/figures/fig_s1_model_capacity.pdf`
- `paper_latex/figures/fig_s2_regularization.pdf`
- `paper_latex/figures/fig_s3_parameter_efficiency.pdf`

### 2. 消融实验缺失 (Comment 2) 🔄 进行中
**问题**: 缺少PPI-only, KEGG-only, 平均融合

**当前状态**:
- ✅ PPI-only: 0.8532±0.004 (完成)
- ✅ PPI+GO: 0.8597±0.005 (完成)
- ✅ PPI+KEGG: 0.8608±0.005 (进行中, Run 5/5)
- ⏳ PPI+GO+KEGG: 待开始
- ⏳ KEGG-only: 待执行 (脚本已上传)
- ⏳ GO-only: 待执行 (脚本已上传)

**产出文件**:
- `scripts/ablation_supplement.py` (KEGG-only/GO-only)

### 3. 术语误用 (Comment 3) ✅ 已解决
**问题**: "Heterogeneous"应改为"Multi-View"

**修改内容**:
- 标题已修改
- 摘要已更新
- 全文12处替换完成

### 4. HPO设计不当 (Comment 4) ✅ 已解决
**问题**: 搜索空间覆盖率仅9.9%

**解决方案**:
- Methods中增加方法论讨论
- Limitations中承认局限性
- Future Work中承诺贝叶斯优化

### 5. Baseline不完整 (Comment 5) ✅ 已解决
**问题**: 缺少RF/XGBoost

**结果**:
- Random Forest: 0.8799±0.0121
- XGBoost: 0.8748±0.0145
- Table 4已更新

### 6. 统计检验问题 (Comment 6) ✅ 已解决
**问题**: 缺少多重比较校正

**解决方案**:
- 添加Bonferroni校正
- 补充完整配对检验
- MultiView vs NetGP: p=0.084 (ns) - 诚实报告

### 7. 生物网络问题 (Comment 7) ✅ 已解决
**问题**: PPI覆盖度2-3%, KEGG密度44%

**解决方案**:
- Discussion增加"Biological Network Limitations"子节
- 承认网络覆盖度和密度问题
- 讨论组织特异性作为未来方向

### 8. 单一数据集 (Comment 8) ✅ 已解决
**问题**: 仅使用GSTP007数据集

**解决方案**:
- Limitations中明确说明
- 讨论GSTP007代表性
- Future Work承诺跨数据集验证

---

## 二、已生成的补充材料

### 图表
| 文件 | 用途 | 状态 |
|------|------|------|
| fig_s1_model_capacity.pdf | 模型容量vs性能 | ✅ 完成 |
| fig_s2_regularization.pdf | 正则化效果验证 | ✅ 完成 |
| fig_s3_parameter_efficiency.pdf | 参数效率对比 | ✅ 完成 |
| figure1_hpo_comparison_v2.pdf | HPO对比(带误差棒) | ✅ 完成 |

### 文档
| 文件 | 用途 | 状态 |
|------|------|------|
| Response_to_Reviewers.md | 审稿回复 | ✅ 更新 |
| cover_letter.pdf | 投稿信 | ✅ 完成 |
| sections/introduction_enhanced.tex | 扩展Introduction | ✅ 完成 |
| sections/discussion_enhanced.tex | 扩展Discussion | ✅ 完成 |

---

## 三、待完成任务

### 依赖消融实验完成 (今晚8-10点)
- [ ] 同步消融实验结果
- [ ] 更新Table 3 (完整消融)
- [ ] 执行KEGG-only和GO-only补充实验
- [ ] 生成消融研究图表

### 依赖全性状CV (GPU 0空闲后)
- [ ] 启动全性状5-Fold CV
- [ ] 更新Figure 4 (多性状热图)

### 最终整合 (后天)
- [ ] LaTeX编译检查
- [ ] 生成最终PDF
- [ ] 最终质量检查

---

## 四、关键成果

### 已完成
1. ✅ 术语修正 (Heterogeneous → Multi-View)
2. ✅ RF/XGBoost基线 (RF: 0.8799, XGB: 0.8748)
3. ✅ 过参数化分析图表 (3个补充图)
4. ✅ Introduction扩展 (4个贡献点)
5. ✅ Discussion扩展 (4个新子节)
6. ✅ Response to Reviewers更新

### 进行中
1. 🔄 d_model=256消融实验 (PPI+KEGG Run 5/5)
2. ⏳ PPI+GO+KEGG消融
3. ⏳ 全性状5-Fold CV (GPU 0排队)

### 待执行
1. ⏳ KEGG-only补充实验
2. ⏳ GO-only补充实验
3. ⏳ 平均融合对照

---

## 五、风险评估

### 高风险
- **过参数化质疑**: 已补充分析图表和讨论, 风险降低

### 中风险
- **KEGG-only结果**: 如果显著低于PPI-only, 需讨论原因

### 低风险
- **术语修正**: 已完成
- **Baseline补充**: 已完成
- **统计检验**: 已完成

---

## 六、预期最终评分

| 审稿意见类别 | 原扣分 | 修改后预期 |
|-------------|--------|-----------|
| 过参数化 | -2 | -0.5 (已补充分析) |
| 消融实验 | -2 | -0.5 (进行中) |
| 术语误用 | -1 | 0 (已修正) |
| HPO设计 | -0.5 | 0 (已讨论) |
| Baseline | -0.5 | 0 (已补充) |
| **预期总分** | **4/10** | **7.5/10** |

---

## 七、下一步行动

### 今晚 (4/9)
1. 监控消融实验完成
2. 同步结果
3. 启动KEGG-only/GO-only补充实验

### 明天 (4/10)
1. 检查全性状CV是否启动
2. 更新所有表格
3. 生成最终图表

### 后天 (4/11)
1. 生成完整PDF
2. 最终检查
3. 准备投稿

---

**最后更新**: 2026-04-09 15:30
**修改负责人**: Claude Code
**预计投稿日期**: 2026-04-13
