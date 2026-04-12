# PlantMEGNN Plant Phenomics 投稿包总结

**生成日期**: 2026-03-30
**目标期刊**: Plant Phenomics
**状态**: ✅ 投稿准备完成

---

## 文件清单

### 主论文文件
| 文件 | 大小 | 说明 |
|------|------|------|
| `paper_plant_phenomics/main.tex` | ~20KB | 完整LaTeX论文（15页） |
| `paper_plant_phenomics/main.pdf` | 2.4MB | 编译后的PDF |
| `paper_plant_phenomics/references.bib` | ~15KB | 参考文献（44条） |
| `paper_plant_phenomics/figures/fig1.png` | 2.1MB | 架构图（高清） |

### 投稿支持文件
| 文件 | 大小 | 说明 |
|------|------|------|
| `paper_plant_phenomics/cover_letter.tex` | ~5KB | Cover Letter（LaTeX） |
| `paper_plant_phenomics/cover_letter.pdf` | ~68KB | Cover Letter（PDF） |
| `GITHUB_README_TEMPLATE.md` | ~10KB | GitHub README模板 |

### 文档与总结
| 文件 | 说明 |
|------|------|
| `REVISION_SUMMARY.md` | 详细修订总结 |
| `SUBMISSION_CHECKLIST.md` | 投稿检查清单 |
| `SUBMISSION_PACKAGE_SUMMARY.md` | 本文件 |

---

## 论文统计

### 基本信息
- **标题**: PlantMEGNN: Multi-View Graph Neural Network Improves Genomic Prediction Accuracy for Rice Breeding
- **页数**: 15 pages
- **字数**: ~3,200 words (正文) + 图表说明
- **参考文献**: 44条
- **图表**: 1 Figure + 5 Tables

### 主要结果
| 指标 | 数值 |
|------|------|
| GSTP007平均PCC | 0.786 (vs GBLUP +6.2%) |
| 跨群体一致性 | +5.5% ~ +7.8% vs GBLUP |
| 多环境平均提升 | +5.0% vs ST-Ridge |

---

## 论文结构

```
1. Introduction (育种背景，应用导向)
2. Materials and Methods
   2.1 Plant Materials and Datasets (3个群体)
   2.2 Biological Network Construction (PPI/GO/KEGG)
   2.3 PlantMEGNN Architecture (5组件)
   2.4 Training and Evaluation
   2.5 Baseline Methods
3. Results
   3.1 Prediction Accuracy on GSTP007
   3.2 Ablation Study (Learnable vs Uniform vs Gene-MLP)
   3.3 Biological Interpretability (注意力权重分析)
   3.4 Cross-Population Generalizability (3群体验证)
   3.5 Cross-Population Transfer Analysis
   3.6 Multi-Environment Prediction on GSTP008 (NEW)
4. Discussion
   4.1 Practical Implications for Rice Breeding (UPDATED)
   4.2 Biological Network Integration (UPDATED)
   4.3 Comparison with Existing Methods
   4.4 Future Directions
5. Conclusion
Supplementary Materials (S1: Permutation Test, S2: Network Complementarity)
```

---

## 主要修改（基于审稿人意见）

### ✅ 已完成的修改

1. **消极表述调整** (2处)
   - "negligible" → "similar"
   - "functionally redundant" → "complementary yet functionally correlated"

2. **GSTP008多环境结果** (新增)
   - 新增小节: "Multi-Environment Prediction on GSTP008"
   - 新增表格: Table 5 (4环境LOEO结果)
   - 平均提升: +5.0% vs ST-Ridge

3. **Table 4标准差补充**
   - 所有GSTP007结果添加±std
   - 基于5-fold CV × 3 seeds (15 runs)

4. **部署细节补充**
   - 训练时间: ~10 min (RTX 3090)
   - 推理速度: 50 samples/sec
   - 模型大小: 3MB
   - 部署建议: 3条实用指南

5. **参考文献更新**
   - 添加4条缺失引用
   - 总参考文献数: 44条

---

## 关键发现总结

### 主实验 (GSTP007)
```
PlantMEGNN: 0.786 (vs GBLUP 0.740, +6.2%)
Grain Length: 0.881 (最高精度)
Grain Weight: 0.836 (+3.5% vs GBLUP)
```

### 跨群体泛化
```
GSTP007 (n=1,495): 0.832 (+5.5% vs GBLUP)
GSTP008 (n=705):   0.804 (+5.4% vs GBLUP)
GSTP009 (n=378):   0.661 (+7.8% vs GBLUP)
```

### 多环境预测 (GSTP008)
```
平均提升: +5.0% vs ST-Ridge
YangZ15: +59.2% (最显著)
LingS15: -21.4% (负迁移，讨论原因)
```

### 生物学发现
```
KEGG主导: 粒形性状 (GL, GW, GWt)
PPI主导: 产量相关性状 (YPP)
GO中间: 株高、抽穗期
注意力稳定性: std ≤ 0.029
```

---

## 投稿前待办事项

### 🔴 高优先级（必须完成）

- [ ] **Figure 1分辨率确认**
  - 当前: PNG格式，2.1MB
  - 建议: 确认≥300 dpi，或提供PDF矢量图

- [ ] **数据登记号核实**
  - GSTP007: 3K Rice Genome Project (需提供具体登记号)
  - GSTP008: CropGS-Hub (需确认DOI或链接)
  - GSTP009: USDA Diversity Panel (需确认登记号)

- [ ] **GitHub仓库填充**
  - 当前状态: 仓库存在但为空
  - 需要上传: README.md, 代码, 预训练模型
  - 参考: `GITHUB_README_TEMPLATE.md`

- [ ] **作者信息确认**
  - 所有作者姓名拼写
  - 通讯作者邮箱: wqy@ahau.edu.cn
  - 作者单位信息

### 🟡 中优先级（建议完成）

- [ ] **Cover Letter最终审查**
  - 文件: `paper_plant_phenomics/cover_letter.pdf`
  - 确认: 2页，内容完整

- [ ] **Highlights准备** (投稿系统需要)
  ```markdown
  • PlantMEGNN improves genomic prediction by 6.2% over GBLUP for rice breeding
  • Multi-view GNN integrates PPI, GO, and KEGG biological networks
  • Cross-population validation shows consistent gains (+5.5% to +7.8%) across 2,578 accessions
  • Learned attention weights reveal trait-specific network preferences
  • Open-source implementation available at GitHub
  ```

- [ ] **Graphical Abstract** (可选，增强展示)
  - 可用Figure 1改编
  - 尺寸: 1200×1600像素

### 🟢 低优先级（可选）

- [ ] **补充材料扩展**
  - S3: GSTP008 per-environment详细结果
  - S4: 基因案例分析 (如GW5)

- [ ] **预训练模型上传**
  - 7个性状的训练模型
  - 模型卡片 (model card) 说明

---

## 投稿系统信息

### Plant Phenomics投稿网址
https://spj.science.org/journal/plantphenomics

### 投稿类型
Research Article

### 建议审稿人
1. Dr. Susan McCouch (Cornell University) - Rice genetics
2. Dr. José Crossa (CIMMYT) - Statistical methods
3. Dr. Xuehui Huang (CAS) - Rice GWAS

### 需回避的审稿人
（如有利益冲突，请在Cover Letter中说明）

---

## 联系方式

**通讯作者**: Qingyong Wang
**邮箱**: wqy@ahau.edu.cn
**单位**: 安徽农业大学人工智能学院
**地址**: 安徽省合肥市蜀山区长江西路130号
**邮编**: 230036

**第一作者**: Guohao Lv
**GitHub**: https://github.com/nblvguohao/PlantMEGNN

---

## 使用说明

### 如何上传GitHub仓库

```bash
# 克隆仓库
git clone https://github.com/nblvguohao/PlantMEGNN.git
cd PlantMEGNN

# 复制README
cp ../GITHUB_README_TEMPLATE.md README.md

# 添加代码文件
cp -r ../src .
cp -r ../scripts .
cp ../requirements.txt .

# 提交
git add .
git commit -m "Initial release: PlantMEGNN v1.0.0"
git push origin main
```

### 如何提交论文

1. **准备文件**:
   ```
   - main.tex
   - main.pdf
   - references.bib
   - figures/fig1.png
   - cover_letter.pdf
   ```

2. **在线投稿**:
   - 访问 Plant Phenomics投稿系统
   - 创建账户/登录
   - 上传所有文件
   - 填写元数据（标题、摘要、关键词等）
   - 添加建议审稿人

3. **投稿后**:
   - 保存投稿确认号
   - 关注审稿进度邮件
   - 准备回复审稿意见

---

## 附录：快速参考

### 论文核心贡献（一句话总结）
> PlantMEGNN通过多视图生物网络集成，在水稻基因组预测中实现6.2%精度提升，并揭示性状特异性网络偏好，为育种实践提供可解释的预测工具。

### 主要创新点
1. 多视图网络集成（PPI+GO+KEGG）
2. 可解释注意力机制
3. 跨群体泛化验证（3个独立群体）
4. 多环境GxE建模（可选扩展）

### 应用价值
- 训练时间: ~10分钟（1,495样本）
- 推理速度: 50样本/秒
- 模型大小: 3MB
- 部署建议: 包含在论文中

---

**祝投稿顺利！** 🎉

*本总结由Claude Code生成 - 2026-03-30*
