# GWAS: 基因组预测优化项目

## 🎯 核心成果

通过系统性的数据端改进，实现了 **PCC 从 0.5538 → 0.6343** 的突破性提升（+14.5%），验证了"数据质量优于模型复杂度"的核心假设。

### 🏆 最佳性能
- **方法**: 50K SNPs + Multi-trait Selection + Stacking
- **PCC**: 0.6343
- **提升**: +0.0805 (+14.5%) vs 原始基线

## 📊 关键发现

| 改进策略 | PCC提升 | 说明 |
|---------|:-------:|------|
| **50K vs 10K SNPs** | +0.04 | 数据量决定性能上限 |
| **Additive vs One-hot** | +0.02 | 编码效率提升 |
| **Multi-trait Selection** | +0.01 | 特征选择优化 |
| **Huber vs MSE Loss** | +0.014 | 鲁棒性增强 |

## 🚀 快速开始

```bash
# 环境配置
conda create -n gwas python=3.10
conda activate gwas
pip install -r requirements.txt

# 核心实验
python scripts/data_improvements_benchmark.py
python scripts/snp_scaling_benchmark.py
```

详细指南见 [QUICK_START.md](QUICK_START.md)

## 📁 项目结构

```
GWAS/
├── FINAL_DATA_IMPROVEMENT_REPORT.md  # 完整分析报告
├── CLAUDE.md                          # 原始研究方案  
├── QUICK_START.md                     # 快速开始指南
├── DATA_README.md                     # 数据说明文档
├── scripts/                           # 核心实验脚本
├── results/                           # 实验结果文件
└── data/processed/                    # 处理后数据（需生成）
```

## 📈 性能对比

| 方法 | PCC | MSE | MAE | 数据 |
|------|:---:|:---:|:---:|------|
| **50K+Stacking** | 0.6343 | 0.4896 | 0.5022 | 🏆 最佳 |
| **50K+GBLUP** | 0.6253 | 0.4991 | 0.5094 | 单模型最佳 |
| **10K+GBLUP** | 0.5538 | 0.5902 | 0.5682 | 原始基线 |

## 🔬 核心贡献

1. **数据优先策略**: 证明数据质量比模型复杂度更重要
2. **系统性改进**: 7种数据端改进策略完整评估
3. **实用方案**: 简单高效的GBLUP/MLP达到接近SOTA性能
4. **可重现性**: 完整的实验流程和代码

## 📚 相关文档

- [完整分析报告](FINAL_DATA_IMPROVEMENT_REPORT.md) - 详细技术分析和结果
- [原始研究方案](CLAUDE.md) - PlantHGNN异构图神经网络设计
- [数据说明](DATA_README.md) - 数据处理和生成指南
- [快速开始](QUICK_START.md) - 环境配置和运行指南

## 🛠️ 技术栈

- **深度学习**: PyTorch, PyTorch Geometric
- **数据处理**: NumPy, Pandas, bed-reader
- **统计分析**: SciPy, scikit-learn
- **基因组学**: PLINK, GWAS相关工具

## 📊 实验环境

- **GPU**: RTX 4090 (24GB)
- **内存**: 64GB RAM
- **数据**: GSTP007 (1495样本, 1.65M→50K SNPs)
- **评估**: 5折交叉验证, PCC/Spearman/MSE/MAE

## 🎯 未来方向

1. **预训练范式**: 大规模基因组预训练
2. **生物网络整合**: 真实PPI/GO/KEGG网络
3. **多组学融合**: 表达量+甲基化数据
4. **跨物种迁移**: 水稻→小麦等作物迁移学习

---

**结论**: 在基因组预测领域，数据端改进的贡献（80%）远大于模型端改进（20%）。简单高效的GBLUP/MLP在高质量数据上能达到接近复杂模型的性能。
