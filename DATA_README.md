# 数据文件说明

## 重要数据文件

由于 Git 仓库大小限制，原始数据文件未上传，但处理后的关键结果文件可在 `results/` 目录找到：

### 核心结果文件
- `results/data_improvement_results.json` - 7种数据端改进完整实验结果
- `results/snp_scaling_results.json` - SNP扩展（10K vs 50K）对比结果  
- `results/enhanced_benchmark_results.json` - 增强基准测试结果
- `results/split.json` - 数据集划分索引
- `results/metadata.json` - 数据元信息

### 关键数据集（需重新生成）
- `genotype_50k_additive.npy` - 50K多性状选择SNP数据（286MB）
- `genotype_10k_mt_additive.npy` - 10K多性状选择SNP数据（58MB）
- `genotype_onehot.npy` - 原始10K one-hot数据（172MB）
- `phenotype_scaled.npy` - 标准化表型数据（376KB）

## 数据生成脚本

### 1. 50K SNP数据处理
```bash
# 从原始PLINK数据生成50K多性状选择SNP
python scripts/process_full_snps.py
```

### 2. 基准实验
```bash
# 数据端改进实验
python scripts/data_improvements_benchmark.py

# SNP扩展实验  
python scripts/snp_scaling_benchmark.py

# 增强基准实验
python scripts/enhanced_benchmark.py
```

## 原始数据来源

### GSTP007数据集
- **基因型**: `data/raw/GSTP/1495Hybrid_MSUv7.bed` (1.65M SNPs)
- **表型**: `data/raw/GSTP/GSTP007.pheno` (32性状)
- **样本**: 1495个玉米杂交种

### Rice469数据集
- **来源**: CropGS-Hub / GPformer论文
- **样本**: 469个水稻品种
- **SNPs**: 5,291个
- **性状**: 6个

## 数据处理流程

1. **质量控制**: 缺失率<10%, MAF>0.05
2. **特征选择**: 多性状max|corr|选择
3. **编码方式**: Additive(0/1/2) vs One-hot
4. **标准化**: Z-score基于训练集统计

## 性能基准

| 方法 | PCC | MSE | MAE | 数据 |
|------|:---:|:---:|:---:|------|
| **50K+Stacking** | 0.6343 | 0.4896 | 0.5022 | 最佳 |
| **50K+GBLUP** | 0.6253 | 0.4991 | 0.5094 | 单模型最佳 |
| **10K+GBLUP** | 0.5538 | 0.5902 | 0.5682 | 基线 |

详细分析见 `FINAL_DATA_IMPROVEMENT_REPORT.md`
