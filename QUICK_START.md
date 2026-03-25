# 快速开始指南

## 环境配置

```bash
# 创建虚拟环境
conda create -n gwas python=3.10
conda activate gwas

# 安装依赖
pip install -r requirements.txt

# PyG安装（CUDA 12.1）
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

## 核心实验运行

### 1. 数据端改进实验（推荐）
```bash
python scripts/data_improvements_benchmark.py
```
**输出**: `results/data_improvement_results.json`  
**内容**: 7种数据改进策略对比（Additive编码、Huber损失、多性状选择等）

### 2. SNP扩展实验
```bash
python scripts/snp_scaling_benchmark.py
```
**输出**: `results/snp_scaling_results.json`  
**内容**: 10K vs 50K SNP性能对比

### 3. 增强基准实验
```bash
python scripts/enhanced_benchmark.py
```
**输出**: `results/enhanced_benchmark_results.json`  
**内容**: 集成学习、深度模型对比

## 关键发现

### 🏆 最佳性能
- **50K SNPs + Multi-trait + Stacking**: PCC=0.6343
- **提升幅度**: +0.0805 (+14.5%) vs 原始基线

### 📊 核心改进
1. **数据质量 > 模型复杂度**
   - 50K vs 10K: +0.04 PCC
   - Additive vs One-hot: +0.02 PCC
   - Multi-trait vs Single: +0.01 PCC

2. **简单模型同样有效**
   - GBLUP: 0.6253 PCC（单模型最佳）
   - MLP: 0.6241 PCC（接近GBLUP）

3. **集成学习持续有效**
   - Stacking: +0.009 PCC vs 最佳单模型

## 文件结构

```
GWAS/
├── FINAL_DATA_IMPROVEMENT_REPORT.md  # 完整分析报告
├── CLAUDE.md                          # 原始研究方案
├── DATA_README.md                     # 数据说明
├── scripts/                           # 核心脚本
│   ├── data_improvements_benchmark.py # 数据改进实验
│   ├── snp_scaling_benchmark.py       # SNP扩展实验
│   ├── enhanced_benchmark.py          # 增强基准实验
│   └── diagnose_and_fix.py            # 问题诊断
├── results/                           # 实验结果
│   ├── data_improvement_results.json  # 数据改进结果
│   ├── snp_scaling_results.json      # SNP扩展结果
│   └── enhanced_benchmark_results.json # 增强基准结果
└── data/processed/                    # 处理后数据（需重新生成）
```

## 预期运行时间

| 实验 | 数据准备 | 训练 | 总时间 |
|------|---------|------|--------|
| 数据改进 | - | 15分钟 | 15分钟 |
| SNP扩展 | 6分钟 | 30分钟 | 36分钟 |
| 增强基准 | - | 20分钟 | 20分钟 |

## 硬件要求

- **GPU**: RTX 4090 (24GB) 或类似
- **内存**: 32GB+ RAM
- **存储**: 10GB 可用空间

## 故障排除

### 1. CUDA内存不足
```bash
# 减少batch size
export CUDA_VISIBLE_DEVICES=0
python scripts/data_improvements_benchmark.py --batch_size 16
```

### 2. 数据文件缺失
```bash
# 重新生成50K数据
python scripts/process_full_snps.py
```

### 3. 依赖安装问题
```bash
# 更新pip
pip install --upgrade pip
# 重新安装PyG
pip uninstall torch_geometric -y
pip install torch_geometric
```

## 结果解读

### PCC (皮尔逊相关系数)
- **>0.6**: 优秀
- **0.5-0.6**: 良好  
- **<0.5**: 需改进

### MSE/MAE
- **越低越好**: 表示预测误差小
- **相对比较**: 关注相对提升而非绝对值

## 下一步

1. **探索更多SNPs**: 尝试100K+ SNPs
2. **生物网络整合**: PPI/GO/KEGG网络
3. **预训练范式**: 大规模基因组预训练
4. **多组学融合**: 表达量+甲基化数据

详细技术细节见 `FINAL_DATA_IMPROVEMENT_REPORT.md`
