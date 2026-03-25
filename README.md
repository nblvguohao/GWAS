# PlantHGNN: Plant Heterogeneous Graph Neural Network

**基于异构图神经网络与自适应残差的植物多性状基因组预测框架**

## 项目概述

PlantHGNN 是首个将异构图 Transformer（TREE/GRAFT 框架）应用于植物基因组预测（GP/GWAS）的工作，结合 Kimi Attention Residuals（AttnRes）实现多网络证据的自适应深度聚合。

### 核心创新

1. **异构图 Transformer 迁移**：将癌症驱动基因预测技术体系首次系统性迁移至植物育种
2. **植物专用异构生物网络**：设计 Gene-TF-Metabolite（GTM）元路径
3. **AttnRes 自适应聚合**：多网络模态的深度方向注意力融合
4. **多性状联合预测**：利用图结构的多任务学习框架

### 目标期刊

- **Plant Biotechnology Journal** (IF ~9.5)
- **Briefings in Bioinformatics** (IF ~9.5)
- **Plant Communications** (IF ~9.4)
- **Bioinformatics** (IF ~5.8)

## 项目结构

```
GWAS/
├── data/                    # 数据目录（不入 git）
│   ├── raw/                 # 原始数据
│   └── processed/           # 预处理数据
├── src/                     # 源代码
│   ├── data/                # 数据处理
│   ├── models/              # 模型实现
│   ├── training/            # 训练模块
│   └── analysis/            # 分析工具
├── experiments/             # 实验配置和结果
├── notebooks/               # Jupyter notebooks
└── paper/                   # 论文图表
```

## 环境配置

```bash
conda create -n planthgnn python=3.10
conda activate planthgnn
pip install -r requirements.txt
```

## 快速开始

### 1. 数据准备

```bash
# 下载数据集
python src/data/download.py --dataset rice469 maize282

# 预处理
python src/data/preprocess.py --dataset rice469
```

### 2. 训练模型

```bash
# 训练 PlantHGNN
python experiments/run_experiment.py --config experiments/configs/base_config.yaml
```

### 3. 评估基线

```bash
# 运行基线模型
python experiments/run_baseline.py --model netgp --dataset rice469
```

## 数据集

| 数据集 | 作物 | 样本数 | SNP数 | 性状数 | 来源 |
|--------|------|--------|-------|--------|------|
| rice469 | 水稻 | 469 | 5,291 | 6 | CropGS-Hub |
| maize282 | 玉米 | 282 | 3,093 | 3 | PANZEA |
| soybean999 | 大豆 | 999 | 7,883 | 6 | SoyDNGP |
| wheat599 | 小麦 | 599 | 1,447 | 3 | CIMMYT |

## 引用

```bibtex
@article{planthgnn2026,
  title={PlantHGNN: Heterogeneous Graph Neural Network with Attention Residuals for Plant Genomic Prediction},
  author={Lyu et al.},
  journal={TBD},
  year={2026}
}
```

## 许可证

MIT License

## 联系方式

- 作者：Lyu（安徽农业大学 AI学院 智慧农业重点实验室）
- GitHub: https://github.com/nblvguohao/GWAS.git
