# PlantHGNN: 植物性状预测异构图神经网络研究方案

> **Claude Code 执行指南**：本文件是精简版科研方案，保留核心指令。详细内容见 CLAUDE_FULL.md。

---

## 0. 项目概览

### 0.1 研究定位

**项目名称**：PlantHGNN — 基于异构图神经网络与自适应残差的植物多性状基因组预测框架

**核心创新**：将癌症驱动基因预测领域（TREE/GRAFT）的技术体系首次系统性迁移至植物育种基因组预测（GP/GWAS），结合 Kimi Attention Residuals（AttnRes）实现多网络证据的自适应深度聚合。

**技术栈对应**：
- 癌症基因预测（源域）→ 植物GP预测（目标域）
- PPI网络 → 植物蛋白互作网络（STRING-plant）
- GO语义相似性网络 → 植物GO功能注释网络
- TF-miRNA-gene异构网络 → PlantTFDB调控网络 + miRBase植物miRNA
- 节点二分类 → 节点回归（GEBV育种值预测）

### 0.2 创新点评估

1. **核心创新**：首个将异构图Transformer（TREE/GRAFT框架）应用于植物GP的工作
2. **植物专用异构生物网络构建**：设计GTM（Gene-TF-Metabolite）元路径
3. **AttnRes自适应多网络证据聚合**：Kimi AttnRes用于跨网络模态融合
4. **多性状联合预测**（可选扩展）

### 0.3 目标期刊评估

**主投目标**：
1. Plant Biotechnology Journal (~9.5 IF) - NetGP发表于此
2. Briefings in Bioinformatics (~9.5 IF) - 方法论导向
3. Plant Communications (~9.4 IF) - 接受新架构
4. Bioinformatics (~5.8 IF) - 稳妥选项

---

## 1. 硬件资源规划

### 1.1 本地配置（4060 8G VRAM + 16G RAM）
- 数据预处理、小规模数据集调试
- 基线模型在小数据集上的复现验证
- 消融实验小数据集版本
- 可解释性可视化

**显存约束**：
- 最大batch size：节点数≤5000时batch=32
- 模型参数上限：~50M参数
- SNP序列长度：PCS降维至≤5000特征

### 1.2 服务器任务清单 `[SERVER]`
- 大规模数据集训练（rice3k, wheat2403）
- 完整基线模型复现（Cropformer, CropARNet等）
- 超参数搜索（网格搜索）
- 5折交叉验证全量实验
- 大规模网络构建（STRING植物版）

---

## 2. 项目结构与GitHub配置

### 2.1 核心目录结构

```
GWAS/
├── CLAUDE.md                  # 本文件（精简版）
├── CLAUDE_FULL.md             # 完整详细版
├── README.md
├── requirements.txt
├── data/                      # 数据目录（不入git）
│   ├── raw/                   # 原始下载数据
│   └── processed/             # 预处理后数据
├── src/                       # 源代码
│   ├── data/                  # 数据处理模块
│   ├── models/                # 模型实现
│   ├── training/              # 训练模块
│   └── analysis/              # 分析模块
├── experiments/               # 实验配置
│   ├── configs/               # 配置文件
│   ├── scripts/               # 运行脚本
│   └── results/               # 实验结果
├── notebooks/                 # Jupyter笔记本
└── paper/                     # 论文材料
```

### 2.2 Git初始化已完成
- 项目已初始化：https://github.com/nblvguohao/GWAS.git
- 目录结构已创建

---

## 3. Phase 1：数据集构建与预处理（核心任务）

### 3.1 核心数据集选择

**主实验数据集**：
- rice469（水稻，469样本，5,291 SNP，6性状）- 本地
- maize282（玉米，282样本，3,093 SNP，3性状）- 本地
- soybean999（大豆，999样本，7,883 SNP，6性状）- 本地
- wheat599（小麦，599样本，1,447 SNP，3性状）- 本地
- rice3k（水稻，3,000样本）- 本地预处理，[SERVER]训练

### 3.2 生物网络数据下载
- STRING v12 plant-specific（combined_score > 700）
- GO注释（当前版本）
- KEGG植物通路（REST API）
- PlantTFDB v5（TF-target调控关系）
- miRBase v22（植物miRNA）

### 3.3 SNP预处理流水线
1. **质量控制**：PLINK过滤（缺失率>10%，MAF<0.05）
2. **PCS特征选择**：Pearson相关过滤（|r|>0.3），VIF<10
3. **one-hot编码**：AA→[1,0,0], AB→[0,1,0], BB→[0,0,1]
4. **按染色体排序**：保留空间信息

### 3.4 生物网络构建

**同构网络**：
1. 蛋白互作网络（PPI）- STRING，score>700
2. GO功能相似性网络 - GOSemSim算法，相似性>0.8
3. KEGG通路共现网络 - Jaccard相似性，weight>0.1

**异构网络GTM**：
- 节点类型：gene, TF, metabolite
- 边类型：gene-gene, TF-gene, TF-metabolite
- 元路径：GG, GTG, GTMTG

### 3.5 数据集划分策略
1. **随机划分**：80% train, 10% val, 10% test（5折CV）
2. **染色体划分**：leave-one-chromosome-out（更严格）
3. **品系划分**：基于系谱/群体结构（推荐，避免亲缘关系泄漏）

---

## 4. Phase 2：模型架构实现

### 4.1 PlantHGNN整体架构

```
输入层（SNP特征 + 多组学特征 + 网络结构）
    ↓
多视图GCN编码器（Module A）
    ├── GCN_PPI → z_ppi
    ├── GCN_GO → z_go
    └── GCN_Pathway → z_path
    ↓ 注意力融合
    z_fusion（可学习权重α∈R³）
    ↓
功能嵌入模块（Module B，参考GRAFT）
    ↓
图结构编码（Module C）- 随机游走 + PageRank
    ↓
特征拼接 → 投影层
    ↓
AttnRes Transformer编码器（Module D）
    ├── Transformer Layer 1 + AttnRes
    ├── ...（L层，L=4~8）
    └── AttnRes: h^(l) = Σ α_{i→l} · v_i
    ↓
回归头 → GEBV预测
```

### 4.2 核心模块实现要点

**Module A: 多视图GCN编码器**
- 三视图独立2层GCN
- 可学习注意力融合：z_fusion = Σ softmax(Z_i @ a) · z_i

**Module B: Attention Residuals（AttnRes）**
- 参考Kimi官方实现：https://github.com/MoonshotAI/Attention-Residuals
- Block数量默认8（Kimi论文推荐）
- 关键公式：h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i
- query向量是可训练参数，不依赖输入

**Module C: PlantHGNN主模型**
- 参数：d_model=128, n_transformer_layers=6, n_attnres_blocks=8
- 消融实验开关：use_heterogeneous, use_attnres, use_functional_embed
- 可解释性方法：get_network_attention_weights(), get_depth_attention_weights()

### 4.3 训练配置（base_config.yaml）
```yaml
model:
  d_model: 128
  n_transformer_layers: 6
  n_attnres_blocks: 8
  dropout: 0.2

training:
  optimizer: adamw
  lr: 1e-3
  batch_size: 32
  max_epochs: 200
  early_stopping_patience: 20
```

---

## 5. Phase 3：基线模型复现

### 5.1 基线模型清单

**优先级1（必须复现）**：
- GBLUP（统计基线）- 本地
- DNNGP（DNN）- 本地
- GPformer（Transformer）- 本地（小数据集）
- Cropformer（CNN+Attention）- [SERVER]
- NetGP（GCN）- 本地

**优先级2（建议复现）**：
- CropARNet（ResNet+Attention）- [SERVER]
- GEFormer（gMLP+LinearAttention）- 本地
- SoyDNGP（3D CNN）- 本地
- LightGBM（GBDT）- 本地

### 5.2 统一评估框架
- 所有模型继承`BaselineModel`抽象类
- 统一接口：`fit()`, `predict()`, `evaluate()`
- 使用相同的评估指标函数（`src/training/metrics.py`）

### 5.3 数据对齐要求（关键）
所有模型必须使用完全相同的：
1. 数据集划分（相同的split JSON文件）
2. 随机种子序列（[42, 123, 456, 789, 1024]）
3. 评估指标函数
4. 表型归一化方式（Z-score，基于训练集统计）

---

## 6. Phase 4：消融实验设计

### 6.1 消融组件清单

**消融组1：网络类型消融**
- Ours-PPI_only：只用PPI网络
- Ours-GO_only：只用GO网络
- Ours-no_hetero：去掉异构GTM网络
- Ours-single_view：只用单一网络（≈NetGP）

**消融组2：AttnRes消融**
- Ours-no_AttnRes：替换为标准残差连接
- Ours-standard_residual：标准Pre-Norm + 残差
- Ours-HyperConn：替换为Hyper-Connections

**消融组3：特征模块消融**
- Ours-no_FuncEmbed：去掉功能嵌入模块
- Ours-no_StructEncode：去掉结构编码
- Ours-no_SNP：只用网络结构

**消融组4：划分策略消融（方法论贡献）**
- Ours-random_split：随机划分
- Ours-chrom_split：染色体划分
- Ours-line_split：品系划分（推荐）

---

## 7. Phase 5：可解释性分析实验

### 7.1 网络贡献度分析
- 提取多视图融合注意力权重α = [α_ppi, α_go, α_path]
- 分析每个性状的网络偏好
- 预期：产量性状→偏向Pathway，抗病性状→偏向PPI，发育性状→偏向GO

### 7.2 AttnRes深度注意力分析
- 提取深度方向注意力权重
- 分析：最终表示主要来自哪几层
- 预期：复杂性状依赖更深层（全局），简单性状依赖浅层（局部）

### 7.3 SNP重要性分析（SHAP）
- 使用SHAP GradientExplainer计算SNP重要性
- 与GWAS显著位点比较
- 计算重叠率作为验证指标

### 7.4 基因聚类分析（UMAP）
- 对基因嵌入做UMAP降维
- 按GO Biological Process着色
- 验证：同一通路基因在嵌入空间中相近

### 7.5 案例研究（水稻粒重）
- 目标基因：GW5/qSW5（已知控制粒宽的主效QTL）
- 分析子图结构、注意力权重矩阵
- 与文献已知调控机制对比

---

## 8. Phase 6：补充实验与鲁棒性分析

### 8.1 小样本鲁棒性实验
- 训练集大小：10%, 20%, 30%, 50%, 80%
- 每个比例：5折CV × 5随机种子 = 25次实验
- 与GBLUP、DNNGP、NetGP对比

### 8.2 跨作物泛化实验
- 在水稻上训练，在小麦上微调/直接预测（零样本）
- 验证生物网络知识的可迁移性

### 8.3 Wilcoxon统计检验
- 对所有比较实验做Wilcoxon signed-rank test
- p < 0.05为显著优势
- 在结果表格中标注 * / ** / ***

---

## 9. 服务器任务汇总（[SERVER]）

### [SERVER-1] 大规模数据预处理
```bash
# rice3k和wheat2403完整预处理
python src/data/preprocess.py --dataset rice3k wheat2403
```

### [SERVER-2] 大规模基线模型复现
```bash
# Cropformer、CropARNet、GEFormer在所有数据集上
for model in cropformer croparnet geformer; do
    for dataset in rice469 maize282 soybean999 wheat599 rice3k wheat2403; do
        python experiments/run_baseline.py --model $model --dataset $dataset
    done
done
```

### [SERVER-3] 超参数搜索
```bash
# 搜索空间：d_model, n_transformer_layers, lr, dropout, n_attnres_blocks
python experiments/hyperparameter_search.py --n_trials 100
```

### [SERVER-4] 完整主实验
```bash
# 6数据集 × 5折 × 5种子
python experiments/run_main_experiment.py --model plant_hgnn --n_folds 5 --seeds 42 123 456 789 1024
```

### [SERVER-5] 扩展消融实验
```bash
# 所有消融配置在全量数据集上
for config in experiments/configs/ablation/*.yaml; do
    python experiments/run_experiment.py --config $config --datasets rice469 maize282 soybean999 wheat599
done
```

---

## 10. 结果处理与论文表格生成

### 10.1 主结果表格格式
**Table 1：主实验对比结果**
- 列：GBLUP | DNNGP | GPformer | Cropformer | NetGP | PlantHGNN(Ours)
- 行：rice469 | maize282 | soybean999 | wheat599 | rice3k | wheat2403
- 指标：PCC (mean ± std) | NDCG@10
- 显著性标注：* p<0.05, ** p<0.01, *** p<0.001

### 10.2 本地结果分析流程
```bash
# 从服务器同步结果
rsync -avz server:/path/to/GWAS/experiments/results/ ./experiments/results/

# 生成表格和图表
python experiments/generate_tables.py
python src/analysis/visualization.py
```

---

## 11. 开发顺序与里程碑

### 里程碑1：数据就绪（1-2周）
- [ ] T1.1 下载CropGS-Hub数据集（rice469, maize282, soybean999, wheat599）
- [ ] T1.2 下载生物网络数据（STRING-plant, GO, KEGG, PlantTFDB）
- [ ] T1.3 实现PCS特征选择（`src/data/preprocess.py`）
- [ ] T1.4 实现网络构建流水线（`src/data/network_builder.py`）
- [ ] T1.5 实现三种数据划分策略（`src/data/splits.py`）
- [ ] T1.6 **验证**：在rice469上运行完整预处理，输出PyG格式图数据

### 里程碑2：基线可复现（1-2周）
- [ ] T2.1 实现/封装GBLUP（`src/models/baselines/gblup.py`）
- [ ] T2.2 复现DNNGP（`src/models/baselines/dnngp.py`）
- [ ] T2.3 复现NetGP（`src/models/baselines/netgp.py`）
- [ ] T2.4 复现GPformer（`src/models/baselines/gpformer.py`）
- [ ] T2.5 **验证**：在rice469上所有基线PCC与原文报告数字相差<0.02

### 里程碑3：主模型可训练（1-2周）
- [ ] T3.1 实现AttnRes（`src/models/attention_residual.py`）
- [ ] T3.2 实现多视图GCN编码器（`src/models/multi_view_gcn.py`）
- [ ] T3.3 实现功能嵌入模块（`src/models/functional_embed.py`）
- [ ] T3.4 集成主模型PlantHGNN（`src/models/plant_hgnn.py`）
- [ ] T3.5 实现训练主循环（`src/training/trainer.py`）
- [ ] T3.6 **验证**：在rice469上PlantHGNN能正常训练10个epoch，loss下降

### 里程碑4：主实验结果（服务器，2-3周）
- [ ] T4.1 [SERVER-3] 超参数搜索，确定最优配置
- [ ] T4.2 [SERVER-2] 大规模基线复现
- [ ] T4.3 [SERVER-4] 主实验（6数据集×5折×5种子）
- [ ] T4.4 本地生成主结果表格
- [ ] T4.5 **验证**：PlantHGNN在≥4/6个数据集上显著优于NetGP（p<0.05）

### 里程碑5：分析实验（1-2周）
- [ ] T5.1 [SERVER-5] 消融实验
- [ ] T5.2 本地可解释性分析（AttnRes权重、网络贡献、SNP SHAP）
- [ ] T5.3 UMAP可视化
- [ ] T5.4 小样本鲁棒性实验
- [ ] T5.5 统计显著性检验（Wilcoxon）

### 里程碑6：论文写作（2-3周）
- [ ] T6.1 生成所有论文图表（paper/figures/, paper/tables/）
- [ ] T6.2 写作（Introduction + Related Work + Method + Experiment + Discussion）
- [ ] T6.3 目标期刊投稿格式调整

---

## 12. 依赖环境

```txt
# requirements.txt核心依赖
torch>=2.0.0
torch-geometric>=2.4.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
umap-learn>=0.5.3
shap>=0.42.0
goatools>=1.3.1
networkx>=3.1
pyyaml>=6.0
tqdm>=4.65.0
```

```bash
# 环境创建
conda create -n planthgnn python=3.10
conda activate planthgnn
pip install -r requirements.txt

# PyG安装（CUDA 12.1）
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

---

## 13. 关键注意事项

### 13.1 公平比较原则（审稿人必看）
1. 所有模型使用**相同的预处理特征**（PCS降维后的SNP）
2. 所有模型使用**相同的数据划分索引**（split JSON文件）
3. GBLUP作为必须基线，任何深度学习方法必须优于GBLUP才有意义
4. 报告 **mean ± std**（5折×5种子），不只报单次结果

### 13.2 与NetGP的差异化（审稿人会问）
| 维度 | NetGP | PlantHGNN（Ours） |
|---|---|---|
| GNN类型 | 简单GCN | 多视图GCN + 注意力融合 |
| 网络类型 | 仅同构 | 同构 + 异构（GTM元路径） |
| Transformer | 无 | AttnRes Transformer |
| 残差机制 | 无（FC层堆叠） | Kimi AttnRes（自适应深度聚合） |
| 可解释性 | 特征重要性 | 多维度（网络贡献+深度注意力+SHAP） |
| 数据集数量 | 4 | 6（覆盖更多作物） |

### 13.3 AttnRes实现参考
- 官方GitHub: https://github.com/MoonshotAI/Attention-Residuals
- 官方论文: arxiv 2603.15031
- 关键实现：query向量是可训练参数，不依赖当前输入
- Block数量默认8，与mHC标准对齐

### 13.4 避免数据泄漏（来自GPCR工作的教训）
- 在论文中专门设计"划分策略对比实验"一节
- 展示随机划分vs品系划分的PCC差异
- 这本身就是一个方法论贡献，增加创新点

---

## 14. Git提交规范

```bash
# 功能开发
git commit -m "feat: implement MultiViewGCNEncoder"
git commit -m "feat: implement BlockAttnRes based on MoonshotAI implementation"

# 实验结果
git commit -m "exp: baseline DNNGP reproduced on rice469, PCC=0.48"
git commit -m "exp: main experiment results on 6 datasets"

# 修复
git commit -m "fix: batch normalization after each GCN layer"

# 分析
git commit -m "analysis: network contribution heatmap for wheat599"

# 论文
git commit -m "paper: Figure 1 model architecture diagram"
```

---

*精简版创建：2026-03-29*
*完整详细版见：CLAUDE_FULL.md*
*作者：Lyu（安徽农业大学 AI学院 智慧农业重点实验室）*
*GitHub：https://github.com/nblvguohao/GWAS.git*