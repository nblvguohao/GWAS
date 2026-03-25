# PlantHGNN: 植物性状预测异构图神经网络研究方案

> **Claude Code 执行指南**：本文件是完整的科研方案，请按 Phase 顺序执行。
> 每个 Task 包含明确的输入、输出和验证标准。
> 标记 `[SERVER]` 的任务需在远程服务器运行，其余可在本地 4060 执行。

---

## 0. 项目概览

### 0.1 研究定位

**项目名称**：PlantHGNN — 基于异构图神经网络与自适应残差的植物多性状基因组预测框架

**核心创新**：将癌症驱动基因预测领域（TREE/GRAFT）的技术体系首次系统性迁移至植物育种基因组预测（GP/GWAS），结合 Kimi Attention Residuals（AttnRes）实现多网络证据的自适应深度聚合。

**技术栈对应关系**：

| 癌症基因预测（源域） | 植物GP预测（目标域） |
|---|---|
| PPI 网络 | 植物蛋白互作网络（STRING-plant） |
| GO 语义相似性网络 | 植物 GO 功能注释网络 |
| KEGG 通路共现网络 | 植物代谢通路关联网络 |
| TF-miRNA-gene 异构网络 | PlantTFDB 调控网络 + miRBase 植物 miRNA |
| 多组学：SNV/METH/GE/CNA | 多组学：SNP/表达/甲基化（可选） |
| 节点二分类（驱动/非驱动） | 节点回归（GEBV 育种值预测） |

### 0.2 创新点评估

**创新点 1（核心）**：首个将异构图 Transformer（TREE/GRAFT 框架）应用于植物GP的工作
- 现有最高水位线 NetGP（2025）仅使用简单 GCN，无注意力融合，无异构网络
- 技术代差约 3 年

**创新点 2**：植物专用异构生物网络构建
- 设计 Gene-TF-Metabolite（GTM）元路径，对应植物转录调控和代谢调控
- 引入 LD 块拓扑结构作为 SNP 级别的图边（植物特有）

**创新点 3**：AttnRes 自适应多网络证据聚合
- 将 AttnRes（Kimi 2026）的深度方向注意力权重用于跨网络模态融合
- 每个性状自适应学习"依赖哪个生物网络"，提供可解释的网络贡献分析

**创新点 4**：多性状联合预测（可选扩展）
- 利用图结构天然支持的多任务学习框架

### 0.3 目标期刊评估

**主投目标**（按优先级排序）：

| 期刊 | IF | 理由 | 工作量要求 |
|---|---|---|---|
| **Plant Biotechnology Journal** | ~9.5 | NetGP 发表于此，直接竞争对手，编辑熟悉方向 | 中等，需 3-4 个作物数据集 |
| **Briefings in Bioinformatics** | ~9.5 | GRAFT/GPformer+KGM 发表于此，方法论导向 | 中等，注重方法严谨性 |
| **Plant Communications** | ~9.4 | Cropformer 发表于此，接受新架构工作 | 中等，需要生物学解释 |
| **Bioinformatics** | ~5.8 | 稳妥选项，方法类论文友好 | 相对低，聚焦方法即可 |
| **CAAI AIR**（备选） | ~3 | 你们已有 CANOPY-Router 投稿经验 | 低，快速发表 |

**推荐策略**：先冲 **Plant Biotechnology Journal** 或 **Briefings in Bioinformatics**，准备好后备投 Bioinformatics。

---

## 1. 硬件资源规划

### 1.1 本地配置（4060 8G VRAM + 16G RAM）

**可执行的任务**：
- 数据预处理（PCS 特征选择、网络构建、图数据序列化）
- 小规模数据集上的模型调试（rice469、maize282）
- 所有基线模型在小数据集上的复现验证
- 消融实验的小数据集版本
- 可解释性可视化（attention 热力图、UMAP 降维）

**显存约束**：
- 最大 batch size：图节点数 ≤ 5000 时可用 batch=32
- 模型参数上限：约 50M 参数以内
- SNP 序列长度：需先 PCS 降维至 ≤ 5000 个特征

### 1.2 服务器任务清单 `[SERVER]`

以下实验**必须在服务器运行**，标记后在方案中单独列出：

```
[SERVER-1] 大规模数据集训练（maize1404 含14M SNP的预处理）
[SERVER-2] 完整基线模型复现（Cropformer、CropARNet在全量数据集）
[SERVER-3] 超参数搜索（learning rate × dropout × GCN层数 网格搜索）
[SERVER-4] 5折交叉验证的全量实验（每个数据集×每个模型）
[SERVER-5] 大规模网络构建（STRING植物版图构建，>10万节点）
```

**服务器推荐配置**：AutoDL A100 40G × 1，约需 80-120 GPU小时

---

## 2. 项目结构与 GitHub 配置

### 2.1 目录结构

```
GWAS/                          # GitHub repo: https://github.com/nblvguohao/GWAS.git
├── CLAUDE.md                  # 本文件（研究方案主文档）
├── README.md                  # 项目简介
├── requirements.txt           # Python 依赖
├── setup.py
│
├── data/                      # 数据目录（不入 git，用 .gitignore 排除原始数据）
│   ├── raw/                   # 原始下载数据
│   │   ├── cropgs/            # CropGS-Hub 下载
│   │   ├── networks/          # 生物网络数据
│   │   └── annotations/       # GO/KEGG 注释
│   ├── processed/             # 预处理后数据
│   │   ├── graphs/            # PyG 格式图数据
│   │   └── splits/            # 数据集划分索引
│   └── README.md              # 数据来源说明
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py        # 数据自动下载脚本
│   │   ├── preprocess.py      # SNP 预处理（PCS 特征选择）
│   │   ├── network_builder.py # 生物网络构建
│   │   ├── graph_dataset.py   # PyG Dataset 类
│   │   └── splits.py          # 数据集划分（随机/染色体/品系）
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── plant_hgnn.py      # 主模型 PlantHGNN
│   │   ├── attention_residual.py  # AttnRes 实现
│   │   ├── multi_view_gcn.py  # 多视图 GCN 编码器
│   │   ├── heterogeneous.py   # 异构图 Transformer（TREE style）
│   │   ├── functional_embed.py # 功能嵌入模块（GRAFT style）
│   │   └── baselines/         # 所有基线模型
│   │       ├── gblup.py
│   │       ├── dnngp.py
│   │       ├── soydngp.py
│   │       ├── gpformer.py
│   │       ├── cropformer.py
│   │       ├── netgp.py
│   │       └── geformer.py
│   │
│   ├── training/
│   │   ├── trainer.py         # 训练主循环
│   │   ├── losses.py          # 损失函数
│   │   └── metrics.py         # 评估指标（PCC、NDCG、AUPR）
│   │
│   └── analysis/
│       ├── interpretability.py # 注意力权重分析
│       ├── network_contrib.py  # 网络贡献度分析
│       ├── snp_importance.py   # SNP 重要性（SHAP）
│       └── visualization.py    # 可视化工具
│
├── experiments/
│   ├── configs/               # YAML 实验配置文件
│   │   ├── base_config.yaml
│   │   ├── ablation/          # 消融实验配置
│   │   └── datasets/          # 各数据集专用配置
│   ├── scripts/               # 实验运行脚本
│   │   ├── run_local.sh       # 本地运行
│   │   └── run_server.sh      # 服务器运行 [SERVER]
│   └── results/               # 实验结果（入 git，只存 CSV/JSON）
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_network_analysis.ipynb
│   ├── 03_results_analysis.ipynb
│   └── 04_visualization.ipynb
│
└── paper/
    ├── figures/               # 论文图表
    └── tables/                # 论文表格
```

### 2.2 Git 初始化任务

```bash
# Task G1: 初始化项目结构
git clone https://github.com/nblvguohao/GWAS.git
cd GWAS
# 创建上述目录结构
# 添加 .gitignore（排除 data/raw/, data/processed/, __pycache__ 等）
# 创建 requirements.txt
git add .
git commit -m "feat: initialize PlantHGNN project structure"
git push
```

---

## 3. Phase 1：数据集构建与预处理

### 3.1 核心数据集选择

选择标准：覆盖不同作物（多样性）、不同样本量（小/中/大）、有公开多组学数据。

**主实验数据集（必须）**：

| 数据集 | 作物 | 样本数 | SNP数 | 性状数 | 来源 | 本地/服务器 |
|---|---|---|---|---|---|---|
| rice469 | 水稻 | 469 | 5,291 | 6 | CropGS-Hub / GPformer | 本地 ✓ |
| maize282 | 玉米 | 282 | 3,093 | 3 | PANZEA / CropGS-Hub | 本地 ✓ |
| soybean999 | 大豆 | 999 | 7,883 | 6 | SoyDNGP / 原文 | 本地 ✓ |
| wheat599 | 小麦 | 599 | 1,447 | 3 | CIMMYT / GPformer | 本地 ✓ |
| rice3k | 水稻 | 3,000 | ~4M→PCS降维 | 分类 | Rice3K / IJCAI-24 | 本地可跑预处理 [SERVER-1]训练 |

**扩展数据集（增强说服力，至少选2个）**：

| 数据集 | 作物 | 样本数 | 性状 | 来源 |
|---|---|---|---|---|
| peanut399 | 花生 | 399 | 多 | NetGP 原文 |
| millet (谷子) | 谷子 | ~900 | 产量/株高 | Cropformer 原文 |
| cotton | 棉花 | ~400 | 纤维品质 | CropARNet 原文 |
| wheat2403 | 小麦 | 2,403 | 6 | DNNGP 原文 |

### 3.2 生物网络数据下载

```bash
# Task D1: 下载植物蛋白互作网络
# STRING v12 plant-specific (Arabidopsis/Rice/Maize/Soybean)
# URL: https://string-db.org/cgi/download?species_text=Oryza+sativa
# 过滤条件：combined_score > 700

# Task D2: 下载 GO 注释
# 来源：https://current.geneontology.org/annotations/
# 物种：rice (IRGSP-1.0), maize (B73), soybean (Wm82.a4)

# Task D3: 下载 KEGG 植物通路
# 使用 KEGG REST API
# python src/data/download.py --kegg --species osa mze gmx

# Task D4: 下载植物转录因子数据库
# PlantTFDB v5: http://planttfdb.gao-lab.org/download.php
# 下载 TF-target 调控关系文件

# Task D5: 下载植物 miRNA 数据
# miRBase v22: https://www.mirbase.org/ftp/CURRENT/
# 植物 miRNA-target: psRNAtarget 数据库

# Task D6: CropGS-Hub 数据下载
# URL: https://iagr.genomics.cn/CropGS
# 下载 rice469, maize282, wheat599 的基因型+表型文件
```

### 3.3 SNP 预处理流水线

```python
# Task P1: src/data/preprocess.py
# 实现以下流水线：

# Step 1: 质量控制（PLINK 格式）
# - 过滤缺失率 > 10% 的 SNP
# - 过滤 MAF < 0.05 的 SNP
# - 过滤缺失率 > 10% 的个体
# 工具：PLINK 1.9 或 Python plink2 封装

# Step 2: PCS 特征选择（参考 NetGP）
# - Pearson 相关过滤（|r| > 0.3 with phenotype）
# - 共线性去除（VIF < 10）
# - 输出：每个性状对应的候选 SNP 子集

# Step 3: one-hot 编码
# AA→[1,0,0], AB→[0,1,0], BB→[0,0,1], Missing→[0,0,0]
# 或使用 8-dimensional 编码（DPCformer 方案）

# Step 4: 按染色体排序
# SNP 按物理位置（染色体:位置）排序，保留空间信息

# 输出格式：
# - snp_matrix.npy: (n_individuals, n_snps, 3)
# - phenotype.csv: (n_individuals, n_traits)
# - snp_metadata.csv: snp_id, chromosome, position, maf
```

### 3.4 生物网络构建

```python
# Task P2: src/data/network_builder.py
# 构建三类同构网络 + 一类异构网络

# === 同构网络 ===

# Network 1: 蛋白互作网络（PPI）
# - 节点：基因
# - 边：STRING 互作，score > 700
# - 边权重：标准化 combined_score
# 预期规模：水稻 ~6000 基因，~50000 边

# Network 2: GO 功能相似性网络
# - 节点：基因
# - 边：GO 语义相似性（GOSemSim 算法）
# - 过滤：相似性 > 0.8
# 工具：Python goatools 库

# Network 3: KEGG 通路共现网络
# - 节点：基因
# - 边：共同出现在同一通路中
# - 边权重：共现通路数 / 总通路数（Jaccard）
# 过滤：weight > 0.1

# === 异构网络 GTM（Gene-TF-Metabolite）===
# 节点类型：gene, TF, metabolite
# 边类型：
#   gene-gene (来自 PPI)
#   TF-gene (来自 PlantTFDB)
#   TF-metabolite (TF 调控代谢酶基因)
#   gene-metabolite (代谢 GWAS 关联，可选)

# 元路径定义：
# - GG: gene-gene (同构基础路径)
# - GTG: gene-TF-gene (共同被同一TF调控)
# - GTMTG: gene-TF-metabolite-TF-gene (通过代谢枢纽连接)

# 输出格式：PyTorch Geometric HeteroData 对象
# 保存：data/processed/graphs/{species}_{network_type}.pt
```

### 3.5 数据集划分策略

```python
# Task P3: src/data/splits.py
# 实现三种划分策略（参考 GPCR 泄漏问题的教训）

# Strategy 1: 随机划分（Random Split）
# 80% train, 10% val, 10% test，5折交叉验证
# 注意：可能存在品系间亲缘关系泄漏

# Strategy 2: 染色体划分（Chromosome Split）
# 按染色体保留测试集（leave-one-chromosome-out）
# 评估跨染色体的泛化能力，更严格

# Strategy 3: 品系划分（Line Split）
# 基于系谱或群体结构（STRUCTURE软件结果）
# 确保训练集和测试集中的品系无直接亲缘关系
# 推荐作为主要划分策略（更符合真实育种场景）

# 实现要求：
# - 所有划分保存为 JSON 索引文件
# - 支持 stratified split（保持表型分布）
# - 输出 split_summary.csv（每折样本数统计）
```

---

## 4. Phase 2：模型架构实现

### 4.1 PlantHGNN 整体架构

```
输入层
 ├── SNP 特征矩阵 (n_genes × d_snp)          ← PCS 降维后
 ├── 多组学特征（可选：表达量、甲基化）
 └── 网络结构（PPI, GO, Pathway, GTM异构图）
         ↓
多视图 GCN 编码器（Module A）
 ├── GCN_PPI: 提取互作网络特征 → z_ppi ∈ R^d
 ├── GCN_GO: 提取功能网络特征 → z_go ∈ R^d
 └── GCN_Pathway: 提取通路网络特征 → z_path ∈ R^d
         ↓ 注意力融合
     z_fusion ∈ R^d  （可学习权重 α ∈ R^3）
         ↓
功能嵌入模块（Module B，参考 GRAFT）
 └── 基于 MSigDB/PlantCyc 基因集的功能表示 e_gene ∈ R^d
         ↓
图结构编码（Module C）
 ├── 随机游走位置编码（局部拓扑）
 └── PageRank 中心性（全局重要性）→ e_struct ∈ R^{d+1}
         ↓
特征拼接
 h_input = [x_snp ⊕ z_fusion ⊕ e_gene ⊕ e_struct]
         ↓
AttnRes Transformer 编码器（Module D）
 ├── 投影层: h_input → h^(0) ∈ R^d
 ├── Transformer Layer 1 + AttnRes
 ├── Transformer Layer 2 + AttnRes
 ├── ...（L层，L=4~8）
 └── AttnRes: h^(l) = Σ α_{i→l} · v_i（软件注意力加权聚合）
         ↓
预测头
 ├── 回归头（GEBV 预测）: Linear → R^n_traits
 └── 分类头（QTL/性状分类，可选）: Linear → Softmax
         ↓
输出: 育种估计值（GEBV）
```

### 4.2 核心模块实现

#### Module A: 多视图 GCN 编码器

```python
# Task M1: src/models/multi_view_gcn.py
# 参考 GRAFT 的 multi_view_graph_encoding 模块

class MultiViewGCNEncoder(nn.Module):
    """
    三视图 GCN 编码器，分别处理 PPI/GO/Pathway 网络
    输出通过可学习注意力向量融合
    """
    def __init__(self, in_dim, hidden_dim, out_dim, n_views=3):
        # 每个视图独立的 2层 GCN
        # GCN: h^(1) = ReLU(A_norm W1 x)
        # GCN: z^(m) = A_norm W2 h^(1)
        # 注意力融合: z_fusion = Σ softmax(Z_i @ a) · z_i^(m)
        pass

    def forward(self, x, adj_list):
        # adj_list: [adj_ppi, adj_go, adj_pathway]
        # 返回: z_fusion, attention_weights
        pass
```

#### Module B: Attention Residuals

```python
# Task M2: src/models/attention_residual.py
# 实现 Kimi AttnRes（arxiv 2603.15031）
# GitHub 参考: https://github.com/MoonshotAI/Attention-Residuals

class BlockAttnRes(nn.Module):
    """
    Block Attention Residuals
    将层分为 N 个 block，每个 block 内用标准残差
    跨 block 用注意力加权聚合（query 是可训练参数，不依赖输入）
    
    关键公式：
    h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i
    其中 α = softmax(q_l K^T / sqrt(d))
    q_l 是该层的可训练参数向量
    """
    def __init__(self, d_model, n_blocks=8):
        # n_blocks: 默认 8（Kimi 论文推荐）
        # 每个 block 的 query 向量：trainable, shape (d_model,)
        pass

    def forward(self, layer_outputs):
        # layer_outputs: list of hidden states from previous layers
        # 返回: 加权聚合的当前层输入
        pass

    def get_block_weights(self):
        # 返回各 block 的注意力权重，用于可解释性分析
        pass
```

#### Module C: PlantHGNN 主模型

```python
# Task M3: src/models/plant_hgnn.py

class PlantHGNN(nn.Module):
    """
    植物异构图神经网络 - 主模型
    
    Args:
        n_snps: PCS 选择后的 SNP 数量
        d_model: 隐层维度（默认 128）
        n_transformer_layers: Transformer 层数（默认 6）
        n_attnres_blocks: AttnRes block 数量（默认 8）
        n_traits: 预测性状数量
        use_heterogeneous: 是否启用异构网络（消融实验开关）
        use_attnres: 是否启用 AttnRes（消融实验开关）
        use_functional_embed: 是否启用功能嵌入（消融实验开关）
    """
    def __init__(self, ...):
        self.snp_encoder = SNPEncoder(n_snps, d_model)
        self.multi_view_gcn = MultiViewGCNEncoder(d_model, d_model, d_model)
        self.functional_embed = FunctionalEmbedding(n_gene_sets, d_model)
        self.struct_encoder = StructuralEncoder(d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model) for _ in range(n_transformer_layers)
        ])
        self.attn_res = BlockAttnRes(d_model, n_attnres_blocks)
        self.regression_head = nn.Linear(d_model, n_traits)

    def forward(self, snp_data, graph_data):
        # 1. SNP 编码
        # 2. 多视图 GCN 编码 + 融合
        # 3. 功能嵌入
        # 4. 结构编码
        # 5. 拼接 → 投影
        # 6. AttnRes Transformer 逐层处理
        # 7. 回归头输出
        pass

    def get_network_attention_weights(self):
        # 返回多视图融合的注意力权重 α (3维向量)
        # 用于分析：不同性状依赖哪个生物网络
        pass

    def get_depth_attention_weights(self):
        # 返回 AttnRes 的深度方向注意力权重
        # 用于分析：最终表示主要来自第几层
        pass
```

### 4.3 训练配置

```yaml
# experiments/configs/base_config.yaml

model:
  d_model: 128
  n_transformer_layers: 6
  n_attnres_blocks: 8
  n_gcn_layers: 2
  dropout: 0.2
  use_heterogeneous: true
  use_attnres: true
  use_functional_embed: true

training:
  optimizer: adamw
  lr: 1e-3
  weight_decay: 1e-4
  batch_size: 32           # 本地 4060 可用
  max_epochs: 200
  early_stopping_patience: 20
  loss: mse                # 回归任务
  scheduler: cosine_annealing

evaluation:
  metrics:
    - pearson_r            # 主要指标，与现有文献对齐
    - spearman_r           # 秩相关
    - mse
    - mae
    - ndcg_at_10           # 排名指标，育种选择场景更实用
  n_folds: 5
  n_seeds: [42, 123, 456, 789, 1024]  # 5 个随机种子

hardware:
  device: cuda
  num_workers: 4
  pin_memory: true
  gradient_checkpointing: false  # 本地跑小数据集时关闭
```

---

## 5. Phase 3：基线模型复现

### 5.1 基线模型清单

**优先级 1（必须复现）**：

| 模型 | 类型 | 代码来源 | 运行环境 |
|---|---|---|---|
| GBLUP | 统计基线 | rrBLUP R包 / Python实现 | 本地 ✓ |
| DNNGP | DNN | https://github.com/AIBreeding/DNNGP | 本地 ✓ |
| GPformer | Transformer | 原文代码（BIB 2024） | 本地 ✓（小数据集）|
| Cropformer | CNN+Attention | https://cgris.net/cropformer | [SERVER-2] |
| NetGP | GCN | 原文代码（PBJ 2025） | 本地 ✓ |

**优先级 2（建议复现）**：

| 模型 | 类型 | 代码来源 | 运行环境 |
|---|---|---|---|
| CropARNet | ResNet+Attention | 原文代码 | [SERVER-2] |
| GEFormer | gMLP+LinearAttention | 原文代码 | 本地 ✓ |
| SoyDNGP | 3D CNN | https://github.com 搜索 | 本地 ✓ |
| LightGBM | GBDT | pip install lightgbm | 本地 ✓ |

### 5.2 统一评估框架

```python
# Task B1: src/models/baselines/ 下各基线的统一接口

class BaselineModel(ABC):
    """所有基线模型的统一接口"""
    
    @abstractmethod
    def fit(self, X_train, y_train, graph_data=None):
        pass
    
    @abstractmethod
    def predict(self, X_test, graph_data=None):
        pass
    
    def evaluate(self, X_test, y_test, graph_data=None):
        # 统一调用 metrics.py 中的评估函数
        pass

# 要求：每个基线模型包装为 BaselineModel 子类
# 便于在 experiments/run_all_baselines.py 中统一调用
```

### 5.3 数据对齐要求

```python
# 关键：所有模型必须使用完全相同的：
# 1. 数据集划分（相同的 split JSON 文件）
# 2. 随机种子序列（[42, 123, 456, 789, 1024]）
# 3. 评估指标函数（src/training/metrics.py）
# 4. 表型归一化方式（Z-score，基于训练集统计）
# 违反此原则会导致不公平比较，审稿人必然指出
```

---

## 6. Phase 4：消融实验设计

### 6.1 消融组件清单

```
PlantHGNN 完整版（Ours-Full）
    ↓ 逐步移除各组件

消融组 1：网络类型消融
  - Ours-PPI_only：只用 PPI 网络（去掉 GO + Pathway）
  - Ours-GO_only：只用 GO 网络
  - Ours-no_hetero：去掉异构 GTM 网络，只用同构三网络
  - Ours-single_view：只用单一网络（GCN 最简单版），≈ NetGP

消融组 2：AttnRes 消融
  - Ours-no_AttnRes：将 AttnRes 替换回标准残差连接
  - Ours-standard_residual：标准 Pre-Norm + 残差（标准 Transformer）
  - Ours-HyperConn：将 AttnRes 替换为 Hyper-Connections（对比验证）

消融组 3：特征模块消融
  - Ours-no_FuncEmbed：去掉功能嵌入模块（Module B）
  - Ours-no_StructEncode：去掉结构编码（PageRank + RandomWalk）
  - Ours-no_SNP：只用网络结构，不用 SNP 特征

消融组 4：融合策略消融
  - Ours-concat_fusion：三视图特征直接拼接（固定权重）
  - Ours-mean_fusion：三视图取平均
  - Ours-attn_fusion：可学习注意力融合（完整版）

消融组 5：划分策略消融（方法论贡献）
  - Ours-random_split：使用随机划分
  - Ours-chrom_split：使用染色体划分
  - Ours-line_split：使用品系划分（推荐，展示泄漏风险）
```

### 6.2 消融实验配置文件模板

```yaml
# experiments/configs/ablation/no_attnres.yaml
extends: base_config.yaml

model:
  use_attnres: false        # 关键开关
  # 其余参数与 base_config 相同

experiment_name: ablation_no_attnres
output_dir: experiments/results/ablation/no_attnres/
```

### 6.3 消融实验运行脚本

```bash
# experiments/scripts/run_ablation.sh
# 在本地 4060 上运行所有消融实验（使用 rice469 和 maize282）

for config in experiments/configs/ablation/*.yaml; do
    python experiments/run_experiment.py \
        --config $config \
        --dataset rice469 maize282 \
        --n_seeds 5 \
        --device cuda
done

# 汇总结果
python experiments/summarize_ablation.py \
    --results_dir experiments/results/ablation/ \
    --output paper/tables/ablation_table.csv
```

---

## 7. Phase 5：可解释性分析实验

### 7.1 实验 1：网络贡献度分析

**目标**：分析不同性状依赖哪个生物网络（PPI/GO/Pathway），验证生物合理性

```python
# Task I1: src/analysis/network_contrib.py

def analyze_network_contribution(model, dataset, trait_names):
    """
    提取多视图融合的注意力权重 α = [α_ppi, α_go, α_path]
    分析每个性状的网络偏好
    
    预期发现（生物学假设）：
    - 产量相关性状 → 偏向 Pathway 网络（代谢通路重要）
    - 抗病相关性状 → 偏向 PPI 网络（蛋白互作重要）
    - 发育性状 → 偏向 GO 网络（基因功能注释重要）
    """
    weights = model.get_network_attention_weights()
    # 输出：热力图（性状 × 网络类型）
    # 保存：paper/figures/network_contribution.pdf
```

**对应论文图**：类似 GRAFT Figure 5（Average attention weights assigned to three networks）

### 7.2 实验 2：AttnRes 深度注意力分析

**目标**：可视化不同 block 的贡献权重，解释模型在哪一层"做决定"

```python
# Task I2: src/analysis/interpretability.py

def analyze_depth_attention(model, dataset):
    """
    提取 AttnRes 的深度方向注意力权重
    分析：最终表示主要来自哪几层（局部 vs 全局特征）
    
    预期发现：
    - 与 TREE 发现类似：同构网络偏向远层（长程依赖）
    - 育种场景：产量等复杂性状依赖更深层（全局）
    - 简单数量性状可能主要依赖浅层（局部 LD 结构）
    """
    depth_weights = model.get_depth_attention_weights()
    # 输出：热力图（block × 性状）
    # 保存：paper/figures/depth_attention_heatmap.pdf
```

### 7.3 实验 3：SNP 重要性分析（SHAP）

**目标**：识别模型认为最重要的 SNP，与 GWAS 显著位点比较

```python
# Task I3: src/analysis/snp_importance.py

def compute_snp_shap(model, dataset, trait_name):
    """
    使用 SHAP GradientExplainer 计算每个 SNP 的重要性
    与传统 GWAS 结果（Manhattan plot）对比验证
    
    步骤：
    1. 用 SHAP 计算模型对各 SNP 的梯度重要性
    2. 提取 top-100 重要 SNP
    3. 与已知 QTL/基因注释数据库（Rice Annotation Project）比较
    4. 计算重叠率作为验证指标
    
    输出：
    - Manhattan plot（X轴染色体位置，Y轴SHAP重要性）
    - top SNP 列表 + 已知基因注释
    """
    pass
```

### 7.4 实验 4：基因聚类分析（UMAP）

**目标**：可视化学习到的基因嵌入空间，验证功能相似的基因是否聚类

```python
# Task I4: src/analysis/visualization.py

def plot_gene_embedding_umap(model, dataset):
    """
    对最终基因嵌入做 UMAP 降维，按功能注释着色
    预期：同一通路的基因在嵌入空间中相近
    参考：GRAFT Figure 2 (UMAP visualization)
    """
    embeddings = model.get_gene_embeddings()
    # UMAP 降维 → 散点图
    # 按 GO Biological Process 着色
    # 保存：paper/figures/gene_embedding_umap.pdf
```

### 7.5 实验 5：案例研究（Case Study）

**目标**：对 1-2 个重要基因做详细分析，参考 TREE Figure 4 (TET2/TP53 案例)

```
选取性状：以水稻粒重（Grain Weight）为例
分析目标基因：GW5/qSW5（已知控制粒宽的主效 QTL）

步骤：
1. 提取该基因的子图结构（邻居基因、TF 调控者）
2. 可视化注意力权重矩阵（基因对 × 注意力强度）
3. 解释模型如何通过网络路径识别该基因
4. 与文献中已知的调控机制对比

输出图：子图网络可视化 + 注意力热力图
（类似 TREE Figure 4d, 4e 的格式）
```

---

## 8. Phase 6：补充实验与鲁棒性分析

### 8.1 小样本鲁棒性实验

```python
# 模拟真实育种中样本稀缺场景
# 训练集大小：10%, 20%, 30%, 50%, 80%
# 对每个样本比例：5折CV × 5随机种子 = 25次实验
# 与 GBLUP、DNNGP、NetGP 对比

# 预期：本模型在小样本下优势更明显（图网络引入先验知识）
```

### 8.2 跨作物泛化实验

```python
# 在水稻上训练，在小麦上微调/直接预测（零样本）
# 验证模型学到的生物网络知识的可迁移性
# 这是 NetGP 未做但非常有说服力的实验
```

### 8.3 Wilcoxon 统计检验

```python
# Task S1: src/training/metrics.py
# 对所有比较实验做 Wilcoxon signed-rank test
# p < 0.05 为显著优势
# 在结果表格中标注 * / ** / ***
# 参考：GRAFT Table 2, 3 的格式
```

---

## 9. 服务器任务汇总

以下任务需要在服务器（推荐 AutoDL A100 40G）上执行：

### [SERVER-1] 大规模数据预处理

```bash
# 运行环境：服务器
# 估计时间：4-8小时
# 显存需求：CPU密集，不需要GPU

# 任务：rice3k 和 wheat2403 的完整预处理
python src/data/preprocess.py \
    --dataset rice3k wheat2403 \
    --pcs_threshold 0.3 \
    --vif_threshold 10 \
    --output_dir data/processed/

# 任务：STRING 植物版完整图构建（>10万节点）
python src/data/network_builder.py \
    --species rice maize soybean wheat \
    --string_cutoff 700 \
    --output_dir data/processed/graphs/
```

### [SERVER-2] 大规模基线模型复现

```bash
# 运行环境：服务器
# 估计时间：20-40小时（含所有数据集+5折CV）
# 显存需求：16G+

for model in cropformer croparnet geformer; do
    for dataset in rice469 maize282 soybean999 wheat599 rice3k wheat2403; do
        python experiments/run_baseline.py \
            --model $model \
            --dataset $dataset \
            --n_folds 5 \
            --n_seeds 5 \
            --device cuda \
            --output_dir experiments/results/baselines/${model}/${dataset}/
    done
done
```

### [SERVER-3] 超参数搜索

```bash
# 运行环境：服务器
# 估计时间：10-20小时
# 显存需求：16G+

python experiments/hyperparameter_search.py \
    --dataset rice469 maize282 \
    --search_space experiments/configs/hparam_search.yaml \
    --n_trials 100 \
    --device cuda \
    --output_dir experiments/results/hparam_search/

# 搜索空间（hparam_search.yaml）：
# d_model: [64, 128, 256]
# n_transformer_layers: [2, 4, 6, 8]
# lr: [1e-4, 5e-4, 1e-3]
# dropout: [0.1, 0.2, 0.3]
# n_attnres_blocks: [4, 8, 16]
```

### [SERVER-4] 完整主实验（5折×5种子）

```bash
# 运行环境：服务器
# 估计时间：30-50小时
# 显存需求：16G+（可降为 batch_size=16 以适配 8G）

python experiments/run_main_experiment.py \
    --model plant_hgnn \
    --datasets rice469 maize282 soybean999 wheat599 rice3k wheat2403 \
    --n_folds 5 \
    --seeds 42 123 456 789 1024 \
    --split_strategy line_split \
    --device cuda \
    --output_dir experiments/results/main_experiment/

# 输出：
# - experiments/results/main_experiment/summary.csv（汇总表格）
# - experiments/results/main_experiment/{dataset}/fold_{k}/predictions.csv
```

### [SERVER-5] 扩展消融实验（全量数据集）

```bash
# 运行环境：服务器
# 估计时间：15-25小时

for config in experiments/configs/ablation/*.yaml; do
    python experiments/run_experiment.py \
        --config $config \
        --datasets rice469 maize282 soybean999 wheat599 \
        --n_seeds 5 \
        --device cuda
done
```

---

## 10. 结果处理与论文表格生成

### 10.1 主结果表格格式

```python
# Task R1: experiments/generate_tables.py
# 生成论文 Table 1（主实验对比结果）

# 格式参考 GRAFT Table 2:
# 列：GBLUP | DNNGP | GPformer | Cropformer | NetGP | PlantHGNN(Ours)
# 行：rice469 | maize282 | soybean999 | wheat599 | rice3k | wheat2403
# 指标：PCC (mean ± std) | NDCG@10
# 显著性标注：* p<0.05, ** p<0.01, *** p<0.001 (vs best baseline)
```

### 10.2 本地结果分析流程

服务器跑完实验后，将 `experiments/results/` 目录同步到本地：

```bash
# 从服务器同步结果（scp 或 rsync）
rsync -avz server:/path/to/GWAS/experiments/results/ \
      ./experiments/results/

# 在本地运行分析和可视化
python experiments/generate_tables.py \
    --results_dir experiments/results/ \
    --output_dir paper/tables/

python src/analysis/visualization.py \
    --results_dir experiments/results/ \
    --output_dir paper/figures/
```

---

## 11. 开发顺序与里程碑

### 里程碑 1：数据就绪（1-2周）
- [ ] T1.1 下载 CropGS-Hub 数据集（rice469, maize282, soybean999, wheat599）
- [ ] T1.2 下载生物网络数据（STRING-plant, GO, KEGG, PlantTFDB）
- [ ] T1.3 实现 PCS 特征选择（`src/data/preprocess.py`）
- [ ] T1.4 实现网络构建流水线（`src/data/network_builder.py`）
- [ ] T1.5 实现三种数据划分策略（`src/data/splits.py`）
- [ ] T1.6 **验证**：在 rice469 上运行完整预处理，输出 PyG 格式图数据

### 里程碑 2：基线可复现（1-2周）
- [ ] T2.1 实现/封装 GBLUP（`src/models/baselines/gblup.py`）
- [ ] T2.2 复现 DNNGP（`src/models/baselines/dnngp.py`）
- [ ] T2.3 复现 NetGP（`src/models/baselines/netgp.py`）
- [ ] T2.4 复现 GPformer（`src/models/baselines/gpformer.py`）
- [ ] T2.5 **验证**：在 rice469 上所有基线 PCC 与原文报告数字相差 < 0.02

### 里程碑 3：主模型可训练（1-2周）
- [ ] T3.1 实现 AttnRes（`src/models/attention_residual.py`，参考官方实现）
- [ ] T3.2 实现多视图 GCN 编码器（`src/models/multi_view_gcn.py`）
- [ ] T3.3 实现功能嵌入模块（`src/models/functional_embed.py`）
- [ ] T3.4 集成主模型 PlantHGNN（`src/models/plant_hgnn.py`）
- [ ] T3.5 实现训练主循环（`src/training/trainer.py`）
- [ ] T3.6 **验证**：在 rice469 上 PlantHGNN 能正常训练 10个 epoch，loss下降

### 里程碑 4：主实验结果（服务器，2-3周）
- [ ] T4.1 [SERVER-3] 超参数搜索，确定最优配置
- [ ] T4.2 [SERVER-2] 大规模基线复现
- [ ] T4.3 [SERVER-4] 主实验（6数据集 × 5折 × 5种子）
- [ ] T4.4 本地生成主结果表格
- [ ] T4.5 **验证**：PlantHGNN 在 ≥ 4/6 个数据集上显著优于 NetGP（p<0.05）

### 里程碑 5：分析实验（1-2周）
- [ ] T5.1 [SERVER-5] 消融实验
- [ ] T5.2 本地可解释性分析（AttnRes 权重、网络贡献、SNP SHAP）
- [ ] T5.3 UMAP 可视化
- [ ] T5.4 小样本鲁棒性实验
- [ ] T5.5 统计显著性检验（Wilcoxon）

### 里程碑 6：论文写作（2-3周）
- [ ] T6.1 生成所有论文图表（paper/figures/, paper/tables/）
- [ ] T6.2 写作（Introduction + Related Work + Method + Experiment + Discussion）
- [ ] T6.3 目标期刊投稿格式调整

---

## 12. 依赖环境

```txt
# requirements.txt
torch>=2.0.0
torch-geometric>=2.4.0
torch-scatter>=2.1.0
torch-sparse>=0.6.18
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
wandb>=0.15.0          # 实验跟踪（可选）
optuna>=3.2.0          # 超参数搜索
plink2>=2.0.0          # SNP 质量控制（通过 subprocess 调用）
```

```bash
# 环境创建
conda create -n planthgnn python=3.10
conda activate planthgnn
pip install -r requirements.txt

# PyG 安装（需匹配 CUDA 版本）
# 4060 本地：CUDA 12.1
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

---

## 13. 关键注意事项

### 13.1 公平比较原则（审稿人必看）
1. 所有模型使用**相同的预处理特征**（PCS 降维后的 SNP）
2. 所有模型使用**相同的数据划分索引**（split JSON 文件）
3. GBLUP 作为必须基线，任何深度学习方法必须优于 GBLUP 才有意义
4. 报告 **mean ± std**（5折 × 5种子），不只报单次结果

### 13.2 与 NetGP 的差异化（审稿人会问）
| 维度 | NetGP | PlantHGNN（Ours） |
|---|---|---|
| GNN类型 | 简单 GCN | 多视图 GCN + 注意力融合 |
| 网络类型 | 仅同构（SNP-gene + 共表达） | 同构 + 异构（GTM元路径） |
| Transformer | 无 | AttnRes Transformer |
| 残差机制 | 无（FC层堆叠） | Kimi AttnRes（自适应深度聚合） |
| 可解释性 | 特征重要性 | 多维度（网络贡献+深度注意力+SHAP） |
| 数据集数量 | 4（rice,peanut,...） | 6（覆盖更多作物） |

### 13.3 AttnRes 实现参考
```
官方 GitHub: https://github.com/MoonshotAI/Attention-Residuals
官方论文: arxiv 2603.15031
关键实现：query 向量是可训练参数，不依赖当前输入
          Block 数量默认 8，与 mHC 标准对齐
```

### 13.4 避免数据泄漏（来自 GPCR 工作的教训）
- 在论文中专门设计一节"划分策略对比实验"
- 展示随机划分 vs 品系划分的 PCC 差异
- 这本身就是一个方法论贡献，增加创新点

---

## 14. Git 提交规范

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

*最后更新：2026-03-25*
*作者：Lyu（安徽农业大学 AI学院 智慧农业重点实验室）*
*GitHub：https://github.com/nblvguohao/GWAS.git*
