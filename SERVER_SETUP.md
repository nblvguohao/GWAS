# 服务器部署指南

## 1. 上传文件清单

只需上传以下目录/文件（**总计约 400MB**，不需要上传原始2.8GB raw数据）：

```
GWAS/
├── src/                              # ~342KB 模型代码
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention_residual.py
│   │   ├── multi_view_gcn.py
│   │   ├── plant_hgnn.py
│   │   ├── gene_seq_model.py         # ← 新架构
│   │   └── baselines/
│   │       ├── gblup.py
│   │       ├── dnngp.py
│   │       └── netgp.py
│   └── training/
│       └── metrics.py
├── scripts/
│   ├── server_benchmark.py           # ← 主运行脚本（相对路径）
│   └── quick_eval.py                 # 快速验证用
├── data/
│   └── processed/
│       └── gstp007/                  # ~357MB 预处理数据
│           ├── Plant_Height/         # 29MB × 7性状
│           ├── Grain_Length/
│           ├── Grain_Width/
│           ├── Days_to_Heading/
│           ├── Panicle_Length/
│           ├── Grain_Weight/
│           ├── Yield_per_plant/
│           └── graph/                # 42MB 图数据
│               ├── ppi_adj.npz       # 831×831 PPI邻接矩阵
│               ├── gene_list_v2.txt  # 基因列表
│               └── {trait}/          # 每性状基因特征
│                   ├── gene_feat_train.npy
│                   ├── gene_feat_val.npy
│                   └── gene_feat_test.npy
├── requirements_server.txt           # 精简依赖
└── SERVER_SETUP.md                   # 本文件
```

## 2. 快速上传命令

```bash
# 本地执行（Git Bash / PowerShell）
# 先打包（排除大文件）
cd E:/
tar -czf GWAS_server.tar.gz \
    GWAS/src \
    GWAS/scripts/server_benchmark.py \
    GWAS/scripts/quick_eval.py \
    GWAS/data/processed/gstp007 \
    GWAS/requirements_server.txt

# 上传到服务器（替换为你的服务器地址）
scp GWAS_server.tar.gz user@server:/path/to/

# 服务器端解压
ssh user@server
cd /path/to/
tar -xzf GWAS_server.tar.gz
```

**AutoDL直接用网盘/JupyterLab上传也可以。**

## 3. 服务器环境安装

```bash
# 创建虚拟环境
conda create -n planthgnn python=3.10 -y
conda activate planthgnn

# 安装PyTorch（根据服务器CUDA版本调整）
# CUDA 12.1：
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8：
# pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 安装其余依赖
pip install numpy pandas scikit-learn scipy tqdm

# 验证
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. 运行实验

```bash
cd /path/to/GWAS

# ── 快速验证（3折×1种子，~20分钟）────────────────────────────────────────────
python scripts/server_benchmark.py --fast

# ── 完整实验（5折×3种子，~3-5小时）──────────────────────────────────────────
python scripts/server_benchmark.py

# ── 只跑 NetGP vs GeneSeqGNN（重点对比）──────────────────────────────────────
python scripts/server_benchmark.py --models NetGP GeneSeqGNN

# ── 指定单个性状调试 ──────────────────────────────────────────────────────────
python scripts/server_benchmark.py --fast --traits Plant_Height

# ── 从断点续跑 ────────────────────────────────────────────────────────────────
python scripts/server_benchmark.py --resume

# ── 调整模型配置（可选）──────────────────────────────────────────────────────
python scripts/server_benchmark.py --d_model 128 --n_layers 6 --n_blocks 6
```

## 5. 结果位置

```
results/gstp007/
├── benchmark_server_results.json    # 完整结果
├── benchmark_server_table.csv       # 论文格式表格
├── benchmark_server_checkpoint.json # 断点续跑用
└── benchmark_server.log             # 训练日志
```

## 6. 模型说明

| 模型 | 说明 | 参数量 |
|------|------|--------|
| GBLUP | 统计基线（RKHS回归） | - |
| DNNGP | 深度MLP（512→256→128→64） | ~1.5M |
| NetGP | GCN基线（当前最优） | ~0.7M |
| GeneSeqGNN | **新架构**：GCN→Transformer on 831 gene tokens | ~1.2M |

## 7. 预期耗时（A100 40G）

| 模式 | 预计时间 |
|------|----------|
| `--fast`（3折×1种子×7性状×4模型） | 20-40分钟 |
| 完整（5折×3种子×7性状×4模型） | 3-6小时 |
| 只跑NetGP+GeneSeqGNN | 1.5-3小时 |
