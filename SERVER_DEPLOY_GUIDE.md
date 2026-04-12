# PlantMEGNN 服务器部署指南

## 快速开始

### 1. 上传并解压

```bash
# 在本地执行
scp PlantMEGNN_server_20260331.tar.gz user@your-server:/workspace/

# 在服务器执行
cd /workspace
tar -xzf PlantMEGNN_server_20260331.tar.gz
cd PlantMEGNN
```

### 2. 环境配置

```bash
# 创建conda环境
conda create -n plantmegnn python=3.10 -y
conda activate plantmegnn

# 安装依赖
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install numpy scipy scikit-learn pandas matplotlib seaborn
pip install pyyaml tqdm
```

### 3. 验证环境

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## 运行消融实验

### 完整消融实验 (4配置 × 5次运行)

```bash
# 使用默认参数 (推荐)
# - batch_size=16 (小batch节省显存)
# - d_model=64 (减小模型维度)
# - n_epochs=50

python scripts/ablation_diverse_views_server.py \
    --output ablation_server_results.json \
    --batch_size 16 \
    --d_model 64 \
    --n_epochs 50
```

### 后台运行 (推荐)

```bash
nohup python scripts/ablation_diverse_views_server.py \
    --output ablation_server_results.json \
    --batch_size 16 \
    --d_model 64 \
    > ablation_run.log 2>&1 &

# 查看日志
tail -f ablation_run.log
```

### 高显存配置 (A100 40GB/80GB)

```bash
python scripts/ablation_diverse_views_server.py \
    --output ablation_server_full.json \
    --batch_size 32 \
    --d_model 128 \
    --n_epochs 100
```

## 预期结果

| 配置 | 预期PCC | 注意力分布 | 显存需求 |
|------|---------|------------|----------|
| PPI-only | 0.79±0.01 | PPI: 1.00 | ~4GB |
| PPI+GO | 0.78±0.01 | PPI: 0.50, GO: 0.50 | ~6GB |
| PPI+KEGG | 0.78±0.01 | PPI: 0.52, KEGG: 0.48 | ~12GB |
| All views | 0.79±0.01 | PPI: 0.36, GO: 0.43, KEGG: 0.21 | ~16GB |

## 监控与故障排除

### 监控GPU使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或
nvidia-smi dmon -s u
```

### 常见错误

**1. CUDA Out of Memory**
```
解决方案: 减小batch_size或d_model
python scripts/ablation_diverse_views_server.py --batch_size 8 --d_model 32
```

**2. 数据文件缺失**
```
确保 data/processed/gstp007/graph_diverse_views/ 目录存在:
- ppi_adj.npz (831 genes)
- go_adj.npz (1,712 genes)
- kegg_adj.npz (4,053 genes)
```

### 结果同步回本地

```bash
# 从服务器下载结果
scp user@server:/workspace/PlantMEGNN/results/gstp007/ablation_server_results.json \
    E:/GWAS/results/gstp007/
```

## 文件清单

已打包的文件:
```
PlantMEGNN/
├── src/                          # 源代码
│   ├── models/                   # 模型实现
│   │   ├── multi_view_gcn_diverse.py
│   │   ├── plant_hgnn_diverse.py
│   │   └── ...
│   └── ...
├── scripts/                      # 运行脚本
│   ├── ablation_diverse_views_server.py  # 服务器消融实验
│   ├── generate_attention_visualizations.py
│   └── generate_paper_figures.py
├── data/processed/gstp007/graph_diverse_views/  # 网络数据
│   ├── ppi_adj.npz
│   ├── go_adj.npz
│   └── kegg_adj.npz
├── results/gstp007/              # 已有结果
│   └── ablation_partial_results.json
└── docs/                         # 文档
    ├── NEXT_STEPS_STATUS.md
    ├── PAPER_REVIEW_REPORT.md
    └── IMPLEMENTATION_SUMMARY.md
```

## 联系与支持

如有问题，请参考:
- `NEXT_STEPS_STATUS.md` - 项目状态和待办事项
- `PAPER_REVIEW_REPORT.md` - 审稿报告与修改建议

---
*打包日期: 2026-03-31*
*包大小: 6.6MB*
