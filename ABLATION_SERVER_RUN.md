# PlantMEGNN 消融实验 - 服务器运行指南

## 快速开始

### 1. 上传部署包到服务器

```bash
# 在本地执行
scp PlantMEGNN_server_20260331.tar.gz user@your-server:/workspace/

# 在服务器执行
ssh user@your-server
cd /workspace
tar -xzf PlantMEGNN_server_20260331.tar.gz
cd PlantMEGNN_server
```

### 2. 配置环境

```bash
# 激活conda环境
conda activate planthgnn  # 或创建新环境

# 验证GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 3. 运行消融实验

#### 方式A：交互式运行（前台）

```bash
python scripts/ablation_diverse_views_server.py \
    --output ablation_server_results.json \
    --batch_size 16 \
    --d_model 64 \
    --n_epochs 50
```

#### 方式B：后台运行（推荐）

```bash
nohup python scripts/ablation_diverse_views_server.py \
    --output ablation_server_results.json \
    --batch_size 16 \
    --d_model 64 \
    --n_epochs 50 \
    > ablation_run.log 2>&1 &

# 查看实时日志
tail -f ablation_run.log
```

#### 方式C：高显存配置（A100 40GB/80GB）

```bash
nohup python scripts/ablation_diverse_views_server.py \
    --output ablation_server_full.json \
    --batch_size 32 \
    --d_model 128 \
    --n_epochs 100 \
    > ablation_full.log 2>&1 &
```

### 4. 监控运行状态

```bash
# 查看GPU使用
watch -n 1 nvidia-smi

# 查看日志进度
tail -f ablation_run.log

# 查看结果文件是否生成
ls -lh results/gstp007/ablation_server_results.json
```

### 5. 预期运行时间

| 配置 | 预计时间 | 显存需求 |
|------|----------|----------|
| PPI-only | ~15分钟 | ~4GB |
| PPI+GO | ~25分钟 | ~6GB |
| PPI+KEGG | ~35分钟 | ~12GB |
| All views | ~45分钟 | ~16GB |
| **总计** | **~2小时** | - |

### 6. 下载结果

实验完成后，在本地执行：

```bash
scp user@your-server:/workspace/PlantMEGNN_server/results/gstp007/ablation_server_results.json \
    ./results/gstp007/
```

---

## 实验配置说明

### 消融实验设计

本次消融实验验证 **PPI-only → PPI+GO → PPI+KEGG → All views** 的递进关系：

| 配置 | 基因覆盖 | 预期PCC | 验证目标 |
|------|----------|---------|----------|
| PPI-only | 831 | ~0.85 | 基准 |
| PPI+GO | 1,712 | ~0.85 | GO视图增益 |
| PPI+KEGG | 4,053 | ~0.84 | KEGG视图增益 |
| All views | 4,053 | ~0.85 | 多视图融合 |

### 关键预期

1. **PPI+GO > PPI-only**：验证GO功能网络的补充价值
2. **All views ≈ PPI+GO**：KEGG网络可能引入噪声
3. **注意力权重分化**：
   - PPI-only: [1.00, -, -]
   - PPI+GO: [0.50, 0.50, -]
   - All views: [0.36, 0.43, 0.21]

---

## 故障排除

### CUDA Out of Memory

减小batch size：
```bash
python scripts/ablation_diverse_views_server.py --batch_size 8 --d_model 64
```

### 运行中断/续跑

脚本不支持断点续跑，但已完成的配置会保存到JSON。重新运行会覆盖之前的结果。

### 数据文件缺失

确保以下文件存在：
```
data/processed/gstp007/graph_diverse_views/
  ├── ppi_adj.npz
  ├── go_adj.npz
  ├── kegg_adj.npz
  └── gene_mapping.json

data/processed/gstp007/Grain_Length/
  ├── X_train.npy
  ├── X_val.npy
  ├── y_train.npy
  └── y_val.npy
```

---

## 结果解读

实验完成后，结果文件格式：

```json
{
  "PPI-only": {
    "pcc_mean": 0.8486,
    "pcc_std": 0.0077,
    "views": ["ppi"],
    "attn_mean": [1.0]
  },
  "PPI+GO": {
    "pcc_mean": 0.8515,
    "pcc_std": 0.0082,
    "views": ["ppi", "go"],
    "attn_mean": [0.4998, 0.5002],
    "gain_vs_ppi_only": 0.0029
  },
  ...
}
```

---

## 联系与支持

如有问题，请检查：
1. GPU驱动和CUDA版本
2. PyTorch和PyG安装
3. 数据文件完整性
