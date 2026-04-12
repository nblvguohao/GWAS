# PlantMEGNN — A100 服务器运行指南

## 包内容

| 路径 | 说明 |
|---|---|
| `scripts/run_multiview_netgp.py` | 主实验脚本（多视图 GCN） |
| `scripts/server_run_a100.sh` | **一键运行脚本（A100 优化参数）** |
| `scripts/finalize_gstp007_results.py` | 结果汇总分析 |
| `scripts/analyze_multiview_results.py` | 结果对比输出 |
| `data/processed/gstp007/` | 7 个性状的 SNP 特征 + 生物网络图 |
| `results/gstp007/benchmark_5fold_cv.json` | 本地基准结果（GBLUP/DNNGP/NetGP） |
| `results/gstp007/final_5fold3seed_results.json` | 本地 150epoch 结果（参考） |

---

## 快速开始

### 1. 上传并解压

```bash
scp PlantMEGNN_server.tar.gz user@server:/your/workspace/
ssh user@server
cd /your/workspace/
tar -xzf PlantMEGNN_server.tar.gz
cd PlantMEGNN_server    # 解压后的根目录
```

### 2. 配置环境

```bash
# 方式 A：使用已有 PyTorch 环境（推荐）
conda activate your_env   # 需要 torch>=2.0, torch_geometric>=2.4

# 方式 B：新建环境
conda create -n planthgnn python=3.10 -y
conda activate planthgnn
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install numpy pandas scipy scikit-learn tqdm
```

### 3. 验证环境（快速，~3分钟）

```bash
bash scripts/server_run_a100.sh --fast
```

输出示例（Plant_Height 3fold×1seed）：
```
NetGP_1view     PCC=0.8xxx
MultiView_PPI_GO PCC=0.8xxx
```

### 4. 运行完整实验（~30-50分钟）

```bash
bash scripts/server_run_a100.sh
```

实验内容：
- **4 个模型变体**：NetGP_1view / MultiView_GO / MultiView_PPI_GO / MultiView_3view
- **7 个性状** × **5-fold** × **3 seeds** = 420 runs（每个模型 105 runs）
- **参数**：epochs=200, patience=30, batch=128, d_hidden=128, lr=5e-4

---

## 参数说明（A100 vs 本地 4060）

| 参数 | 本地 4060 (已完成) | **A100（本次）** | 说明 |
|---|---|---|---|
| `--epochs` | 150 | **200** | 关键变化，更充分收敛 |
| `--patience` | 20 | **30** | 给模型更多早停余地 |
| `--batch` | 64 | **128** | A100 显存大，可翻倍 |
| `--models` | 2个 | **4个** | 补充 GO-only 和 3-view 消融 |
| checkpoint | `multiview_checkpoint.json` | `multiview_checkpoint_200e.json` | **不覆盖**本地结果 |

> 为什么用 200 epoch？
> 本地 150epoch 版本：NetGP_1view avg=0.752，benchmark NetGP=0.757（差 0.005）。
> 200epoch 版本预计让双方都提升 ~0.005，保持相对优势的同时与 benchmark 可比。

---

## 结果文件

实验完成后：

| 文件 | 内容 |
|---|---|
| `results/gstp007/multiview_results_200e.json` | 最终结果（所有模型）|
| `results/gstp007/multiview_checkpoint_200e.json` | 断点续跑 checkpoint |
| `results/gstp007/server_run_200e.log` | 完整运行日志 |
| `results/gstp007/server_summary.log` | 汇总对比表格 |

### 查看结果

```bash
# 实验结束后自动运行，也可手动执行：
python scripts/finalize_gstp007_results.py
```

---

## 拷回本地

```bash
# 只需要结果 JSON（~几百KB），不用拷全部数据
scp user@server:/workspace/PlantMEGNN_server/results/gstp007/multiview_results_200e.json \
    ./results/gstp007/
scp user@server:/workspace/PlantMEGNN_server/results/gstp007/server_summary.log \
    ./results/gstp007/
```

本地然后运行：
```bash
python scripts/analyze_multiview_results.py
```

---

## 断点续跑

脚本自动检查 checkpoint，中断后直接重新运行即可：

```bash
bash scripts/server_run_a100.sh   # 自动跳过已完成的 (fold, seed) 组合
```

---

## 常见问题

**Q: `torch_geometric` 安装失败？**
A: 先确认 CUDA 版本：`nvcc --version`，然后到 https://data.pyg.org/whl/ 找对应轮子。

**Q: `CUDA out of memory`？**
A: 减小 batch size：在 `server_run_a100.sh` 中将 `BATCH=128` 改为 `BATCH=64`。

**Q: 速度比预期慢？**
A: 检查 `torch.cuda.is_available()` 是否为 True。若为 False，脚本会用 CPU（速度慢 10-20x）。

**Q: 想只跑部分性状？**
A: 直接调用 Python 脚本：
```bash
python scripts/run_multiview_netgp.py \
    --epochs 200 --patience 30 --batch 128 \
    --traits Plant_Height Grain_Width \
    --ckpt results/gstp007/multiview_checkpoint_200e.json \
    --out  results/gstp007/multiview_results_200e.json
```
