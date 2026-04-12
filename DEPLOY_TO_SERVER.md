# 服务器部署和运行指南

**目标**: 在服务器上运行d_model=256的消融实验，匹配主实验配置

**服务器**: user@100.112.165.109  
**密码**: ppp321

---

## 一、部署步骤

### 1.1 上传GWAS项目到服务器

```bash
# 从本地Windows上传到服务器
# 方法1: 使用scp (在Windows PowerShell或Git Bash中)
scp -r E:/GWAS/scripts user@100.112.165.109:~/GWAS/
scp -r E:/GWAS/src user@100.112.165.109:~/GWAS/
scp -r E:/GWAS/data/processed/gstp007 user@100.112.165.109:~/GWAS/data/processed/

# 方法2: 如果scp不可用，使用rsync
rsync -avz --progress E:/GWAS/scripts user@100.112.165.109:~/GWAS/
rsync -avz --progress E:/GWAS/src user@100.112.165.109:~/GWAS/
```

### 1.2 服务器端设置

```bash
# SSH登录服务器
ssh user@100.112.165.109

# 创建目录结构
mkdir -p ~/GWAS/results/gstp007/ablation
mkdir -p ~/GWAS/data/processed/gstp007/graph_diverse_views

# 检查Python环境
python3 --version  # 应 >= 3.8
pip3 list | grep -E "torch|numpy|scipy"

# 安装依赖 (如需要)
pip3 install torch numpy scipy scikit-learn pandas matplotlib
```

---

## 二、运行d_model=256消融实验

### 2.1 创建优化版消融脚本

将以下脚本保存为 `~/GWAS/scripts/run_ablation_d256.py`:

```python
#!/usr/bin/env python3
"""
消融实验 - d_model=256版本
匹配主实验最优配置
"""

import sys
sys.path.insert(0, '/home/user/GWAS')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from pathlib import Path
from scipy.stats import pearsonr
import json
import logging
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PROC_DIR = Path('/home/user/GWAS/data/processed/gstp007')
GRAPH_DIR = PROC_DIR / 'graph_diverse_views'
RESULT_DIR = Path('/home/user/GWAS/results/gstp007/ablation')
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 最优配置 (HPO-2)
BEST_CONFIG = {
    'd_model': 256,
    'batch_size': 64,
    'lr': 1.4e-4,
    'dropout': 0.25,
    'weight_decay': 1e-4,
    'n_epochs': 50,
    'patience': 20
}

# [模型定义与run_ablation_study.py相同]
# ... (省略，实际使用时需要完整代码)

def main():
    trait = 'Grain_Length'
    seeds = [42, 123, 456]
    n_folds = 5
    
    configs = [
        ('PPI+KEGG-d256', ['ppi', 'kegg']),
        ('PPI+GO-d256', ['ppi', 'go']),
        ('PPI-only-d256', ['ppi']),
        ('GO-only-d256', ['go']),
        ('KEGG-only-d256', ['kegg']),
    ]
    
    all_results = {}
    
    for name, views in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {name}")
        logger.info(f"{'='*60}")
        
        # 这里调用运行函数
        # result = run_config(name, views, trait, seeds, n_folds)
        # all_results[name] = result
    
    # 保存结果
    output_file = RESULT_DIR / f'ablation_d256_{trait}_{json.dumps(seeds)}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")

if __name__ == '__main__':
    main()
```

### 2.2 启动实验

```bash
# SSH登录服务器
ssh user@100.112.165.109

# 进入项目目录
cd ~/GWAS

# 后台运行消融实验
nohup python3 scripts/run_ablation_d256.py > results/ablation_d256.log 2>&1 &

# 查看运行状态
tail -f results/ablation_d256.log

# 检查GPU使用
nvidia-smi
```

---

## 三、运行RF/XGB基线

### 3.1 上传并运行RF基线

```bash
# 在服务器上运行
ssh user@100.112.165.109
cd ~/GWAS

# Random Forest
nohup python3 scripts/run_rf_xgboost_baseline.py \
    --method rf \
    --trait Grain_Length \
    --seeds 42 123 456 \
    --n_folds 5 \
    > results/rf_baseline.log 2>&1 &

# XGBoost
nohup python3 scripts/run_rf_xgboost_baseline.py \
    --method xgboost \
    --trait Grain_Length \
    --seeds 42 123 456 \
    --n_folds 5 \
    > results/xgboost_baseline.log 2>&1 &
```

---

## 四、监控和获取结果

### 4.1 监控运行状态

```bash
# 查看所有Python进程
ps aux | grep python | grep -v grep

# 查看特定进程
ps aux | grep run_ablation

# 查看GPU使用
nvidia-smi

# 查看日志
tail -f ~/GWAS/results/ablation_d256.log
```

### 4.2 下载结果到本地

```bash
# 从服务器下载结果到本地
scp user@100.112.165.109:~/GWAS/results/gstp007/ablation/*.json E:/GWAS/results/server_d256/

# 或下载整个结果目录
scp -r user@100.112.165.109:~/GWAS/results/gstp007/ablation E:/GWAS/results/server_d256/
```

---

## 五、预期运行时间

| 实验 | 配置数 | 每配置时间 | 总时间 | GPU需求 |
|------|--------|-----------|--------|---------|
| d=256消融 | 5 | ~30分钟 | ~2.5小时 | 1x GPU |
| Random Forest | 1 | ~3小时 | ~3小时 | CPU |
| XGBoost | 1 | ~3小时 | ~3小时 | CPU |

**并行策略**: 
- GPU实验 (消融) 串行运行
- CPU实验 (RF/XGB) 可与GPU实验并行

---

## 六、验证结果

### 6.1 检查结果完整性

```python
# 验证脚本 (在本地运行)
import json

with open('results/server_d256/ablation_d256_Grain_Length.json') as f:
    data = json.load(f)

print("消融结果验证:")
for config, results in data.items():
    print(f"  {config}: {results['mean_pcc']:.4f} ± {results['std_pcc']:.4f}")
    
# 检查是否d=256比d=64好
d64_ppi_kegg = 0.8608  # 已知结果
d256_ppi_kegg = data['PPI+KEGG-d256']['mean_pcc']
print(f"\nd=64 PCC: {d64_ppi_kegg:.4f}")
print(f"d=256 PCC: {d256_ppi_kegg:.4f}")
print(f"提升: {(d256_ppi_kegg - d64_ppi_kegg)*100:.2f}%")
```

### 6.2 预期结果

| 配置 | d=64 (已有) | d=256 (预期) | 预期提升 |
|------|-------------|--------------|----------|
| PPI+KEGG | 0.8608 | 0.875-0.885 | +1-2% |
| PPI+GO | 0.8549 | 0.870-0.880 | +1-2% |
| PPI-only | 0.8541 | 0.868-0.878 | +1-2% |

---

## 七、常见问题

### Q1: 上传失败/速度慢？
**解决**: 使用压缩后上传
```bash
# 本地压缩
tar -czf gwas_scripts.tar.gz scripts/ src/
# 上传
scp gwas_scripts.tar.gz user@100.112.165.109:~/
# 服务器解压
tar -xzf gwas_scripts.tar.gz
```

### Q2: 缺少依赖包？
**解决**: 
```bash
pip3 install torch==2.1.0 torch-geometric numpy scipy scikit-learn pandas matplotlib
```

### Q3: GPU内存不足？
**解决**: 减小batch_size到32或16

### Q4: 实验中断？
**解决**: 使用checkpoint功能恢复 (需在脚本中实现)

---

**部署检查清单**:
- [ ] 上传scripts目录
- [ ] 上传src目录  
- [ ] 上传数据文件
- [ ] 安装依赖
- [ ] 测试运行
- [ ] 启动完整实验
- [ ] 设置监控
- [ ] 等待结果
