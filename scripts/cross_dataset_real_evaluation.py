#!/usr/bin/env python
"""
跨数据集真实验证
在Rice469, Maize282, Wheat599等数据集上验证我们的方法
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
import time
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import warnings
warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ============================================================
# Metrics
# ============================================================

def full_metrics(preds, targets):
    """计算完整指标，与其他论文保持一致"""
    pccs, sccs, mses, maes = [], [], [], []
    for t in range(targets.shape[1]):
        m = ~np.isnan(targets[:, t])
        if m.sum() > 10 and np.std(preds[m, t]) > 1e-8:
            p, _ = pearsonr(targets[m, t], preds[m, t])
            s, _ = spearmanr(targets[m, t], preds[m, t])
        else:
            p, s = 0.0, 0.0
        mse = float(np.mean((preds[m, t] - targets[m, t])**2)) if m.sum() > 0 else 0
        mae = float(np.mean(np.abs(preds[m, t] - targets[m, t]))) if m.sum() > 0 else 0
        pccs.append(float(p)); sccs.append(float(s)); mses.append(mse); maes.append(mae)
    return {'pcc': float(np.mean(pccs)), 'spearman': float(np.mean(sccs)),
            'mse': float(np.mean(mses)), 'mae': float(np.mean(maes))}

def nan_huber(pred, target, delta=1.0):
    mask = ~torch.isnan(target)
    if mask.sum() == 0: return torch.tensor(0.0, device=pred.device, requires_grad=True)
    diff = (pred - target).abs()
    loss = torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    return (loss * mask).sum() / mask.sum()

# ============================================================
# Models
# ============================================================

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, n_traits, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_traits),
        )
    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, input_dim, n_traits, dropout=0.3):
        super().__init__()
        # 将SNPs重塑为2D
        self.grid_size = int(np.sqrt(input_dim))
        if self.grid_size * self.grid_size != input_dim:
            self.grid_size = int(np.ceil(np.sqrt(input_dim)))
            self.padding_size = self.grid_size * self.grid_size - input_dim
        else:
            self.padding_size = 0
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_traits)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 填充到完全平方数
        if self.padding_size > 0:
            x = F.pad(x, (0, self.padding_size))
        
        # 重塑为2D
        x = x.view(batch_size, 1, self.grid_size, self.grid_size)
        
        # 卷积特征提取
        features = self.conv_layers(x)
        features = features.view(batch_size, -1)
        
        # 回归
        return self.regressor(features)

# ============================================================
# Training Functions
# ============================================================

def train_gblup(genotype, phenotype, splits, lambdas=[0.01, 0.1, 1.0, 10.0, 100.0]):
    """训练GBLUP模型"""
    train_idx, val_idx, test_idx = [np.array(splits[k]) for k in ('train','val','test')]
    X = genotype.astype(np.float64)
    X_c = X - X.mean(axis=0)
    K = X_c @ X_c.T / X_c.shape[1]
    K_tr = K[np.ix_(train_idx, train_idx)]
    K_va = K[np.ix_(val_idx, train_idx)]
    K_te = K[np.ix_(test_idx, train_idx)]
    Y_tr = phenotype[train_idx]
    n_tr, n_traits = len(train_idx), phenotype.shape[1]
    
    pv, pt = np.zeros((len(val_idx), n_traits)), np.zeros((len(test_idx), n_traits))
    for t in range(n_traits):
        y = Y_tr[:, t].copy()
        y[np.isnan(y)] = 0.0
        best_pcc = -999
        
        for lam in lambdas:
            try:
                a = np.linalg.solve(K_tr + lam * np.eye(n_tr), y)
                pred_v = K_va @ a
                mv = ~np.isnan(phenotype[val_idx, t])
                if mv.sum() > 5 and np.std(pred_v[mv]) > 1e-8:
                    pcc = pearsonr(phenotype[val_idx[mv], t], pred_v[mv])[0]
                    if pcc > best_pcc:
                        best_pcc = pcc
                        pv[:, t] = pred_v
                        pt[:, t] = K_te @ a
            except:
                continue
    
    return pv, pt

def train_model(model, train_data, phenotype, splits, device, 
                n_epochs=50, lr=0.001, wd=1e-4, bs=32, name="Model",
                patience=15, loss_fn='huber'):
    """训练深度学习模型"""
    model = model.to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
    train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
    best_val, best_state, wait = -999, None, 0
    loss_func = nan_huber if loss_fn == 'huber' else nn.MSELoss()

    for epoch in range(1, n_epochs+1):
        model.train()
        idx = train_idx.copy()
        np.random.shuffle(idx)
        tot, nb = 0, 0
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            x, y = train_data[bi].to(device), torch.from_numpy(phenotype[bi]).float().to(device)
            opt.zero_grad()
            loss = loss_func(model(x), y)
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item(); nb += 1
        
        if nb == 0: continue

        model.eval()
        with torch.no_grad():
            vp = model(train_data[val_idx].to(device))
        vm = full_metrics(vp.cpu().numpy(), phenotype[val_idx])
        
        if vm['pcc'] > best_val:
            best_val = vm['pcc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
        
        if epoch % 10 == 0:
            print(f"    [{name}] ep {epoch:3d}  loss={tot/nb:.4f}  val_PCC={vm['pcc']:.4f}")

    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = model(train_data[val_idx].to(device))
        pt = model(train_data[test_idx].to(device))
    return pv.cpu().numpy(), pt.cpu().numpy()

def train_stacking(genotype, phenotype, splits, device):
    """训练Stacking集成模型"""
    print("  训练Stacking集成...")
    
    train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
    
    # 训练基础模型
    base_predictions = {}
    
    # 1. GBLUP
    gblup_pv, gblup_pt = train_gblup(genotype, phenotype, splits)
    base_predictions['GBLUP'] = {'val': gblup_pv, 'test': gblup_pt}
    
    # 2. MLP
    n_snps, n_traits = genotype.shape[1], phenotype.shape[1]
    mlp_model = SimpleMLP(n_snps, n_traits, hidden_dim=128, dropout=0.3)
    X_t = torch.from_numpy(genotype).float().to(device)
    mlp_pv, mlp_pt = train_model(mlp_model, X_t, phenotype, splits, device, 
                               name="MLP", n_epochs=30, bs=16)
    base_predictions['MLP'] = {'val': mlp_pv, 'test': mlp_pt}
    
    # 3. CNN (如果SNP数不太大)
    if n_snps <= 10000:
        cnn_model = SimpleCNN(n_snps, n_traits, dropout=0.3)
        try:
            cnn_pv, cnn_pt = train_model(cnn_model, X_t, phenotype, splits, device,
                                       name="CNN", n_epochs=30, bs=16)
            base_predictions['CNN'] = {'val': cnn_pv, 'test': cnn_pt}
        except:
            print("    CNN训练失败，跳过")
    
    # 训练元学习器
    print("  训练元学习器...")
    
    # 准备元训练数据
    meta_X = []
    for model_name, preds in base_predictions.items():
        meta_X.append(preds['val'])
    
    meta_X = np.hstack(meta_X)
    meta_y = phenotype[val_idx]
    
    # 训练Ridge回归作为元学习器
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(meta_X, meta_y)
    
    # 生成最终预测
    test_meta_X = []
    for model_name, preds in base_predictions.items():
        test_meta_X.append(preds['test'])
    
    test_meta_X = np.hstack(test_meta_X)
    stacking_pred = meta_learner.predict(test_meta_X)
    
    return stacking_pred

# ============================================================
# Data Loading
# ============================================================

def load_dataset(dataset_name):
    """加载数据集"""
    data_dir = Path(f"data/external/{dataset_name.lower()}")
    
    if not data_dir.exists():
        print(f"❌ 数据集目录不存在: {data_dir}")
        return None, None, None
    
    try:
        genotype = np.load(data_dir / "genotype.npy")
        phenotype = np.load(data_dir / "phenotype.npy")
        
        # 加载元信息
        metadata = {}
        if (data_dir / "metadata.json").exists():
            with open(data_dir / "metadata.json") as f:
                metadata = json.load(f)
        
        print(f"✅ 成功加载 {dataset_name}:")
        print(f"   基因型: {genotype.shape}")
        print(f"   表型: {phenotype.shape}")
        if metadata:
            print(f"   元信息: {metadata}")
        
        return genotype, phenotype, metadata
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None, None, None

# ============================================================
# Evaluation
# ============================================================

def evaluate_single_dataset(dataset_name, genotype, phenotype, metadata, device):
    """评估单个数据集"""
    print(f"\n{'='*80}")
    print(f"评估数据集: {dataset_name}")
    print(f"{'='*80}")
    
    n_samples, n_snps = genotype.shape
    n_traits = phenotype.shape[1]
    
    print(f"数据规模: {n_samples} × {n_snps} × {n_traits}")
    
    # 数据质量检查
    print(f"数据质量:")
    print(f"  基因型范围: [{genotype.min():.1f}, {genotype.max():.1f}]")
    print(f"  表型范围: [{phenotype.min():.2f}, {phenotype.max():.2f}]")
    print(f"  缺失值: 基因型={np.isnan(genotype).sum()}, 表型={np.isnan(phenotype).sum()}")
    
    # 创建数据划分
    indices = np.random.permutation(n_samples)
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)
    
    splits = {
        'train': indices[:train_end],
        'val': indices[train_end:val_end],
        'test': indices[val_end:]
    }
    
    val_idx, test_idx = splits['val'], splits['test']
    Y_val, Y_test = phenotype[val_idx], phenotype[test_idx]
    
    results = {}
    
    # 1. GBLUP
    print(f"\n1. GBLUP评估...")
    t0 = time.time()
    gblup_pv, gblup_pt = train_gblup(genotype, phenotype, splits)
    gblup_time = time.time() - t0
    
    gblup_metrics = full_metrics(gblup_pt, Y_test)
    print(f"GBLUP: PCC={gblup_metrics['pcc']:.4f}, MSE={gblup_metrics['mse']:.4f}, MAE={gblup_metrics['mae']:.4f} ({gblup_time:.0f}s)")
    results['GBLUP'] = {**gblup_metrics, 'time': gblup_time}
    
    # 2. MLP
    print(f"\n2. MLP评估...")
    t0 = time.time()
    n_traits = phenotype.shape[1]
    mlp_model = SimpleMLP(n_snps, n_traits, hidden_dim=min(256, n_snps//2))
    X_t = torch.from_numpy(genotype).float().to(device)
    mlp_pv, mlp_pt = train_model(mlp_model, X_t, phenotype, splits, device, 
                               name="MLP", n_epochs=40, bs=16)
    mlp_time = time.time() - t0
    
    mlp_metrics = full_metrics(mlp_pt, Y_test)
    print(f"MLP: PCC={mlp_metrics['pcc']:.4f}, MSE={mlp_metrics['mse']:.4f}, MAE={mlp_metrics['mae']:.4f} ({mlp_time:.0f}s)")
    results['MLP'] = {**mlp_metrics, 'time': mlp_time}
    
    # 3. CNN (如果SNP数不太大)
    if n_snps <= 10000:
        print(f"\n3. CNN评估...")
        t0 = time.time()
        cnn_model = SimpleCNN(n_snps, n_traits, dropout=0.3)
        try:
            cnn_pv, cnn_pt = train_model(cnn_model, X_t, phenotype, splits, device,
                                       name="CNN", n_epochs=30, bs=16)
            cnn_time = time.time() - t0
            
            cnn_metrics = full_metrics(cnn_pt, Y_test)
            print(f"CNN: PCC={cnn_metrics['pcc']:.4f}, MSE={cnn_metrics['mse']:.4f}, MAE={cnn_metrics['mae']:.4f} ({cnn_time:.0f}s)")
            results['CNN'] = {**cnn_metrics, 'time': cnn_time}
        except Exception as e:
            print(f"CNN训练失败: {e}")
    else:
        print(f"\n3. CNN跳过 (SNP数过多: {n_snps})")
    
    # 4. Stacking
    print(f"\n4. Stacking评估...")
    t0 = time.time()
    stacking_pred = train_stacking(genotype, phenotype, splits, device)
    stacking_time = time.time() - t0
    
    stacking_metrics = full_metrics(stacking_pred, Y_test)
    print(f"Stacking: PCC={stacking_metrics['pcc']:.4f}, MSE={stacking_metrics['mse']:.4f}, MAE={stacking_metrics['mae']:.4f} ({stacking_time:.0f}s)")
    results['Stacking'] = {**stacking_metrics, 'time': stacking_time}
    
    return results

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("跨数据集真实验证")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 要测试的数据集
    datasets_to_test = ["Rice469", "Maize282", "Wheat599"]
    
    all_results = {}
    
    for dataset_name in datasets_to_test:
        genotype, phenotype, metadata = load_dataset(dataset_name)
        
        if genotype is None:
            print(f"跳过 {dataset_name} (数据不可用)")
            continue
        
        try:
            results = evaluate_single_dataset(dataset_name, genotype, phenotype, metadata, device)
            all_results[dataset_name] = results
        except Exception as e:
            print(f"评估 {dataset_name} 时出错: {e}")
            continue
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("跨数据集验证结果汇总")
    print(f"{'='*80}")
    
    # 与我们的基线对比
    baseline_pcc = 0.6343  # GSTP007上的最佳结果
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        for model_name, metrics in results.items():
            pcc = metrics['pcc']
            improvement = pcc - baseline_pcc
            print(f"  {model_name:<10}: PCC={pcc:.4f}  vs基线={improvement:+.4f} ({improvement/baseline_pcc*100:+.1f}%)")
    
    # 保存结果
    output_dir = Path("data/processed/GSTP007_full_10000snps_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "cross_dataset_real_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果保存: {output_dir / 'cross_dataset_real_results.json'}")
    
    # 分析
    print(f"\n{'='*80}")
    print("泛化能力分析")
    print(f"{'='*80}")
    
    if len(all_results) > 0:
        # 计算平均性能
        model_names = set()
        for results in all_results.values():
            model_names.update(results.keys())
        
        model_performance = {}
        for model in model_names:
            pccs = []
            for results in all_results.values():
                if model in results:
                    pccs.append(results[model]['pcc'])
            
            if pccs:
                model_performance[model] = {
                    'mean': np.mean(pccs),
                    'std': np.std(pccs),
                    'count': len(pccs)
                }
        
        print(f"各模型平均性能:")
        for model, perf in model_performance.items():
            print(f"  {model:<10}: PCC={perf['mean']:.4f} ± {perf['std']:.4f} ({perf['count']}个数据集)")
        
        # 与基线对比
        print(f"\n基线 (GSTP007): {baseline_pcc:.4f}")
        
        if 'Stacking' in model_performance:
            stacking_avg = model_performance['Stacking']['mean']
            stacking_diff = stacking_avg - baseline_pcc
            print(f"\nStacking泛化:")
            print(f"平均PCC: {stacking_avg:.4f}")
            print(f"vs基线: {stacking_diff:+.4f} ({stacking_diff/baseline_pcc*100:+.1f}%)")
        
        # 结论
        best_model = max(model_performance.keys(), key=lambda k: model_performance[k]['mean'])
        best_perf = model_performance[best_model]['mean']
        
        print(f"\n最佳泛化模型: {best_model} (PCC={best_perf:.4f})")
        
        if best_perf >= 0.5:
            print("✅ 方法具有良好的泛化性能")
        elif best_perf >= 0.3:
            print("🤔 方法泛化性能中等，存在改进空间")
        else:
            print("⚠️  方法泛化性能有限，需要数据集特定优化")

if __name__ == "__main__":
    main()
