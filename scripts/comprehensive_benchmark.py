#!/usr/bin/env python
"""
综合基准测试
在多个数据集上对比不同算法的性能
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
from sklearn.ensemble import GradientBoostingRegressor
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
    """计算完整指标"""
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
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, n_traits)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        if self.padding_size > 0:
            x = F.pad(x, (0, self.padding_size))
        x = x.view(batch_size, 1, self.grid_size, self.grid_size)
        features = self.conv_layers(x)
        features = features.view(batch_size, -1)
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

def train_lightgbm(genotype, phenotype, splits):
    """训练LightGBM模型"""
    from lightgbm import LGBMRegressor
    
    train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
    
    pv = np.zeros((len(val_idx), phenotype.shape[1]))
    pt = np.zeros((len(test_idx), phenotype.shape[1]))
    
    for t in range(phenotype.shape[1]):
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=SEED,
            verbose=-1
        )
        
        y_train = phenotype[train_idx, t]
        # 处理缺失值
        valid_train = ~np.isnan(y_train)
        if valid_train.sum() < 10:
            continue
            
        model.fit(genotype[train_idx][valid_train], y_train[valid_train])
        
        pv[:, t] = model.predict(genotype[val_idx])
        pt[:, t] = model.predict(genotype[test_idx])
    
    return pv, pt

def train_deep_model(model, train_data, phenotype, splits, device, 
                     model_name="Model", n_epochs=50, lr=0.001):
    """训练深度学习模型"""
    model = model.to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
    best_val, best_state, wait = -999, None, 0
    
    for epoch in range(1, n_epochs+1):
        model.train()
        idx = train_idx.copy()
        np.random.shuffle(idx)
        
        total_loss = 0
        n_batches = 0
        batch_size = 32
        
        for i in range(0, len(idx), batch_size):
            bi = idx[i:i+batch_size]
            x, y = train_data[bi].to(device), torch.from_numpy(phenotype[bi]).float().to(device)
            
            opt.zero_grad()
            loss = nn.MSELoss()(model(x), y)
            if torch.isnan(loss): continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if n_batches == 0: continue
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(train_data[val_idx].to(device))
        val_metrics = full_metrics(val_pred.cpu().numpy(), phenotype[val_idx])
        val_pcc = val_metrics['pcc']
        
        if val_pcc > best_val:
            best_val = val_pcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 15: break
        
        if epoch % 10 == 0:
            print(f"    [{model_name}] ep {epoch:3d}  loss={total_loss/n_batches:.4f}  val_PCC={val_pcc:.4f}")
    
    if best_state: 
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        val_pred = model(train_data[val_idx].to(device))
        test_pred = model(train_data[test_idx].to(device))
    
    return val_pred.cpu().numpy(), test_pred.cpu().numpy()

def train_stacking(genotype, phenotype, splits, device):
    """训练Stacking集成模型"""
    print("  训练Stacking集成...")
    
    # 训练基础模型
    base_predictions = {}
    
    # 1. GBLUP
    gblup_pv, gblup_pt = train_gblup(genotype, phenotype, splits)
    base_predictions['GBLUP'] = {'val': gblup_pv, 'test': gblup_pt}
    
    # 2. LightGBM
    try:
        lgb_pv, lgb_pt = train_lightgbm(genotype, phenotype, splits)
        base_predictions['LightGBM'] = {'val': lgb_pv, 'test': lgb_pt}
    except:
        print("    LightGBM训练失败，跳过")
    
    # 3. MLP
    n_snps, n_traits = genotype.shape[1], phenotype.shape[1]
    mlp_model = SimpleMLP(n_snps, n_traits, hidden_dim=128)
    X_t = torch.from_numpy(genotype).float().to(device)
    mlp_pv, mlp_pt = train_deep_model(mlp_model, X_t, phenotype, splits, device, "MLP")
    base_predictions['MLP'] = {'val': mlp_pv, 'test': mlp_pt}
    
    # 4. CNN (如果SNP数不太大)
    if n_snps <= 10000:
        cnn_model = SimpleCNN(n_snps, n_traits)
        try:
            cnn_pv, cnn_pt = train_deep_model(cnn_model, X_t, phenotype, splits, device, "CNN")
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
    meta_y = phenotype[splits['val']]
    
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
# Main Evaluation
# ============================================================

def evaluate_all_methods(dataset_name, genotype, phenotype, device):
    """在单个数据集上评估所有方法"""
    print(f"\n{'='*80}")
    print(f"评估数据集: {dataset_name}")
    print(f"{'='*80}")
    
    n_samples, n_snps = genotype.shape
    n_traits = phenotype.shape[1]
    
    print(f"数据规模: {n_samples} × {n_snps} × {n_traits}")
    
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
    
    # 2. LightGBM
    print(f"\n2. LightGBM评估...")
    t0 = time.time()
    try:
        lgb_pv, lgb_pt = train_lightgbm(genotype, phenotype, splits)
        lgb_time = time.time() - t0
        lgb_metrics = full_metrics(lgb_pt, Y_test)
        print(f"LightGBM: PCC={lgb_metrics['pcc']:.4f}, MSE={lgb_metrics['mse']:.4f}, MAE={lgb_metrics['mae']:.4f} ({lgb_time:.0f}s)")
        results['LightGBM'] = {**lgb_metrics, 'time': lgb_time}
    except Exception as e:
        print(f"LightGBM失败: {e}")
    
    # 3. MLP
    print(f"\n3. MLP评估...")
    t0 = time.time()
    mlp_model = SimpleMLP(n_snps, n_traits, hidden_dim=min(256, n_snps//2))
    X_t = torch.from_numpy(genotype).float().to(device)
    mlp_pv, mlp_pt = train_deep_model(mlp_model, X_t, phenotype, splits, device, "MLP")
    mlp_time = time.time() - t0
    mlp_metrics = full_metrics(mlp_pt, Y_test)
    print(f"MLP: PCC={mlp_metrics['pcc']:.4f}, MSE={mlp_metrics['mse']:.4f}, MAE={mlp_metrics['mae']:.4f} ({mlp_time:.0f}s)")
    results['MLP'] = {**mlp_metrics, 'time': mlp_time}
    
    # 4. CNN
    if n_snps <= 10000:
        print(f"\n4. CNN评估...")
        t0 = time.time()
        cnn_model = SimpleCNN(n_snps, n_traits)
        try:
            cnn_pv, cnn_pt = train_deep_model(cnn_model, X_t, phenotype, splits, device, "CNN")
            cnn_time = time.time() - t0
            cnn_metrics = full_metrics(cnn_pt, Y_test)
            print(f"CNN: PCC={cnn_metrics['pcc']:.4f}, MSE={cnn_metrics['mse']:.4f}, MAE={cnn_metrics['mae']:.4f} ({cnn_time:.0f}s)")
            results['CNN'] = {**cnn_metrics, 'time': cnn_time}
        except Exception as e:
            print(f"CNN失败: {e}")
    else:
        print(f"\n4. CNN跳过 (SNP数过多: {n_snps})")
    
    # 5. Stacking
    print(f"\n5. Stacking评估...")
    t0 = time.time()
    stacking_pred = train_stacking(genotype, phenotype, splits, device)
    stacking_time = time.time() - t0
    stacking_metrics = full_metrics(stacking_pred, Y_test)
    print(f"Stacking: PCC={stacking_metrics['pcc']:.4f}, MSE={stacking_metrics['mse']:.4f}, MAE={stacking_metrics['mae']:.4f} ({stacking_time:.0f}s)")
    results['Stacking'] = {**stacking_metrics, 'time': stacking_time}
    
    return results

def main():
    print("=" * 80)
    print("综合基准测试 - 多算法对比")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 要测试的数据集
    datasets_to_test = ["GSTP007", "Rice469", "Maize282", "Wheat599"]
    
    all_results = {}
    
    for dataset_name in datasets_to_test:
        # 加载数据
        if dataset_name == "GSTP007":
            data_dir = Path("data/processed/GSTP007_full_10000snps_processed")
            genotype = np.load(data_dir / "genotype_50k_additive.npy")
            phenotype = np.load(data_dir / "phenotype_scaled.npy")
        else:
            data_dir = Path(f"data/external/{dataset_name.lower()}")
            genotype = np.load(data_dir / "genotype.npy")
            phenotype = np.load(data_dir / "phenotype.npy")
        
        try:
            results = evaluate_all_methods(dataset_name, genotype, phenotype, device)
            all_results[dataset_name] = results
        except Exception as e:
            print(f"评估 {dataset_name} 失败: {e}")
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("综合基准测试结果汇总")
    print(f"{'='*80}")
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        for model_name, metrics in results.items():
            pcc = metrics['pcc']
            print(f"  {model_name:<12}: PCC={pcc:.4f}, MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")
    
    # 保存结果
    output_dir = Path("data/processed/GSTP007_full_10000snps_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "comprehensive_benchmark_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果保存: {output_dir / 'comprehensive_benchmark_results.json'}")

if __name__ == "__main__":
    main()
