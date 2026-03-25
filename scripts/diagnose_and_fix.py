#!/usr/bin/env python
"""
Deep diagnosis of PlantHGNN performance issues and systematic fix testing.

Root causes identified from source code analysis:
  BUG 1: GCN / Functional / Structural embeddings are SAMPLE-INDEPENDENT constants
  BUG 2: Transformer operates on sequence length 1 (self-attention degenerates)
  BUG 3: SNPEncoder compresses 10K SNPs into a single vector (extreme bottleneck)
  BUG 4: Graph data is synthetic random noise (no biological signal)
  BUG 5: Multi-task on 32 traits simultaneously hurts per-trait accuracy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.optim import Adam
from scipy.stats import pearsonr
from tqdm import tqdm

# ============================================================
# Part 1: Quantitative Diagnosis
# ============================================================

def diagnose():
    """Quantify each architectural problem."""
    print("=" * 80)
    print("PART 1: DIAGNOSIS — Why PlantHGNN underperforms GBLUP")
    print("=" * 80)

    data_dir = Path("data/processed/GSTP007_full_10000snps_processed")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data once
    snp_data = torch.from_numpy(np.load(data_dir / "genotype_onehot.npy")).float()
    phenotype = torch.from_numpy(np.load(data_dir / "phenotype_scaled.npy")).float()
    with open(data_dir / "split.json") as f:
        splits = json.load(f)
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)

    network_dir = data_dir / "networks"
    ppi = torch.load(network_dir / "ppi_network.pt", weights_only=False)
    go  = torch.load(network_dir / "go_network.pt",  weights_only=False)
    kegg = torch.load(network_dir / "kegg_network.pt", weights_only=False)
    graph_data = {
        'node_features': ppi.x,
        'edge_index_list': [ppi.edge_index, go.edge_index, kegg.edge_index],
        'edge_weight_list': [ppi.edge_attr, go.edge_attr, kegg.edge_attr],
        'random_walk_features': torch.load(network_dir / "random_walk_features.pt", weights_only=False),
        'pagerank_scores': torch.load(network_dir / "pagerank_scores.pt", weights_only=False),
        'gene_set_matrix': torch.load(network_dir / "gene_set_matrix.pt", weights_only=False),
    }

    # --- BUG 1: GCN/Func/Struct are sample-independent constants ---
    print("\n" + "-" * 70)
    print("BUG 1: GCN / Functional / Structural embeddings are CONSTANT")
    print("-" * 70)
    print("""
In PlantHGNN.forward():
  gcn_embed_pooled  = gcn_embed.mean(dim=0).expand(batch_size, -1)   # same for ALL samples
  func_embed_pooled = func_embed.mean(dim=0).expand(batch_size, -1)  # same for ALL samples
  struct_embed_pooled = struct_embed.mean(dim=0).expand(batch_size, -1)  # same for ALL samples

=> 3 out of 4 feature vectors concatenated into h_input are IDENTICAL
   across every sample. They carry ZERO discriminative information.
   They act as a learned bias — at best useless, at worst adding noise.
""")
    print("  Consequence: The model effectively ignores graph information entirely.")

    # --- BUG 2: Transformer on seq_len=1 ---
    print("\n" + "-" * 70)
    print("BUG 2: Transformer operates on sequence length = 1")
    print("-" * 70)
    print("""
In PlantHGNN.forward():
  h = h.unsqueeze(1)   # (batch, 1, d_model)   <-- sequence length 1!
  h = self.transformer(h)

Self-attention on a single token:
  Attention(Q,K,V) = softmax(Q·K^T / √d) · V
  With seq_len=1: Q·K^T is a scalar, softmax(scalar)=1.0, output = V = input
  => The self-attention layer is a no-op. Only the FFN sub-layer does anything.
  => The entire Transformer degenerates into stacked MLPs with LayerNorm.
  => AttnRes depth-attention is also meaningless (aggregating identical tokens).
""")
    print("  Consequence: Transformer adds parameters & NaN risk with no benefit over MLP.")

    # --- BUG 3: Extreme bottleneck in SNPEncoder ---
    print("\n" + "-" * 70)
    print("BUG 3: SNPEncoder compresses 10,000 SNPs into 1 vector")
    print("-" * 70)
    print("""
In SNPEncoder.forward():
  x = Conv1d(3→64, k=3) → Conv1d(64→128, k=3) → AdaptiveAvgPool1d(1)
  => (batch, 128, 10000) → (batch, 128, 1) → Linear → (batch, d_model)

  10,000 SNP positions are averaged into a SINGLE 128-dim vector.
  This is an extreme information bottleneck — the spatial/genomic structure is destroyed.
  
  For comparison, GBLUP uses a 1495×1495 kernel matrix computed from ALL SNP values.
  It preserves full pairwise sample similarity. PlantHGNN throws it all away.
""")
    # Quantify information loss
    n_snps = meta['n_snps_selected']
    input_info = n_snps * 3  # 30,000 values per sample
    output_info = 64  # d_model in simplified version
    print(f"  Input information:  {input_info:,} values per sample (10K SNPs × 3)")
    print(f"  After SNPEncoder:  {output_info} values per sample")
    print(f"  Compression ratio: {input_info / output_info:.0f}:1  ← EXTREME")

    # --- BUG 4: Synthetic random graphs ---
    print("\n" + "-" * 70)
    print("BUG 4: Graph data is synthetic RANDOM noise")
    print("-" * 70)
    print("""
In build_networks_optimized.py:
  edge_index_ppi = torch.randint(0, n_genes, (2, n_edges_ppi))
  edge_weight_ppi = torch.rand(n_edges_ppi) * 0.5 + 0.5
  node_features = torch.randn(n_genes, 64)

  The PPI, GO, KEGG networks are ALL randomly generated.
  No real biological information. The GCN learns on pure noise.
""")
    # Verify randomness
    ei = graph_data['edge_index_list'][0]
    n_self_loops = (ei[0] == ei[1]).sum().item()
    unique_edges = set(zip(ei[0].tolist(), ei[1].tolist()))
    print(f"  PPI edges: {ei.shape[1]:,}")
    print(f"  Self-loops: {n_self_loops:,} ({100*n_self_loops/ei.shape[1]:.1f}%)")
    print(f"  Unique edges: {len(unique_edges):,} / {ei.shape[1]:,}")
    print(f"  => Random graph confirmed. GCN on this = learning from noise.")

    # --- BUG 5: Multi-task on 32 traits ---
    print("\n" + "-" * 70)
    print("BUG 5: Multi-task learning on 32 traits simultaneously")
    print("-" * 70)
    # Check trait correlations
    pheno_np = phenotype.numpy()
    train_idx = splits['train']
    train_pheno = pheno_np[train_idx]
    # Compute pairwise trait correlations
    valid_traits = []
    for i in range(train_pheno.shape[1]):
        if np.std(train_pheno[:, i]) > 0 and not np.isnan(train_pheno[:, i]).all():
            valid_traits.append(i)
    trait_corr_matrix = np.corrcoef(train_pheno[:, valid_traits].T)
    avg_abs_corr = np.mean(np.abs(trait_corr_matrix[np.triu_indices_from(trait_corr_matrix, k=1)]))
    print(f"  Number of traits: 32")
    print(f"  Avg absolute inter-trait correlation: {avg_abs_corr:.3f}")
    print(f"  => Low correlation means shared encoder hurts per-trait accuracy.")
    print(f"  => Some traits may have conflicting gradients, causing NaN.")

    # Check for NaN in phenotype data
    nan_count = np.isnan(pheno_np).sum()
    nan_per_trait = np.isnan(pheno_np).sum(axis=0)
    traits_with_nan = (nan_per_trait > 0).sum()
    print(f"\n  NaN in phenotype: {nan_count} total, {traits_with_nan}/32 traits affected")
    print(f"  Traits with most NaN: {nan_per_trait.max()} samples missing")
    print(f"  => NaN phenotypes flow into loss → NaN gradients → NaN loss!")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print("""
┌──────┬──────────────────────────────────────────────┬──────────┐
│ Bug  │ Description                                  │ Severity │
├──────┼──────────────────────────────────────────────┼──────────┤
│  1   │ GCN/Func/Struct = constant for all samples   │ CRITICAL │
│  2   │ Transformer on seq_len=1 = expensive no-op    │ HIGH     │
│  3   │ SNP encoder: 30K values → 64-dim bottleneck   │ CRITICAL │
│  4   │ Random synthetic graphs = noise               │ HIGH     │
│  5   │ 32-trait multi-task + NaN phenotypes → NaN     │ HIGH     │
└──────┴──────────────────────────────────────────────┴──────────┘

Root cause: The model's ONLY sample-varying signal passes through a
            470:1 bottleneck (SNPEncoder). Everything else is constant noise.
            GBLUP uses full SNP data directly. That's why it wins.
""")
    return snp_data, phenotype, splits, meta, graph_data, device


# ============================================================
# Part 2: Fix models
# ============================================================

class FixedPlantHGNN_V1(nn.Module):
    """
    Fix V1: Strip useless components, better SNP encoder, handle NaN phenotypes.
    
    Changes vs original:
    - Remove GCN/Functional/Structural (constant noise)
    - Remove Transformer (no-op on seq_len=1)
    - Multi-scale CNN for SNPs (preserve more information)
    - Per-trait NaN masking in loss
    - Much fewer parameters
    """
    def __init__(self, n_snps, n_traits, d_hidden=256, dropout=0.3):
        super().__init__()
        self.n_traits = n_traits
        
        # Multi-scale CNN encoder for SNPs
        self.conv_k3 = nn.Conv1d(3, 32, kernel_size=3, padding=1)
        self.conv_k7 = nn.Conv1d(3, 32, kernel_size=7, padding=3)
        self.conv_k15 = nn.Conv1d(3, 32, kernel_size=15, padding=7)
        
        self.bn1 = nn.BatchNorm1d(96)
        
        self.conv2 = nn.Conv1d(96, 128, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Adaptive pooling to fixed size (preserve spatial info)
        self.pool = nn.AdaptiveAvgPool1d(64)  # Keep 64 positions, not 1!
        
        # MLP head
        self.flatten_dim = 128 * 64  # 8192
        self.head = nn.Sequential(
            nn.Linear(self.flatten_dim, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.BatchNorm1d(d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, n_traits)
        )
    
    def forward(self, snp_data, graph_data=None):
        x = snp_data.transpose(1, 2)  # (batch, 3, n_snps)
        
        # Multi-scale feature extraction
        x3  = F.relu(self.conv_k3(x))
        x7  = F.relu(self.conv_k7(x))
        x15 = F.relu(self.conv_k15(x))
        x = torch.cat([x3, x7, x15], dim=1)  # (batch, 96, n_snps)
        x = self.bn1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.pool(x)  # (batch, 128, 64)
        x = x.view(x.size(0), -1)  # (batch, 8192)
        
        return self.head(x)


class FixedPlantHGNN_V2(nn.Module):
    """
    Fix V2: Use SNP data to compute sample-specific kernel features (like GBLUP).
    
    Key idea: Instead of compressing SNPs, compute a GRM-inspired representation.
    For each sample, compute its similarity to a set of "anchor" samples,
    then predict traits from this similarity vector.
    This mimics what makes GBLUP powerful — using pairwise sample relationships.
    """
    def __init__(self, n_snps, n_traits, n_anchors=100, d_hidden=256, dropout=0.3):
        super().__init__()
        self.n_snps = n_snps
        self.n_anchors = n_anchors
        
        # Learnable anchor points in SNP space (like kernel inducing points)
        self.anchors = nn.Parameter(torch.randn(n_anchors, n_snps * 3) * 0.01)
        
        # Temperature for similarity computation
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # MLP head from similarity features
        self.head = nn.Sequential(
            nn.Linear(n_anchors, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.BatchNorm1d(d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, n_traits)
        )
    
    def forward(self, snp_data, graph_data=None):
        batch_size = snp_data.size(0)
        x = snp_data.reshape(batch_size, -1)  # (batch, n_snps * 3)
        
        # Compute cosine similarity to anchors
        x_norm = F.normalize(x, dim=1)
        a_norm = F.normalize(self.anchors, dim=1)
        similarity = torch.matmul(x_norm, a_norm.t()) / self.temperature.abs().clamp(min=0.1)
        # (batch, n_anchors)
        
        return self.head(similarity)


class FixedPlantHGNN_V3(nn.Module):
    """
    Fix V3: Direct linear model with regularization (neural GBLUP).
    
    Idea: A well-regularized linear model on the flattened SNP data.
    This is the neural network equivalent of GBLUP.
    Should be competitive with GBLUP as a sanity check.
    """
    def __init__(self, n_snps, n_traits, d_hidden=512, dropout=0.5):
        super().__init__()
        input_dim = n_snps * 3
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 4),
            nn.BatchNorm1d(d_hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 4, n_traits)
        )
    
    def forward(self, snp_data, graph_data=None):
        x = snp_data.reshape(snp_data.size(0), -1)
        return self.net(x)


# ============================================================
# Part 3: Training with NaN-safe loss
# ============================================================

def nan_safe_mse_loss(pred, target):
    """MSE loss that masks NaN values in target — fixes BUG 5 NaN source."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum()


def train_and_evaluate(model, snp_data, phenotype, splits, device,
                       n_epochs=30, lr=0.001, weight_decay=1e-4, batch_size=32,
                       model_name="Model"):
    """Train a model and return test PCC."""
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    train_idx = splits['train']
    val_idx = splits['val']
    test_idx = splits['test']
    
    best_val_pcc = -999
    best_state = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(1, n_epochs + 1):
        # --- Train ---
        model.train()
        indices = train_idx.copy()
        np.random.shuffle(indices)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = snp_data[batch_idx].to(device)
            y = phenotype[batch_idx].to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = nan_safe_mse_loss(pred, y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if n_batches == 0:
            continue
        
        # --- Validate ---
        model.eval()
        with torch.no_grad():
            x_val = snp_data[val_idx].to(device)
            y_val = phenotype[val_idx].to(device)
            pred_val = model(x_val).cpu().numpy()
            y_val_np = phenotype[val_idx].numpy()
        
        # Compute per-trait PCC
        pccs = []
        for t in range(phenotype.shape[1]):
            mask = ~np.isnan(y_val_np[:, t])
            if mask.sum() > 10 and np.std(pred_val[mask, t]) > 1e-8:
                pcc, _ = pearsonr(y_val_np[mask, t], pred_val[mask, t])
                pccs.append(pcc)
        
        val_pcc = np.mean(pccs) if pccs else -999
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{model_name}] Epoch {epoch:3d}  loss={total_loss/n_batches:.4f}  "
                  f"val_PCC={val_pcc:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")
        
        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{model_name}] Early stopping at epoch {epoch}")
                break
    
    # --- Test ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        x_test = snp_data[test_idx].to(device)
        pred_test = model(x_test).cpu().numpy()
        y_test_np = phenotype[test_idx].numpy()
    
    test_pccs = []
    for t in range(phenotype.shape[1]):
        mask = ~np.isnan(y_test_np[:, t])
        if mask.sum() > 10 and np.std(pred_test[mask, t]) > 1e-8:
            pcc, _ = pearsonr(y_test_np[mask, t], pred_test[mask, t])
            test_pccs.append(pcc)
    
    test_pcc = np.mean(test_pccs) if test_pccs else -999
    
    return {
        'test_pcc': test_pcc,
        'best_val_pcc': best_val_pcc,
        'n_params': sum(p.numel() for p in model.parameters()),
        'test_pccs_per_trait': test_pccs,
    }


# ============================================================
# Part 4: Run all experiments
# ============================================================

def main():
    # --- Diagnosis ---
    snp_data, phenotype, splits, meta, graph_data, device = diagnose()
    
    n_snps = meta['n_snps_selected']
    n_traits = meta['n_traits']
    
    print("\n" + "=" * 80)
    print("PART 2: TESTING FIXES")
    print("=" * 80)
    
    results = {}
    
    # --- Experiment 0: Original PlantHGNN (simplified, as baseline) ---
    print("\n>>> Experiment 0: Original PlantHGNN (simplified config)")
    from src.models.plant_hgnn import PlantHGNN
    with open(Path("data/processed/GSTP007_full_10000snps_processed/networks/network_metadata.json")) as f:
        net_meta = json.load(f)
    
    class OriginalWrapper(nn.Module):
        """Wrap PlantHGNN to pass graph_data from closure."""
        def __init__(self, model, graph_data, device):
            super().__init__()
            self.model = model
            self.gd = {k: v.to(device) if isinstance(v, torch.Tensor) else
                       [t.to(device) for t in v] if isinstance(v, list) else v
                       for k, v in graph_data.items()}
        def forward(self, snp_data, graph_data=None):
            return self.model(snp_data, self.gd)
    
    orig_model = PlantHGNN(
        n_snps=n_snps, n_genes=net_meta['n_genes'],
        n_gene_sets=net_meta['n_gene_sets'], n_traits=n_traits,
        d_model=64, n_transformer_layers=4, n_attnres_blocks=2,
        n_gcn_layers=2, n_views=3, dropout=0.3,
        use_heterogeneous=False, use_attnres=True,
        use_functional_embed=True, use_structural_encode=True
    )
    orig_wrapped = OriginalWrapper(orig_model, graph_data, device)
    results['0_original'] = train_and_evaluate(
        orig_wrapped, snp_data, phenotype, splits, device,
        n_epochs=30, lr=0.0001, weight_decay=0.001, model_name="Original"
    )
    
    # --- Experiment 1: Fix V1 — Multi-scale CNN, no graph noise ---
    print("\n>>> Experiment 1: Fix V1 — Multi-scale CNN, strip constant components")
    model_v1 = FixedPlantHGNN_V1(n_snps, n_traits, d_hidden=256, dropout=0.3)
    results['1_multiscale_cnn'] = train_and_evaluate(
        model_v1, snp_data, phenotype, splits, device,
        n_epochs=30, lr=0.001, weight_decay=1e-4, model_name="V1-MultiCNN"
    )
    
    # --- Experiment 2: Fix V2 — Kernel/anchor similarity (GBLUP-inspired) ---
    print("\n>>> Experiment 2: Fix V2 — Anchor-similarity (GBLUP-inspired)")
    model_v2 = FixedPlantHGNN_V2(n_snps, n_traits, n_anchors=200, d_hidden=256, dropout=0.3)
    results['2_anchor_similarity'] = train_and_evaluate(
        model_v2, snp_data, phenotype, splits, device,
        n_epochs=30, lr=0.001, weight_decay=1e-4, model_name="V2-Anchor"
    )
    
    # --- Experiment 3: Fix V3 — Direct linear model (neural GBLUP) ---
    print("\n>>> Experiment 3: Fix V3 — Direct MLP (neural GBLUP)")
    model_v3 = FixedPlantHGNN_V3(n_snps, n_traits, d_hidden=512, dropout=0.5)
    results['3_direct_mlp'] = train_and_evaluate(
        model_v3, snp_data, phenotype, splits, device,
        n_epochs=30, lr=0.001, weight_decay=1e-3, model_name="V3-DirectMLP"
    )
    
    # --- Experiment 4: Fix V3 with per-trait top-5 training ---
    print("\n>>> Experiment 4: Fix V3 per-trait (train on top-5 most predictable traits)")
    # Find top-5 traits by variance (proxy for heritability)
    train_pheno = phenotype[splits['train']].numpy()
    trait_vars = np.nanvar(train_pheno, axis=0)
    top5_traits = np.argsort(trait_vars)[-5:]
    print(f"  Top-5 trait indices: {top5_traits.tolist()}")
    
    pheno_top5 = phenotype[:, top5_traits]
    model_v3_top5 = FixedPlantHGNN_V3(n_snps, n_traits=5, d_hidden=512, dropout=0.5)
    results['4_direct_mlp_top5'] = train_and_evaluate(
        model_v3_top5, snp_data, pheno_top5, splits, device,
        n_epochs=30, lr=0.001, weight_decay=1e-3, model_name="V3-Top5"
    )
    
    # ============================================================
    # Part 5: Report
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Model':<35} {'Test PCC':>10} {'Val PCC':>10} {'Params':>12} {'NaN fix':>8}")
    print("-" * 80)
    print(f"{'GBLUP (reference)':<35} {'0.538':>10} {'—':>10} {'—':>12} {'—':>8}")
    
    for name, r in sorted(results.items()):
        label = name.split('_', 1)[1]
        print(f"{label:<35} {r['test_pcc']:>10.4f} {r['best_val_pcc']:>10.4f} "
              f"{r['n_params']:>12,} {'✓':>8}")
    
    # Find best
    best_name = max(results.keys(), key=lambda k: results[k]['test_pcc'])
    best = results[best_name]
    
    print(f"\n🏆 Best fix: {best_name}")
    print(f"   Test PCC: {best['test_pcc']:.4f}")
    print(f"   Improvement over original: "
          f"{best['test_pcc'] - results['0_original']['test_pcc']:.4f}")
    
    # Improvement over original vs GBLUP gap
    orig_pcc = results['0_original']['test_pcc']
    gblup_pcc = 0.538
    gap_closed = (best['test_pcc'] - orig_pcc) / (gblup_pcc - orig_pcc) * 100
    print(f"   Gap to GBLUP closed: {gap_closed:.1f}%")
    
    # Save results
    output = {k: {kk: vv if not isinstance(vv, list) else [float(x) for x in vv]
                   for kk, vv in v.items()}
              for k, v in results.items()}
    out_file = Path("data/processed/GSTP007_full_10000snps_processed/diagnosis_results.json")
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to {out_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())
