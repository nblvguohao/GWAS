# PlantHGNN Project Status

**Last Updated**: 2026-03-25  
**Status**: ✅ Initial Implementation Complete

## Project Overview

PlantHGNN (Plant Heterogeneous Graph Neural Network) is a novel deep learning framework for plant genomic prediction that combines:
- **Heterogeneous Graph Neural Networks** (from cancer gene prediction domain)
- **Attention Residuals** (Kimi's AttnRes mechanism)
- **Multi-view biological networks** (PPI, GO, KEGG)
- **Multi-trait genomic prediction**

**Target Journals**: Plant Biotechnology Journal (IF ~9.5), Briefings in Bioinformatics (IF ~9.5)

## Implementation Status

### ✅ Completed Components

#### 1. Project Structure
- [x] Complete directory structure created
- [x] Git repository initialized with first commit
- [x] Documentation files (README.md, data/README.md, CLAUDE.md)
- [x] Configuration management (YAML configs)

#### 2. Data Processing Modules (`src/data/`)
- [x] **download.py**: Automated data download from CropGS-Hub, STRING, GO, KEGG
- [x] **preprocess.py**: SNP quality control, PCS feature selection, one-hot encoding
- [x] **network_builder.py**: PPI, GO, KEGG network construction, heterogeneous GTM network
- [x] **splits.py**: Three splitting strategies (random, chromosome, line-based)

#### 3. Core Model Implementations (`src/models/`)
- [x] **attention_residual.py**: Kimi AttnRes implementation with block-wise attention
- [x] **multi_view_gcn.py**: Multi-view GCN encoder with attention fusion
- [x] **functional_embed.py**: Functional embedding + structural encoding modules
- [x] **plant_hgnn.py**: Complete PlantHGNN model integrating all components

#### 4. Training Infrastructure (`src/training/`)
- [x] **trainer.py**: Training loop with early stopping, checkpointing
- [x] **losses.py**: MSE, MAE, Huber, multi-task, ranking losses
- [x] **metrics.py**: PCC, Spearman, NDCG, MSE, MAE, Wilcoxon test

#### 5. Baseline Models (`src/models/baselines/`)
- [x] **base.py**: Abstract base class for unified interface
- [x] **gblup.py**: GBLUP (Genomic BLUP) implementation

#### 6. Experiment Configuration
- [x] Base configuration (base_config.yaml)
- [x] Ablation configs (no_attnres, no_functional_embed, single_view)
- [x] Experiment runner script (run_experiment.py)
- [x] Shell scripts for local and ablation experiments

## Code Statistics

```
Total Files: 35
Total Lines: ~5,200
Python Modules: 18
Configuration Files: 4
Shell Scripts: 2
Documentation: 3
```

### Module Breakdown

| Module | Lines | Description |
|--------|-------|-------------|
| plant_hgnn.py | ~450 | Main model architecture |
| attention_residual.py | ~350 | AttnRes implementation |
| preprocess.py | ~350 | SNP preprocessing pipeline |
| network_builder.py | ~400 | Biological network construction |
| trainer.py | ~300 | Training infrastructure |
| multi_view_gcn.py | ~250 | Multi-view GCN encoder |
| metrics.py | ~300 | Evaluation metrics |
| Others | ~2,800 | Supporting modules |

## Next Steps (按 CLAUDE.md 里程碑)

### 🔄 Milestone 1: Data Preparation (1-2 weeks)
**Status**: Infrastructure ready, needs data download

- [ ] T1.1: Download CropGS-Hub datasets (rice469, maize282, soybean999, wheat599)
- [ ] T1.2: Download biological networks (STRING, GO, KEGG, PlantTFDB)
- [ ] T1.3: Run PCS feature selection on rice469
- [ ] T1.4: Build biological networks for rice
- [ ] T1.5: Create data splits (random, chromosome, line)
- [ ] T1.6: Verify PyG format graph data

**Commands to run**:
```bash
# Download data
python src/data/download.py --dataset rice469 maize282 --networks

# Preprocess (after manual download if needed)
python src/data/preprocess.py \
    --genotype data/raw/cropgs/rice469/rice469_genotype.csv \
    --phenotype data/raw/cropgs/rice469/rice469_phenotype.csv \
    --output-dir data/processed \
    --dataset-name rice469

# Build networks
python src/data/network_builder.py \
    --species oryza_sativa \
    --output-dir data/processed/graphs

# Create splits
python src/data/splits.py \
    --strategy random \
    --n-samples 469 \
    --output data/processed/splits/rice469_random_split.json
```

### 📋 Milestone 2: Baseline Reproduction (1-2 weeks)
**Status**: GBLUP implemented, others pending

- [ ] T2.1: ✅ GBLUP implemented
- [ ] T2.2: Implement/adapt DNNGP
- [ ] T2.3: Implement/adapt NetGP
- [ ] T2.4: Implement/adapt GPformer
- [ ] T2.5: Verify baseline PCC matches literature

**Required baselines**:
- GBLUP ✅
- DNNGP (DNN-based)
- NetGP (GCN-based, direct competitor)
- GPformer (Transformer-based)
- Cropformer (CNN+Attention, for server)

### 🧪 Milestone 3: Main Model Training (1-2 weeks)
**Status**: Model implemented, needs data integration

- [ ] T3.1: ✅ AttnRes implemented
- [ ] T3.2: ✅ Multi-view GCN implemented
- [ ] T3.3: ✅ Functional embedding implemented
- [ ] T3.4: ✅ PlantHGNN integrated
- [ ] T3.5: ✅ Training loop implemented
- [ ] T3.6: Create PyG Dataset class for data loading
- [ ] T3.7: Test training on rice469 (10 epochs)

### 🖥️ Milestone 4: Main Experiments (Server, 2-3 weeks)
**Status**: Not started, requires Milestone 1-3

- [ ] T4.1: Hyperparameter search
- [ ] T4.2: Large-scale baseline reproduction
- [ ] T4.3: Main experiment (6 datasets × 5 folds × 5 seeds)
- [ ] T4.4: Generate result tables
- [ ] T4.5: Verify PlantHGNN > NetGP (p<0.05)

### 📊 Milestone 5: Analysis Experiments (1-2 weeks)
**Status**: Analysis modules pending

- [ ] T5.1: Ablation experiments
- [ ] T5.2: Network contribution analysis
- [ ] T5.3: AttnRes depth attention analysis
- [ ] T5.4: SNP importance (SHAP)
- [ ] T5.5: UMAP visualization
- [ ] T5.6: Statistical significance tests

### 📝 Milestone 6: Paper Writing (2-3 weeks)
**Status**: Not started

- [ ] T6.1: Generate all figures and tables
- [ ] T6.2: Write manuscript sections
- [ ] T6.3: Format for target journal

## Key Features Implemented

### 1. Attention Residuals (AttnRes)
- ✅ Block-wise attention aggregation
- ✅ Trainable query vectors (not input-dependent)
- ✅ Compatible with standard Transformer layers
- ✅ Interpretability: depth attention weights extraction

### 2. Multi-View GCN
- ✅ Separate encoding for PPI, GO, KEGG networks
- ✅ Learnable attention fusion weights
- ✅ Batch normalization and dropout
- ✅ View-specific embedding extraction

### 3. Functional & Structural Encoding
- ✅ Gene set membership-based functional embedding
- ✅ Random walk positional encoding
- ✅ PageRank centrality encoding
- ✅ Fusion mechanism

### 4. Data Processing Pipeline
- ✅ Quality control (missing rate, MAF filtering)
- ✅ PCS feature selection (Pearson + VIF)
- ✅ One-hot SNP encoding
- ✅ Three splitting strategies

### 5. Training Infrastructure
- ✅ Early stopping with patience
- ✅ Learning rate scheduling
- ✅ Gradient clipping
- ✅ Checkpoint saving/loading
- ✅ Comprehensive metrics tracking

## Hardware Requirements

### Local (4060 8G VRAM)
**Can run**:
- Small datasets (rice469, maize282)
- All preprocessing
- Model debugging
- Ablation experiments (small scale)
- Visualization

**Limitations**:
- Max batch size: 32
- Max SNPs: ~5,000 (after PCS)
- Max model params: ~50M

### Server (Recommended: A100 40G)
**Required for**:
- Large datasets (rice3k, wheat2403)
- Full baseline reproduction
- Hyperparameter search
- 5-fold × 5-seed experiments
- Large-scale network construction

**Estimated GPU hours**: 80-120

## Testing Status

All core modules include test functions:
- ✅ `test_attn_res()` - AttnRes mechanism
- ✅ `test_multi_view_gcn()` - Multi-view encoding
- ✅ `test_functional_embed()` - Functional embedding
- ✅ `test_plant_hgnn()` - Complete model
- ✅ `test_gblup()` - GBLUP baseline
- ✅ `test_metrics()` - Evaluation metrics
- ✅ `test_losses()` - Loss functions

**Run tests**:
```bash
python src/models/attention_residual.py
python src/models/multi_view_gcn.py
python src/models/functional_embed.py
python src/models/plant_hgnn.py
python src/models/baselines/gblup.py
python src/training/metrics.py
python src/training/losses.py
```

## Dependencies

All dependencies specified in `requirements.txt`:
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.4.0
- NumPy, Pandas, Scikit-learn
- NetworkX, SHAP, UMAP
- Visualization: Matplotlib, Seaborn

**Installation**:
```bash
conda create -n planthgnn python=3.10
conda activate planthgnn
pip install -r requirements.txt
```

## Repository Information

- **GitHub**: https://github.com/nblvguohao/GWAS.git
- **Branch**: master
- **Commits**: 1 (initial structure)
- **Author**: Lyu (nblvguohao@gmail.com)
- **Institution**: 安徽农业大学 AI学院 智慧农业重点实验室

## Critical Next Actions

1. **Download Data** (Priority: HIGH)
   - Manually download CropGS-Hub datasets if automated download fails
   - Download STRING networks for rice, maize, soybean
   - Download GO annotations

2. **Create PyG Dataset Class** (Priority: HIGH)
   - Implement `PlantGPDataset` in `src/data/graph_dataset.py`
   - Handle SNP features + graph data + phenotypes
   - Support batch loading

3. **Test End-to-End Pipeline** (Priority: HIGH)
   - Run preprocessing on rice469
   - Build networks
   - Train PlantHGNN for 10 epochs
   - Verify training loss decreases

4. **Implement Remaining Baselines** (Priority: MEDIUM)
   - DNNGP (simple DNN)
   - NetGP (GCN, main competitor)
   - GPformer (Transformer)

5. **Prepare for Server Deployment** (Priority: MEDIUM)
   - Test local experiments work
   - Prepare data upload to server
   - Configure AutoDL environment

## Notes

- All code follows the architecture specified in CLAUDE.md
- Model design based on TREE/GRAFT papers (cancer gene prediction)
- AttnRes implementation follows Kimi paper (arxiv 2603.15031)
- Fair comparison ensured: same preprocessing, same splits, same metrics
- Statistical testing (Wilcoxon) built-in for significance claims

## Contact

For questions or issues:
- Email: nblvguohao@gmail.com
- GitHub Issues: https://github.com/nblvguohao/GWAS/issues
