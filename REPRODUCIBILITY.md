# Reproducibility Guide for PlantMEGNN

This document provides step-by-step instructions to reproduce all experiments, tables, and figures reported in the manuscript "PlantMEGNN: Multi-View Graph Neural Networks for Single- and Multi-Environment Genomic Prediction in Rice."

## Environment Setup

```bash
# Create conda environment
conda create -n planthgnn python=3.10
conda activate planthgnn

# Install core dependencies
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install numpy pandas scikit-learn scipy matplotlib seaborn networkx pyyaml tqdm

# Optional: for LightGBM baseline
pip install lightgbm
```

## Data Preprocessing

### GSTP007 (Single-environment)

```bash
python src/data/preprocess_gstp007.py \
    --input data/raw/gstp007 \
    --output data/processed/gstp007 \
    --n_snps 5000
```

This produces:
- `snp_features.npy` (1,495 × 5,000)
- `phenotypes.csv` (7 traits, BLUP-adjusted)
- `networks/ppi_network.pt`, `go_network.pt`, `kegg_network.pt`

### GSTP008 (Multi-environment)

```bash
python src/data/preprocess_gstp008.py \
    --input data/raw/gstp008 \
    --output data/processed/gstp008 \
    --n_snps 5000
```

## Reproduce Main Experiments

### GSTP007 — 5-fold CV × 3 seeds

```bash
for trait in Plant_Height Grain_Length Grain_Width Days_to_Heading Panicle_Length Grain_Weight Yield_per_plant; do
    for seed in 42 123 456; do
        python experiments/run_plantmegnn.py \
            --dataset gstp007 \
            --trait $trait \
            --config configs/plantmegnn_3view.yaml \
            --seed $seed
    done
done
```

Results are written to `results/gstp007/multiview_results_v3.json`.

### GSTP008 — Leave-one-environment-out CV

```bash
for seed in 42 123 456; do
    python experiments/run_plantmegnn_multienv.py \
        --dataset gstp008 \
        --config configs/plantmegnn_e.yaml \
        --seed $seed
done
```

Results are written to `results/gstp008/multitask_results.json`.

## Reproduce Baselines

### GBLUP, Ridge, DNNGP, Transformer, NetGP

```bash
python experiments/run_baselines.py --dataset gstp007 --baselines all
```

### LightGBM

```bash
python scripts/run_lightgbm_baseline.py --dataset gstp007 --n_snps 5000
```

## Reproduce Tables

```bash
python scripts/generate_paper_tables_from_truth.py
```

Outputs:
- `paper/tables/table1a_gstp007.tex`
- `paper/tables/table1b_gstp008.tex`
- `paper/tables/table2a_ablation.tex`
- `paper/tables/table3_attention.tex`
- `paper/tables/table5_statistical_tests.tex`

## Reproduce Figures

```bash
# Figure 1 — Architecture
python scripts/generate_fig1_architecture_v2.py

# Figure 4 — Attention weights (lollipop plot)
python scripts/generate_fig4_attention_lollipop.py

# Figure S3 — GS3 neighborhood
python scripts/generate_figS3_gs3_network.py
```

## Statistical Validation

```bash
# Wilcoxon signed-rank tests (1view vs 3view)
python scripts/compute_view_wilcoxon.py

# Output: results/gstp007/view_comparison_wilcoxon.json
```

## Expected Runtime

| Experiment | Hardware | Approximate Time |
|---|---|---|
| GSTP007 full benchmark (7 traits × 5 folds × 3 seeds) | RTX 4060 (8 GB) | ~6 hours |
| GSTP008 LOEO-CV (4 env × 3 seeds) | RTX 4060 (8 GB) | ~2 hours |
| LightGBM baseline | CPU | ~5 minutes |
| Table / figure generation | CPU | ~1 minute |

## Contact

For questions about reproduction, please open an issue at https://github.com/nblvguohao/GWAS or contact the corresponding authors.
