# PlantMEGNN: Multi-View Graph Neural Network for Genomic Prediction in Rice

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **PlantMEGNN: Multi-View Graph Neural Network Improves Genomic Prediction Accuracy for Rice Breeding** (submitted to *Plant Phenomics*).

## Overview

PlantMEGNN is a multi-view graph neural network framework that integrates genomic data with three complementary biological networks (PPI, GO, KEGG) to improve genomic prediction accuracy for rice breeding.

### Key Features

- **Multi-view network integration**: Simultaneously leverages PPI, GO, and KEGG networks
- **Interpretable attention**: Learned attention weights reveal trait-specific biological preferences
- **Cross-population generalizability**: Validated on 3 independent populations (2,578 accessions)
- **Multi-environment support**: G×E modeling with environment similarity graphs

## Installation

```bash
# Clone the repository
git clone https://github.com/nblvguohao/PlantMEGNN.git
cd PlantMEGNN

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (CUDA 12.1)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### Requirements

- Python 3.8+
- PyTorch 2.1+
- PyTorch Geometric 2.4+
- CUDA 12.1 (optional, for GPU support)

## Quick Start

### 1. Data Preparation

Download the preprocessed data:

```bash
# GSTP007 (primary dataset)
python scripts/download_data.py --dataset gstp007

# Or use your own data
python scripts/preprocess.py --vcf your_data.vcf --pheno your_phenotype.csv
```

### 2. Train PlantMEGNN

```bash
# Train on GSTP007 (single environment)
python scripts/train_planthgnn.py --dataset gstp007 --trait Grain_Length

# Multi-environment training (GSTP008)
python scripts/train_planthgnn.py --dataset gstp008 --multitask --env_graph
```

### 3. Evaluate Model

```bash
# Run 5-fold cross-validation
python scripts/evaluate.py --dataset gstp007 --trait Grain_Length --n_folds 5 --seeds 42 123 456

# Cross-population transfer
python scripts/evaluate_transfer.py --source gstp007 --target gstp008
```

## Repository Structure

```
PlantMEGNN/
├── data/                      # Data directory (not in git)
│   ├── processed/             # Preprocessed data
│   └── raw/                   # Raw downloaded data
├── src/                       # Source code
│   ├── models/                # Model implementations
│   │   ├── plantmegnn.py      # Main PlantMEGNN model
│   │   ├── multiview_gcn.py   # Multi-view GCN encoders
│   │   └── attention.py       # LightGatedFusion mechanism
│   ├── data/                  # Data loading utilities
│   └── training/              # Training loops
├── scripts/                   # Executable scripts
│   ├── train_planthgnn.py     # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── benchmark_gstp008.py   # GSTP008 multi-environment benchmark
│   └── build_networks.py      # Biological network construction
├── experiments/               # Experiment configurations
├── results/                   # Experiment results
├── paper/                     # Paper materials
└── README.md                  # This file
```

## Main Results

### GSTP007 (Primary Breeding Panel, n=1,495)

| Method | Avg PCC | vs GBLUP |
|--------|---------|----------|
| GBLUP | 0.740 | - |
| DNNGP | 0.749 | +1.2% |
| NetGP | 0.757 | +2.3% |
| **PlantMEGNN** | **0.786** | **+6.2%** |

### Cross-Population Validation

| Population | n | PlantMEGNN | Improvement |
|------------|---|------------|-------------|
| GSTP007 | 1,495 | 0.832 | +5.5% |
| GSTP008 | 705 | 0.804 | +5.4% |
| GSTP009 | 378 | 0.661 | +7.8% |

### Multi-Environment (GSTP008)

| Environment | MT-NetGP-E | ST-Ridge | Improvement |
|-------------|------------|----------|-------------|
| BeiJ15 | 0.283 | 0.232 | +22.0% |
| WenJ15 | 0.478 | 0.467 | +2.2% |
| YangZ15 | 0.258 | 0.162 | +59.2% |
| LingS15 | 0.340 | 0.433 | -21.4% |
| **Average** | **0.340** | **0.324** | **+5.0%** |

## Pre-trained Models

Download pre-trained models:

```bash
# Download all models
python scripts/download_models.py

# Or download specific dataset
python scripts/download_models.py --dataset gstp007
```

Available models:
- GSTP007 (7 traits)
- GSTP008 (4 environments)
- GSTP009 (4 traits)

## Usage Examples

### Example 1: Predict New Accessions

```python
import torch
from src.models.plantmegnn import PlantMEGNN

# Load pre-trained model
model = PlantMEGNN.from_pretrained('gstp007_grain_length')
model.eval()

# Prepare your data
snp_features = ...  # Your SNP data

# Predict
with torch.no_grad():
    prediction = model.predict(snp_features)
```

### Example 2: Extract Attention Weights

```python
# Get attention weights for interpretability
attention_weights = model.get_attention_weights(snp_features)
print(f"PPI weight: {attention_weights['PPI']:.3f}")
print(f"GO weight: {attention_weights['GO']:.3f}")
print(f"KEGG weight: {attention_weights['KEGG']:.3f}")
```

### Example 3: Multi-Environment Prediction

```python
# Train multi-task model for G×E modeling
from scripts.benchmark_gstp008_multitask import run_multitask

results = run_multitask(
    X=snp_matrix,
    Y=phenotype_matrix,
    environments=['BeiJ15', 'WenJ15', 'YangZ15', 'LingS15'],
    model='MT-NetGP-E'
)
```

## Datasets

### GSTP007
- **Accessions**: 1,495
- **Source**: 3K Rice Genome Project / Rice Diversity Panel
- **Traits**: Plant Height, Grain Length, Grain Width, Days to Heading, Panicle Length, Grain Weight, Yield per Plant
- **Availability**: [RAP-DB](https://rapdb.dna.affrc.go.jp/), [Rice Diversity](http://www.ricediversity.org/)

### GSTP008
- **Accessions**: 705
- **Source**: Chinese natural rice population (LCZ)
- **Environments**: Beijing, Wenjiang, Yangzhou, Lingshui (2015)
- **Availability**: [CropGS-Hub](https://iagr.genomics.cn/CropGS)

### GSTP009
- **Accessions**: 378
- **Source**: USDA Rice Diversity Panel
- **Availability**: [USDA GRIN](https://www.ars-grin.gov/)

## Biological Networks

- **PPI**: STRING v12.0 for *Oryza sativa* ([link](https://string-db.org/))
- **GO**: Gene Ontology Consortium ([link](http://geneontology.org/))
- **KEGG**: KEGG Pathway Database ([link](https://www.genome.jp/kegg/))

## Citation

If you use PlantMEGNN in your research, please cite:

```bibtex
@article{lv2026plantmegnn,
  title={PlantMEGNN: Multi-View Graph Neural Network Improves Genomic Prediction Accuracy for Rice Breeding},
  author={Lv, Guohao and Xia, Yingchun and Wang, Xiaosong and Li, Xiaowei and Zhu, Xiaolei and Yang, Shuai and Wang, Qingyong and Gu, Lichuan},
  journal={Plant Phenomics},
  year={2026},
  publisher={Science Partner Journal Program}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please:

- Open an issue on GitHub
- Contact: wqy@ahau.edu.cn (Qingyong Wang)
- Website: [Anhui Agricultural University](http://www.ahau.edu.cn/)

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (32472007, 62301006, 62301008) and the Natural Science Foundation of Anhui Province (2308085MF217, 2308085QF202).

We thank the 3K Rice Genome Project, CropGS-Hub, and USDA GRIN for providing the genomic and phenotypic data used in this study.

## FAQ

**Q: Can PlantMEGNN be applied to other crops?**

A: Yes, with modification. You need to replace the rice biological networks (PPI, GO, KEGG) with species-specific networks for your crop of interest.

**Q: How much GPU memory is required?**

A: Training requires ~8GB VRAM for batch size 128. Inference requires only ~2GB.

**Q: Can I use PlantMEGNN for traits not in the paper?**

A: Yes. The model can be trained on any continuous phenotype. Ensure you have sufficient sample size (>200 recommended).

**Q: How long does training take?**

A: ~10 minutes for 1,495 samples on NVIDIA RTX 3090. Training time scales linearly with dataset size.

## Changelog

### v1.0.0 (2026-03-30)
- Initial release
- GSTP007 single-environment models
- GSTP008 multi-environment support
- Pre-trained models for 7 traits

---

**Note**: This repository is under active development. Please check back regularly for updates and improvements.
