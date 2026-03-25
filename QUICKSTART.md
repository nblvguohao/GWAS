# PlantHGNN Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### 1. Environment Setup

```bash
cd /data/lgh/GWAS

# Create conda environment
conda create -n planthgnn python=3.10
conda activate planthgnn

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (adjust CUDA version as needed)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### 2. Verify Installation

```bash
# Test core modules
python src/models/attention_residual.py
python src/models/multi_view_gcn.py
python src/models/plant_hgnn.py
python src/models/baselines/gblup.py
```

Expected output: "test passed!" for each module.

## 📊 Data Preparation

### Option A: Automated Download (Recommended)

```bash
# Download CropGS datasets
python src/data/download.py --dataset rice469 maize282 soybean999 wheat599

# Download biological networks
python src/data/download.py --networks
```

### Option B: Manual Download

If automated download fails:

1. **CropGS-Hub**: https://iagr.genomics.cn/CropGS
   - Download genotype and phenotype CSV files
   - Place in `data/raw/cropgs/{dataset_name}/`

2. **STRING Database**: https://string-db.org/
   - Download protein.links files for rice/maize/soybean
   - Place in `data/raw/networks/`

3. **GO Annotations**: http://current.geneontology.org/annotations/
   - Download .gaf.gz files
   - Place in `data/raw/annotations/`

### Preprocessing Pipeline

```bash
# Example: Preprocess rice469 dataset
python src/data/preprocess.py \
    --genotype data/raw/cropgs/rice469/rice469_genotype.csv \
    --phenotype data/raw/cropgs/rice469/rice469_phenotype.csv \
    --output-dir data/processed \
    --dataset-name rice469 \
    --missing-threshold 0.1 \
    --maf-threshold 0.05 \
    --corr-threshold 0.3 \
    --vif-threshold 10

# Build biological networks
python src/data/network_builder.py \
    --species oryza_sativa \
    --output-dir data/processed/graphs \
    --string-threshold 700

# Create data splits
python src/data/splits.py \
    --strategy random \
    --n-samples 469 \
    --n-folds 5 \
    --output data/processed/splits/rice469_random_split.json \
    --seed 42
```

## 🧪 Running Experiments

### Local Experiments (Small Datasets)

```bash
# Run single experiment
python experiments/run_experiment.py \
    --config experiments/configs/base_config.yaml \
    --dataset rice469 \
    --output-dir experiments/results/test_run \
    --seed 42 \
    --gpu 0

# Run batch experiments
bash experiments/scripts/run_local.sh

# Run ablation studies
bash experiments/scripts/run_ablation.sh
```

### Configuration Options

Edit `experiments/configs/base_config.yaml`:

```yaml
model:
  d_model: 128              # Hidden dimension
  n_transformer_layers: 6   # Number of transformer layers
  n_attnres_blocks: 8       # Number of AttnRes blocks
  use_attnres: true         # Enable/disable AttnRes
  use_functional_embed: true # Enable/disable functional embedding

training:
  lr: 0.001                 # Learning rate
  batch_size: 32            # Batch size (adjust for GPU memory)
  max_epochs: 200           # Maximum epochs
  early_stopping_patience: 20
```

## 📈 Baseline Models

### GBLUP (Statistical Baseline)

```python
from src.models.baselines.gblup import GBLUP
import numpy as np

# Load data
X_train = np.load('data/processed/rice469/train_genotype.npy')
y_train = np.load('data/processed/rice469/train_phenotype.npy')
X_test = np.load('data/processed/rice469/test_genotype.npy')

# Train GBLUP
model = GBLUP(lambda_reg=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Adding New Baselines

Create new file in `src/models/baselines/`:

```python
from .base import BaselineModel

class MyBaseline(BaselineModel):
    def __init__(self):
        super().__init__(name="MyBaseline")
    
    def fit(self, X_train, y_train, **kwargs):
        # Training logic
        pass
    
    def predict(self, X_test, **kwargs):
        # Prediction logic
        pass
```

## 🔍 Analysis & Visualization

### Network Contribution Analysis

```python
from src.models.plant_hgnn import PlantHGNN

# Load trained model
model = PlantHGNN(...)
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

# Get network attention weights
network_attn = model.get_network_attention_weights()
print(f"PPI: {network_attn[0]:.3f}")
print(f"GO: {network_attn[1]:.3f}")
print(f"KEGG: {network_attn[2]:.3f}")

# Get depth attention weights
depth_attn = model.get_depth_attention_weights()
```

### Evaluation Metrics

```python
from src.training.metrics import compute_metrics

metrics = compute_metrics(y_true, y_pred, 
                         metrics=['pearson', 'spearman', 'mse', 'ndcg'])
print(f"Pearson: {metrics['pearson']:.4f}")
print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
```

## 🖥️ Server Deployment (AutoDL)

### 1. Prepare Data Package

```bash
# On local machine
tar -czf planthgnn_data.tar.gz data/processed/
```

### 2. Upload to Server

```bash
# Upload code
scp -r /data/lgh/GWAS/ server:/workspace/

# Upload data
scp planthgnn_data.tar.gz server:/workspace/GWAS/
```

### 3. Run on Server

```bash
# On server
cd /workspace/GWAS
tar -xzf planthgnn_data.tar.gz

# Install environment
conda create -n planthgnn python=3.10
conda activate planthgnn
pip install -r requirements.txt

# Run experiments
python experiments/run_experiment.py \
    --config experiments/configs/base_config.yaml \
    --dataset rice469 \
    --output-dir results/ \
    --gpu 0
```

## 📊 Results Analysis

### Generate Summary Tables

```python
import pandas as pd
import json
from pathlib import Path

results_dir = Path('experiments/results/')
all_results = []

for result_file in results_dir.rglob('test_results.json'):
    with open(result_file) as f:
        data = json.load(f)
        all_results.append({
            'dataset': result_file.parent.parent.name,
            'seed': result_file.parent.name,
            'pearson': data['metrics']['pearson'],
            'mse': data['metrics']['mse']
        })

df = pd.DataFrame(all_results)
summary = df.groupby('dataset').agg(['mean', 'std'])
print(summary)
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```yaml
# Reduce batch size in config
training:
  batch_size: 16  # or 8
```

### Import Errors

```bash
# Add project to PYTHONPATH
export PYTHONPATH=/data/lgh/GWAS:$PYTHONPATH
```

### Data Download Fails

- Check internet connection
- Try manual download from sources
- Verify file paths in download.py

### Model Training Diverges

```yaml
# Reduce learning rate
training:
  lr: 0.0001
  
# Add gradient clipping
training:
  gradient_clip: 0.5
```

## 📚 Key Files Reference

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Complete research plan and milestones |
| `PROJECT_STATUS.md` | Current implementation status |
| `README.md` | Project overview |
| `requirements.txt` | Python dependencies |
| `experiments/configs/base_config.yaml` | Default configuration |
| `src/models/plant_hgnn.py` | Main model |
| `src/training/trainer.py` | Training loop |
| `src/data/preprocess.py` | Data preprocessing |

## 🎯 Next Steps

1. **Download data** (see Data Preparation section)
2. **Run preprocessing** on rice469 dataset
3. **Test training** for 10 epochs to verify setup
4. **Implement remaining baselines** (DNNGP, NetGP, GPformer)
5. **Run full experiments** on all datasets
6. **Analyze results** and generate paper figures

## 💡 Tips

- Start with rice469 (smallest dataset) for testing
- Use `--seed` parameter for reproducibility
- Save checkpoints frequently during long training runs
- Monitor GPU memory usage with `nvidia-smi`
- Use WandB for experiment tracking (set `use_wandb: true` in config)

## 📞 Support

- GitHub: https://github.com/nblvguohao/GWAS
- Email: nblvguohao@gmail.com
- Documentation: See `CLAUDE.md` for detailed research plan

## 📄 License

MIT License - See LICENSE file for details
