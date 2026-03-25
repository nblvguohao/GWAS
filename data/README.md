# Data Directory

This directory contains datasets and biological networks for PlantHGNN.

## Directory Structure

```
data/
├── raw/                    # Raw downloaded data (not in git)
│   ├── cropgs/             # CropGS-Hub datasets
│   │   ├── rice469/
│   │   ├── maize282/
│   │   ├── soybean999/
│   │   └── wheat599/
│   ├── networks/           # Biological networks
│   │   ├── oryza_sativa_string_v12.txt.gz
│   │   ├── zea_mays_string_v12.txt.gz
│   │   └── glycine_max_string_v12.txt.gz
│   └── annotations/        # GO/KEGG annotations
│       ├── rice_go.gaf.gz
│       ├── maize_go.gaf.gz
│       └── soybean_go.gaf.gz
│
├── processed/              # Preprocessed data (not in git)
│   ├── graphs/             # PyG format graph data
│   │   ├── rice469_ppi.pt
│   │   ├── rice469_go.pt
│   │   └── rice469_kegg.pt
│   └── splits/             # Data split indices
│       ├── rice469_random_split.json
│       ├── rice469_chromosome_split.json
│       └── rice469_line_split.json
│
└── README.md               # This file
```

## Data Sources

### CropGS-Hub Datasets

Download from: https://iagr.genomics.cn/CropGS

**Available datasets:**
- **rice469**: 469 rice accessions, 5,291 SNPs, 6 traits
- **maize282**: 282 maize lines, 3,093 SNPs, 3 traits
- **soybean999**: 999 soybean accessions, 7,883 SNPs, 6 traits
- **wheat599**: 599 wheat lines, 1,447 SNPs, 3 traits

### Biological Networks

**STRING Database (v12)**
- Download: https://string-db.org/
- Species: Oryza sativa (rice), Zea mays (maize), Glycine max (soybean)
- Filter: combined_score > 700

**GO Annotations**
- Download: http://current.geneontology.org/annotations/
- Files: osa.gaf.gz (rice), zma.gaf.gz (maize), gma.gaf.gz (soybean)

**KEGG Pathways**
- Access via KEGG REST API
- Species codes: osa (rice), mze (maize), gmx (soybean)

**PlantTFDB (v5)**
- Download: http://planttfdb.gao-lab.org/download.php
- Files: TF-target regulatory relationships

## Download Instructions

### Automated Download

```bash
# Download CropGS datasets
python src/data/download.py --dataset rice469 maize282 soybean999 wheat599

# Download biological networks
python src/data/download.py --networks
```

### Manual Download

If automated download fails, manually download from the sources above and place files in the appropriate directories.

## Preprocessing

After downloading raw data, run preprocessing:

```bash
# Preprocess a dataset
python src/data/preprocess.py \
    --genotype data/raw/cropgs/rice469/rice469_genotype.csv \
    --phenotype data/raw/cropgs/rice469/rice469_phenotype.csv \
    --output-dir data/processed \
    --dataset-name rice469

# Build biological networks
python src/data/network_builder.py \
    --species oryza_sativa \
    --output-dir data/processed/graphs

# Create data splits
python src/data/splits.py \
    --strategy random \
    --n-samples 469 \
    --n-folds 5 \
    --output data/processed/splits/rice469_random_split.json
```

## Data Format

### Genotype Data
- Format: CSV
- Rows: Samples/accessions
- Columns: SNP markers
- Values: 0 (AA), 1 (AB), 2 (BB), -1 or NaN (missing)

### Phenotype Data
- Format: CSV
- Rows: Samples/accessions (matching genotype)
- Columns: Traits
- Values: Continuous trait values

### Network Data
- Format: PyTorch Geometric Data objects (.pt files)
- Contains: node features, edge indices, edge weights

## Citation

If you use these datasets, please cite:

- **CropGS-Hub**: Zhang et al. (2024)
- **STRING**: Szklarczyk et al. (2023)
- **GO**: Gene Ontology Consortium (2023)
- **PlantTFDB**: Jin et al. (2017)
