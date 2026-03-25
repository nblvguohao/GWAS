#!/bin/bash
# Run ablation experiments on local machine

set -e

# Configuration
ABLATION_CONFIGS=(
    "experiments/configs/ablation/no_attnres.yaml"
    "experiments/configs/ablation/no_functional_embed.yaml"
    "experiments/configs/ablation/single_view.yaml"
)
DATASETS=("rice469" "maize282")
SEEDS=(42 123 456)
OUTPUT_BASE="experiments/results/ablation"

echo "=========================================="
echo "Running PlantHGNN Ablation Experiments"
echo "=========================================="

# Create output directory
mkdir -p $OUTPUT_BASE

# Run ablation experiments
for config in "${ABLATION_CONFIGS[@]}"; do
    config_name=$(basename $config .yaml)
    
    echo ""
    echo "Ablation: $config_name"
    echo "=========================================="
    
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Running: $dataset, seed=$seed"
            
            output_dir="$OUTPUT_BASE/$config_name/$dataset/seed_$seed"
            
            python experiments/run_experiment.py \
                --config $config \
                --dataset $dataset \
                --output-dir $output_dir \
                --seed $seed \
                --gpu 0
        done
    done
done

echo ""
echo "=========================================="
echo "All ablation experiments complete!"
echo "Results saved to: $OUTPUT_BASE"
echo "=========================================="
