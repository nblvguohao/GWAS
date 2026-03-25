#!/bin/bash
# Run experiments on local machine (4060 8G VRAM)
# Suitable for small datasets: rice469, maize282

set -e

# Configuration
CONFIG="experiments/configs/base_config.yaml"
DATASETS=("rice469" "maize282")
SEEDS=(42 123 456)
OUTPUT_BASE="experiments/results/local"

echo "=========================================="
echo "Running PlantHGNN Local Experiments"
echo "=========================================="
echo "Config: $CONFIG"
echo "Datasets: ${DATASETS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "Output: $OUTPUT_BASE"
echo "=========================================="

# Create output directory
mkdir -p $OUTPUT_BASE

# Run experiments
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "Running: $dataset, seed=$seed"
        
        output_dir="$OUTPUT_BASE/$dataset/seed_$seed"
        
        python experiments/run_experiment.py \
            --config $CONFIG \
            --dataset $dataset \
            --output-dir $output_dir \
            --seed $seed \
            --gpu 0
        
        echo "Completed: $dataset, seed=$seed"
    done
done

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_BASE"
echo "=========================================="
