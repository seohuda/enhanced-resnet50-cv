#!/bin/bash
set -e

SEEDS="42 123 456"
EPOCHS=200
OUTPUT_BASE="results"

echo "=== Running ablation experiments ==="
echo "Seeds: $SEEDS | Epochs: $EPOCHS"
echo ""

for SEED in $SEEDS; do
    echo "--- Seed: $SEED ---"

    echo "[1/4] Baseline ResNet50"
    python train.py \
        --model resnet50 \
        --epochs $EPOCHS \
        --seed $SEED \
        --output-dir "${OUTPUT_BASE}/baseline_seed${SEED}"

    echo "[2/4] ResNet50 + CutMix"
    python train.py \
        --model resnet50 \
        --cutmix \
        --epochs $EPOCHS \
        --seed $SEED \
        --output-dir "${OUTPUT_BASE}/cutmix_seed${SEED}"

    echo "[3/4] SE-ResNet50"
    python train.py \
        --model se_resnet50 \
        --epochs $EPOCHS \
        --seed $SEED \
        --output-dir "${OUTPUT_BASE}/se_seed${SEED}"

    echo "[4/4] SE-ResNet50 + CutMix"
    python train.py \
        --model se_resnet50 \
        --cutmix \
        --epochs $EPOCHS \
        --seed $SEED \
        --output-dir "${OUTPUT_BASE}/se_cutmix_seed${SEED}"

    echo ""
done

echo "=== All experiments complete ==="
echo "Run 'python plot_results.py --results-dir ${OUTPUT_BASE}' to generate comparison plots."
