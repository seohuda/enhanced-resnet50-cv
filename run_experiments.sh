#!/bin/bash
set -e

SEEDS="42 123 456"
EPOCHS=200
OUTPUT_BASE="results"

echo "=== Running ablation experiments ==="
echo "Seeds: $SEEDS | Epochs: $EPOCHS"
echo ""

run_config() {
    local label="$1"
    local out_dir="$2"
    shift 2

    if [ -f "${out_dir}/summary.json" ]; then
        echo "${label}: already complete, skipping (${out_dir}/summary.json found)"
        return
    fi

    local resume_args=()
    if [ -f "${out_dir}/last.pth" ]; then
        echo "${label}: resuming from ${out_dir}/last.pth"
        resume_args=(--resume "${out_dir}/last.pth")
    else
        echo "${label}: starting fresh"
    fi

    python3 train.py "$@" --output-dir "$out_dir" "${resume_args[@]}"
}

for SEED in $SEEDS; do
    echo "--- Seed: $SEED ---"

    run_config "[1/4] Baseline ResNet50" "${OUTPUT_BASE}/baseline_seed${SEED}" \
        --model resnet50 --epochs $EPOCHS --seed $SEED --amp

    run_config "[2/4] ResNet50 + CutMix" "${OUTPUT_BASE}/cutmix_seed${SEED}" \
        --model resnet50 --cutmix --epochs $EPOCHS --seed $SEED --amp

    run_config "[3/4] SE-ResNet50" "${OUTPUT_BASE}/se_seed${SEED}" \
        --model se_resnet50 --epochs $EPOCHS --seed $SEED --amp

    run_config "[4/4] SE-ResNet50 + CutMix" "${OUTPUT_BASE}/se_cutmix_seed${SEED}" \
        --model se_resnet50 --cutmix --epochs $EPOCHS --seed $SEED --amp

    echo ""
done

echo "=== All experiments complete ==="
echo "Run 'python3 plot_results.py --results-dir ${OUTPUT_BASE}' to generate comparison plots."
