#!/bin/bash
# Evaluate các models từ ablation test với OGB protocol

source ~/miniconda3/etc/profile.d/conda.sh
conda activate grail

RESULT_DIR="experiments/ablation_test_5k"

echo "=========================================="
echo "OGB EVALUATION - Ablation Models"
echo "=========================================="

# Các experiments cần evaluate
EXPERIMENTS=(
    "ablation_5k_baseline"
    "ablation_5k_rel_emb"
    "ablation_5k_dropout"
    "ablation_5k_rel_dropout"
    "ablation_5k_more_neg"
    "ablation_5k_paper_config"
    "ablation_5k_best_combined"
)

# Clear previous eval summary
> "${RESULT_DIR}/eval_summary.txt"
echo "OGB Evaluation Summary - $(date)" >> "${RESULT_DIR}/eval_summary.txt"
echo "========================================" >> "${RESULT_DIR}/eval_summary.txt"
echo "" >> "${RESULT_DIR}/eval_summary.txt"

for exp in "${EXPERIMENTS[@]}"; do
    exp_dir="experiments/${exp}"

    if [ -d "$exp_dir" ]; then
        echo ""
        echo ">>> Evaluating: $exp"

        # Chạy OGB evaluation (giới hạn 500 samples để test nhanh)
        python evaluate_grail_ogb.py \
            --experiment_name "$exp" \
            --max_samples 500 \
            2>&1 | tee "${exp_dir}/ogb_eval.log"

        # Extract MRR
        echo "=== $exp ===" >> "${RESULT_DIR}/eval_summary.txt"
        grep -E "MRR|Hits@" "${exp_dir}/ogb_eval.log" | tail -5 >> "${RESULT_DIR}/eval_summary.txt"
        echo "" >> "${RESULT_DIR}/eval_summary.txt"
    else
        echo ">>> Skipping $exp (not found)"
    fi
done

echo ""
echo "=========================================="
echo "EVALUATION COMPLETE!"
echo "=========================================="
echo ""
echo "Results:"
cat "${RESULT_DIR}/eval_summary.txt"
