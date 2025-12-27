#!/bin/bash
# Test ablation với max_links=5000 để so sánh nhanh các cấu hình

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grail

# Base settings
DATASET="ogbl-biokg-full"
MAX_LINKS=5000
HOP=2
MAX_NODES=200
BATCH_SIZE=32
GPU=0
EPOCHS=10

# Tạo thư mục kết quả
RESULT_DIR="experiments/ablation_test_5k"
mkdir -p $RESULT_DIR

echo "=========================================="
echo "ABLATION TEST - 5000 links, $EPOCHS epochs"
echo "=========================================="

# Function để train và extract metrics
run_experiment() {
    local exp_name=$1
    local extra_args=$2
    local exp_dir="experiments/ablation_5k_${exp_name}"

    echo ""
    echo "============================================"
    echo ">>> Running: $exp_name"
    echo ">>> Args: $extra_args"
    echo "============================================"
    echo ""

    # Skip nếu đã chạy xong
    if [ -f "${exp_dir}/log_train.txt" ] && grep -q "Best" "${exp_dir}/log_train.txt"; then
        echo ">>> $exp_name already completed, skipping..."
        # Extract metrics cho summary
        echo "=== $exp_name ===" >> "${RESULT_DIR}/summary.txt"
        grep -E "Best.*auc" "${exp_dir}/log_train.txt" | tail -1 >> "${RESULT_DIR}/summary.txt"
        echo "" >> "${RESULT_DIR}/summary.txt"
        return 0
    fi

    python train.py \
        -e "ablation_5k_${exp_name}" \
        -d $DATASET \
        --gpu $GPU \
        --max_links $MAX_LINKS \
        --hop $HOP \
        --max_nodes_per_hop $MAX_NODES \
        --batch_size $BATCH_SIZE \
        --num_epochs $EPOCHS \
        --eval_every 2 \
        $extra_args \
        2>&1 | tee "${RESULT_DIR}/${exp_name}.log"

    # Extract final metrics
    echo "=== $exp_name ===" >> "${RESULT_DIR}/summary.txt"
    grep -E "Best.*auc" "${exp_dir}/log_train.txt" | tail -1 >> "${RESULT_DIR}/summary.txt"
    echo "" >> "${RESULT_DIR}/summary.txt"
}

# Clear previous summary
> "${RESULT_DIR}/summary.txt"

echo "Starting ablation study at $(date)"
echo "" >> "${RESULT_DIR}/summary.txt"
echo "Ablation Study - $(date)" >> "${RESULT_DIR}/summary.txt"
echo "max_links=$MAX_LINKS, epochs=10" >> "${RESULT_DIR}/summary.txt"
echo "========================================" >> "${RESULT_DIR}/summary.txt"

# Case 1: Baseline (giống setting hiện tại của bạn)
run_experiment "baseline" ""

# Case 2: Với use_rel_emb (quan trọng nhất theo paper)
run_experiment "rel_emb" "--use_rel_emb"

# Case 3: Với edge_dropout=0.5 (paper setting)
run_experiment "dropout" "--edge_dropout 0.5"

# Case 4: use_rel_emb + edge_dropout
run_experiment "rel_dropout" "--use_rel_emb --edge_dropout 0.5"

# Case 5: Tăng num_neg_samples_per_link
run_experiment "more_neg" "--use_rel_emb --num_neg_samples_per_link 5"

# Case 6: Full paper config (emb_dim=32)
run_experiment "paper_config" "--use_rel_emb --edge_dropout 0.5 --emb_dim 32"

# Case 7: Best combined (dự đoán)
run_experiment "best_combined" "--use_rel_emb --edge_dropout 0.5 --num_neg_samples_per_link 5"

echo ""
echo "=========================================="
echo "ABLATION COMPLETE!"
echo "=========================================="
echo ""

# Tạo bảng kết quả đẹp
echo "RESULTS SUMMARY" | tee "${RESULT_DIR}/final_results.txt"
echo "===============" | tee -a "${RESULT_DIR}/final_results.txt"
echo "" | tee -a "${RESULT_DIR}/final_results.txt"
printf "%-20s | %8s | %8s\n" "Experiment" "AUC" "AUC-PR" | tee -a "${RESULT_DIR}/final_results.txt"
printf "%s\n" "------------------------------------------------" | tee -a "${RESULT_DIR}/final_results.txt"

for exp in baseline rel_emb dropout rel_dropout more_neg paper_config best_combined; do
    log_file="experiments/ablation_5k_${exp}/log_train.txt"
    if [ -f "$log_file" ]; then
        # Extract best metrics
        best_line=$(grep "Best" "$log_file" | tail -1)
        auc=$(echo "$best_line" | grep -oP "auc: \K[0-9.]+")
        auc_pr=$(echo "$best_line" | grep -oP "auc_pr: \K[0-9.]+")
        if [ -z "$auc" ]; then auc="N/A"; fi
        if [ -z "$auc_pr" ]; then auc_pr="N/A"; fi
        printf "%-20s | %8s | %8s\n" "$exp" "$auc" "$auc_pr" | tee -a "${RESULT_DIR}/final_results.txt"
    else
        printf "%-20s | %8s | %8s\n" "$exp" "FAILED" "FAILED" | tee -a "${RESULT_DIR}/final_results.txt"
    fi
done

echo "" | tee -a "${RESULT_DIR}/final_results.txt"
echo "Full logs: ${RESULT_DIR}/"
echo "Model checkpoints: experiments/ablation_5k_*/"
