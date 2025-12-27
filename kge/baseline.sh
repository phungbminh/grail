#!/bin/bash
# Run all KGE models for fair comparison with GraIL (emb_dim=64)
# Usage: ./run_all_fair.sh

set -e

cd "$(dirname "$0")"

echo "=============================================="
echo "KGE Fair Comparison Benchmark (emb_dim=64)"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Create results directory
RESULTS_DIR="fair_comparison_results_v2"
mkdir -p $RESULTS_DIR
SUMMARY_FILE="$RESULTS_DIR/summary_v2.txt"

# Clear previous summary
> $SUMMARY_FILE

echo "Running 5 KGE models sequentially..."
echo "Each model: 100K steps, emb_dim=64"
echo ""

# Function to extract metrics from log
extract_metrics() {
    local model=$1
    local log_file=$2

    if [ -f "$log_file" ]; then
        mrr=$(grep "Test MRR" "$log_file" | tail -1 | awk '{print $NF}')
        mr=$(grep "Test MR" "$log_file" | tail -1 | awk '{print $NF}')
        hits1=$(grep "Test HITS@1" "$log_file" | tail -1 | awk '{print $NF}')
        hits3=$(grep "Test HITS@3" "$log_file" | tail -1 | awk '{print $NF}')
        hits10=$(grep "Test HITS@10" "$log_file" | tail -1 | awk '{print $NF}')

        echo "$model,$mrr,$mr,$hits1,$hits3,$hits10" >> $SUMMARY_FILE
        echo "  MRR: $mrr | MR: $mr | H@1: $hits1 | H@3: $hits3 | H@10: $hits10"
    else
        echo "$model,FAILED,,,," >> $SUMMARY_FILE
        echo "  FAILED - log file not found"
    fi
}

# Header for summary
echo "Model,MRR,MR,Hits@1,Hits@3,Hits@10" > $SUMMARY_FILE

# 1. TransE
echo "=============================================="
echo "[1/5] Training TransE..."
echo "=============================================="
./run.sh train TransE ogbl-biokg-full 0 transe_fair 1024 256 64 9.0 1.0 0.0001 100000 64 2>&1 | tee $RESULTS_DIR/transe.log
extract_metrics "TransE" "$RESULTS_DIR/transe.log"
echo ""

# 2. DistMult
echo "=============================================="
echo "[2/5] Training DistMult..."
echo "=============================================="
./run.sh train DistMult ogbl-biokg-full 0 distmult_fair 1024 256 64 200.0 1.0 0.0001 100000 64 2>&1 | tee $RESULTS_DIR/distmult.log
extract_metrics "DistMult" "$RESULTS_DIR/distmult.log"
echo ""

# 3. ComplEx
echo "=============================================="
echo "[3/5] Training ComplEx..."
echo "=============================================="
./run.sh train ComplEx ogbl-biokg-full 0 complex_fair 1024 256 64 200.0 1.0 0.0001 100000 64 --double_entity_embedding --double_relation_embedding 2>&1 | tee $RESULTS_DIR/complex.log
extract_metrics "ComplEx" "$RESULTS_DIR/complex.log"
echo ""

# 4. RotatE
echo "=============================================="
echo "[4/5] Training RotatE..."
echo "=============================================="
./run.sh train RotatE ogbl-biokg-full 0 rotate_fair 1024 256 64 9.0 1.0 0.0001 100000 64 --double_entity_embedding 2>&1 | tee $RESULTS_DIR/rotate.log
extract_metrics "RotatE" "$RESULTS_DIR/rotate.log"
echo ""

# 5. pRotatE
echo "=============================================="
echo "[5/5] Training pRotatE..."
echo "=============================================="
./run.sh train pRotatE ogbl-biokg-full 0 protate_fair 1024 256 64 9.0 1.0 0.0001 100000 64 2>&1 | tee $RESULTS_DIR/protate.log
extract_metrics "pRotatE" "$RESULTS_DIR/protate.log"
echo ""

# Print final summary
echo ""
echo "=============================================="
echo "FINAL SUMMARY - Fair Comparison (emb_dim=64)"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Configuration:"
echo "  - emb_dim: 64 (match GraIL)"
echo "  - max_steps: 100000 (~20 epochs)"
echo "  - batch_size: 1024"
echo "  - neg_samples: 256"
echo "  - lr: 0.0001"
echo ""

# Pretty print summary table
echo "Results:"
echo "------------------------------------------------------------------------"
printf "| %-10s | %-8s | %-10s | %-8s | %-8s | %-8s |\n" "Model" "MRR" "MR" "Hits@1" "Hits@3" "Hits@10"
echo "------------------------------------------------------------------------"

# Read summary file and print
tail -n +2 $SUMMARY_FILE | while IFS=',' read -r model mrr mr hits1 hits3 hits10; do
    printf "| %-10s | %-8s | %-10s | %-8s | %-8s | %-8s |\n" "$model" "$mrr" "$mr" "$hits1" "$hits3" "$hits10"
done

echo "------------------------------------------------------------------------"
echo ""
echo "Full results saved to: $RESULTS_DIR/"
echo "Summary CSV: $SUMMARY_FILE"
