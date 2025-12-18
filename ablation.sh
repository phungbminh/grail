#!/bin/bash
# GraIL Ablation Study - Compare different configurations
# Usage: ./ablation.sh

set -e

echo "=============================================="
echo "GraIL Ablation Study"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Base parameters
DB_PATH="data/ogbl-biokg/100_tssp_hop2"
DATASET="ogbl-biokg"
BATCH_SIZE=32
NUM_WORKERS=12
EMB_DIM=64
HOP=2
MAX_NODES=500
SEMANTIC_EMB="./kge/1760606770.5893373/entity_embedding.npy"
STAGE1_RATIO=10
PATH_WEIGHT=0.6
SEMANTIC_WEIGHT=0.4
MAX_LINKS=5000000
LR=0.01
GNN_TYPE="compgcn"
COMP_FN="sub"
EDGE_DROPOUT=0.0
TARGET_SIZE=100
NUM_EPOCHS=3
EVAL_EVERY=50000  # Eval & save best model every 50K iterations

# Common arguments
COMMON_ARGS="--dataset $DATASET \
  --db_path $DB_PATH \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --emb_dim $EMB_DIM \
  --hop $HOP \
  --max_nodes_per_hop $MAX_NODES \
  --use_semantic_pruning \
  --semantic_embeddings_path $SEMANTIC_EMB \
  --stage1_ratio $STAGE1_RATIO \
  --path_weight $PATH_WEIGHT \
  --semantic_weight $SEMANTIC_WEIGHT \
  --max_links $MAX_LINKS \
  --num_neg_samples_per_link 1 \
  --lr $LR \
  --gnn_type $GNN_TYPE \
  --comp_fn $COMP_FN \
  --edge_dropout $EDGE_DROPOUT \
  --eval_every_iter $EVAL_EVERY \
  --target_subgraph_size $TARGET_SIZE \
  --num_epochs $NUM_EPOCHS"

echo "Configuration:"
echo "  - DB: $DB_PATH"
echo "  - emb_dim: $EMB_DIM"
echo "  - GNN: $GNN_TYPE"
echo "  - Epochs: $NUM_EPOCHS"
echo ""

# Create results directory
RESULTS_DIR="ablation_results"
mkdir -p $RESULTS_DIR

# ============================================
# Case 1: Query attention 4 heads, with rel_emb
# ============================================
# [ALREADY COMPLETED - SKIPPED]
# echo "=============================================="
# echo "[1/6] Query attention 4 heads, WITH rel_emb"
# echo "=============================================="
# python train.py \
#   --experiment_name ablation_v2_query_4h_rel \
#   $COMMON_ARGS \
#   --pool_type query_attention \
#   --pool_heads 4 \
#   --pool_dropout 0.1 \
#   --use_rel_emb \
#   2>&1 | tee $RESULTS_DIR/1_query_4h_rel.log

# ============================================
# Case 2: Query attention 4 heads, no rel_emb
# ============================================
echo "=============================================="
echo "[2/6] Query attention 4 heads, NO rel_emb"
echo "=============================================="
python train.py \
  --experiment_name ablation_v2_query_4h_no_rel \
  $COMMON_ARGS \
  --pool_type query_attention \
  --pool_heads 4 \
  --pool_dropout 0.1 \
  2>&1 | tee $RESULTS_DIR/2_query_4h_no_rel.log

# ============================================
# Case 3: Query attention 1 head, with rel_emb
# ============================================
echo "=============================================="
echo "[3/6] Query attention 1 head, WITH rel_emb"
echo "=============================================="
python train.py \
  --experiment_name ablation_v2_query_1h_rel \
  $COMMON_ARGS \
  --pool_type query_attention \
  --pool_heads 1 \
  --pool_dropout 0.1 \
  --use_rel_emb \
  2>&1 | tee $RESULTS_DIR/3_query_1h_rel.log

# ============================================
# Case 4: Query attention 1 head, no rel_emb
# ============================================
echo "=============================================="
echo "[4/6] Query attention 1 head, NO rel_emb"
echo "=============================================="
python train.py \
  --experiment_name ablation_v2_query_1h_no_rel \
  $COMMON_ARGS \
  --pool_type query_attention \
  --pool_heads 1 \
  --pool_dropout 0.1 \
  2>&1 | tee $RESULTS_DIR/4_query_1h_no_rel.log

# ============================================
# Case 5: Mean pooling, with rel_emb
# ============================================
echo "=============================================="
echo "[5/6] Mean pooling, WITH rel_emb"
echo "=============================================="
python train.py \
  --experiment_name ablation_v2_mean_rel \
  $COMMON_ARGS \
  --pool_type mean \
  --use_rel_emb \
  2>&1 | tee $RESULTS_DIR/5_mean_rel.log

# ============================================
# Case 6: Mean pooling, no rel_emb (baseline)
# ============================================
echo "=============================================="
echo "[6/6] Mean pooling, NO rel_emb (baseline)"
echo "=============================================="
python train.py \
  --experiment_name ablation_v2_mean_no_rel \
  $COMMON_ARGS \
  --pool_type mean \
  2>&1 | tee $RESULTS_DIR/6_mean_no_rel.log

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "ABLATION STUDY COMPLETED"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Experiments completed:"
echo "  1. mean_no_rel     - Mean pooling, no relation embedding"
echo "  2. mean_rel        - Mean pooling, with relation embedding"
echo "  3. query_1h_no_rel - Query attention 1 head, no rel_emb"
echo "  4. query_1h_rel    - Query attention 1 head, with rel_emb"
echo "  5. query_4h_no_rel - Query attention 4 heads, no rel_emb"
echo "  6. query_4h_rel    - Query attention 4 heads, with rel_emb"
echo ""
echo "Logs saved to: $RESULTS_DIR/"
echo ""
echo "To evaluate, run test_ranking.py for each experiment:"
echo "  python test_ranking.py --experiment_name ablation_v2_mean_no_rel ..."
