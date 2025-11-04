#!/bin/bash
# Quick test script for OGB BioKG dataset (10K samples)
# Usage: bash train_biokg_quick.sh

echo "Quick training GraIL on ogbl-biokg (10K samples)..."

python train.py \
    --dataset ogbl-biokg \
    --experiment_name grail_biokg_quick \
    --num_epochs 10 \
    --eval_every 2 \
    --save_every 5 \
    --num_gcn_layers 2 \
    --emb_dim 32 \
    --num_bases 4 \
    --hop 2 \
    --max_links 10000 \
    --enclosing_sub_graph True \
    --num_neg_samples_per_link 1 \
    --batch_size 32 \
    --lr 0.001 \
    --optimizer Adam \
    --clip 1000 \
    --l2 5e-4 \
    --edge_dropout 0.3 \
    --gnn_agg_type sum \
    --add_traspose_rels True \
    --add_ht_emb True

echo "Quick training complete! Results saved in experiments/grail_biokg_quick/"
