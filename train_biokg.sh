#!/bin/bash
# Training script for OGB BioKG dataset
# Usage: bash train_biokg.sh

echo "Training GraIL on ogbl-biokg dataset..."

python train.py \
    --dataset ogbl-biokg \
    --experiment_name grail_biokg \
    --num_epochs 30 \
    --eval_every 5 \
    --save_every 5 \
    --num_gcn_layers 3 \
    --emb_dim 32 \
    --num_bases 4 \
    --hop 3 \
    --enclosing_sub_graph \
    --num_neg_samples_per_link 1 \
    --batch_size 16 \
    --lr 0.001 \
    --optimizer Adam \
    --clip 1000 \
    --l2 5e-4 \
    --edge_dropout 0.3 \
    --gnn_agg_type sum

echo "Training complete! Results saved in experiments/grail_biokg/"
