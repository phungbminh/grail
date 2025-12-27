"""
Add Test Data to Existing LMDB
==============================

Script to add test_pos and test_neg databases to an existing LMDB.
This is needed when the original LMDB only has train/valid splits.

Usage:
    python scripts/add_test_to_lmdb.py \
        --db_path data/ogbl-biokg-full/200_hop2_new \
        --dataset ogbl-biokg-full \
        --hop 2 \
        --max_nodes_per_hop 200 \
        --use_semantic_pruning \
        --semantic_embeddings_path ./kge/1760606770.5893373/entity_embedding.npy \
        --target_subgraph_size 200 \
        --stage1_ratio 10 \
        --path_weight 0.6 \
        --semantic_weight 0.4

Author: GRAIL Evaluation
"""

import argparse
import os
import sys
import logging
import json
import time
import lmdb
import numpy as np
import torch

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subgraph_extraction.graph_sampler import links2subgraphs, sample_fair_neg, sample_mixed_neg
from utils.data_utils import process_files

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def check_existing_databases(db_path):
    """Check what databases exist in the LMDB"""
    env = lmdb.open(db_path, readonly=True, max_dbs=10, lock=False)

    common_db_names = ['train_pos', 'train_neg', 'valid_pos', 'valid_neg',
                       'test_pos', 'test_neg']

    existing = []
    for name in common_db_names:
        try:
            with env.begin() as txn:
                db = env.open_db(name.encode(), txn=txn)
                stat = txn.stat(db)
                existing.append((name, stat['entries']))
        except Exception:
            pass

    env.close()
    return existing


def main():
    parser = argparse.ArgumentParser(description='Add Test Data to Existing LMDB')

    # Required
    parser.add_argument('--db_path', type=str, required=True,
                        help='Path to existing LMDB')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='Dataset name')

    # Subgraph extraction params
    parser.add_argument('--hop', '-ho', type=int, default=2)
    parser.add_argument('--max_nodes_per_hop', '-max_h', type=int, default=200)
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True)

    # TSSP params
    parser.add_argument('--use_semantic_pruning', '-sp', action='store_true')
    parser.add_argument('--semantic_embeddings_path', '-sep', type=str, default=None)
    parser.add_argument('--target_subgraph_size', '-tss', type=int, default=200)
    parser.add_argument('--stage1_ratio', '-s1r', type=int, default=10)
    parser.add_argument('--path_weight', '-pw', type=float, default=0.6)
    parser.add_argument('--semantic_weight', '-sw', type=float, default=0.4)

    # Negative sampling
    parser.add_argument('--num_neg_samples_per_link', '-neg', type=int, default=1)
    parser.add_argument('--hard_negative_ratio', type=float, default=1.0,
                        help='Ratio of hard negatives (1.0 = all same type)')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0)
    parser.add_argument('--max_links', type=int, default=5000000)

    # Other
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--experiment_name', '-e', type=str, default=None,
                        help='Load relation2id from experiment')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("ADD TEST DATA TO EXISTING LMDB")
    print("="*60)

    # Check existing databases
    print(f"\n[STEP 1] Checking existing LMDB at: {args.db_path}")

    if not os.path.exists(args.db_path):
        print(f"  ERROR: LMDB not found at {args.db_path}")
        sys.exit(1)

    existing = check_existing_databases(args.db_path)
    print("  Existing databases:")
    for name, count in existing:
        print(f"    {name}: {count:,} entries")

    # Check if test already exists
    test_exists = any(name == 'test_pos' for name, _ in existing)
    if test_exists:
        print("\n  WARNING: test_pos already exists!")
        response = input("  Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("  Aborted.")
            sys.exit(0)

    # Setup params
    params = argparse.Namespace()
    for key, value in vars(args).items():
        setattr(params, key, value)

    params.main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(params.main_dir, 'data', args.dataset)

    params.file_paths = {
        'train': os.path.join(data_path, 'train.txt'),
        'valid': os.path.join(data_path, 'valid.txt'),
        'test': os.path.join(data_path, 'test.txt')
    }

    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        params.device = torch.device(f'cuda:{args.gpu}')
        logging.info(f"Using GPU: {args.gpu}")
    else:
        params.device = torch.device('cpu')
        logging.info("Using CPU")

    # Load relation2id from experiment or file
    print(f"\n[STEP 2] Loading relation2id...")
    saved_relation2id = None
    max_label_value = None

    if args.experiment_name:
        model_path = os.path.join('experiments', args.experiment_name, 'best_graph_classifier.pth')
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location='cpu')
            saved_relation2id = model.relation2id
            max_label_value = model.gnn.max_label_value
            print(f"  Loaded from experiment: {len(saved_relation2id)} relations, max_label={max_label_value}")

    if saved_relation2id is None:
        relation2id_path = os.path.join(data_path, 'relation2id.json')
        if os.path.exists(relation2id_path):
            with open(relation2id_path) as f:
                saved_relation2id = json.load(f)
            print(f"  Loaded from file: {len(saved_relation2id)} relations")

    # Load graph
    print(f"\n[STEP 3] Loading graph...")
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(
        params.file_paths, saved_relation2id
    )
    print(f"  Entities: {len(entity2id):,}")
    print(f"  Relations: {len(relation2id)}")
    print(f"  Test triplets: {len(triplets['test']):,}")

    # Sample negatives
    print(f"\n[STEP 4] Sampling negatives for test...")
    data_dir = os.path.join(params.main_dir, f'data/{args.dataset}')

    test_triplets = triplets['test']
    if args.max_links < len(test_triplets):
        perm = np.random.permutation(len(test_triplets))[:args.max_links]
        test_triplets = test_triplets[perm]
        print(f"  Sampled {len(test_triplets):,} test triplets")

    if args.hard_negative_ratio > 0:
        print(f"  Using mixed sampling: {args.hard_negative_ratio*100:.0f}% hard negatives")
        pos, neg = sample_mixed_neg(
            adj_list,
            test_triplets,
            data_dir,
            args.num_neg_samples_per_link,
            max_size=args.max_links,
            hard_negative_ratio=args.hard_negative_ratio,
            constrained_neg_prob=args.constrained_neg_prob
        )
    else:
        print("  Using random negative sampling")
        from subgraph_extraction.graph_sampler import sample_neg
        pos, neg = sample_neg(
            adj_list,
            test_triplets,
            args.num_neg_samples_per_link,
            max_size=args.max_links,
            constrained_neg_prob=args.constrained_neg_prob
        )

    print(f"  Positives: {len(pos):,}")
    print(f"  Negatives: {len(neg):,}")

    # Load semantic embeddings
    semantic_embeddings = None
    if args.use_semantic_pruning and args.semantic_embeddings_path:
        print(f"\n[STEP 5] Loading semantic embeddings...")
        semantic_embeddings = np.load(args.semantic_embeddings_path)
        print(f"  Shape: {semantic_embeddings.shape}")

        if torch.cuda.is_available() and args.gpu >= 0:
            semantic_embeddings = torch.from_numpy(semantic_embeddings).float().to(params.device)
            print(f"  Moved to GPU: {params.device}")

    # Prepare graphs dict for links2subgraphs
    graphs = {
        'test': {
            'pos': pos,
            'neg': neg,
            'triplets': test_triplets,
            'max_size': args.max_links
        }
    }

    # Extract subgraphs
    print(f"\n[STEP 6] Extracting subgraphs...")
    print(f"  LMDB path: {args.db_path}")
    print(f"  Hop: {args.hop}")
    print(f"  Max nodes per hop: {args.max_nodes_per_hop}")
    if args.use_semantic_pruning:
        print(f"  TSSP: stage1_ratio={args.stage1_ratio}, target_size={args.target_subgraph_size}")
        print(f"  Weights: path={args.path_weight}, semantic={args.semantic_weight}")

    start_time = time.time()

    # Call links2subgraphs (will add test_pos and test_neg to LMDB)
    links2subgraphs(adj_list, graphs, params, max_label_value, semantic_embeddings)

    elapsed = time.time() - start_time

    # Verify
    print(f"\n[STEP 7] Verifying...")
    new_existing = check_existing_databases(args.db_path)

    print("  Updated databases:")
    for name, count in new_existing:
        print(f"    {name}: {count:,} entries")

    test_pos_count = next((c for n, c in new_existing if n == 'test_pos'), 0)
    test_neg_count = next((c for n, c in new_existing if n == 'test_neg'), 0)

    if test_pos_count > 0 and test_neg_count > 0:
        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}")
        print(f"  test_pos: {test_pos_count:,} entries")
        print(f"  test_neg: {test_neg_count:,} entries")
        print(f"  Time: {elapsed/60:.1f} minutes")
    else:
        print(f"\n  ERROR: Failed to add test data!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("COMPLETED!")
    print(f"{'='*60}")
    print(f"\nNow you can run:")
    print(f"  python test_classification.py \\")
    print(f"      --grail_experiment <experiment_name> \\")
    print(f"      --db_path {args.db_path} \\")
    print(f"      --kge_path <kge_path>")


if __name__ == '__main__':
    main()
