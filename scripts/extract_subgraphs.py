"""
Extract Subgraphs with TSSP (Two-Stage Semantic Pruning)
========================================================

Standalone script để extract subgraphs mà không cần train.

Usage:
    python scripts/extract_subgraphs.py \
        --dataset ogbl-biokg-inductive \
        --hop 2 \
        --max_nodes_per_hop 200 \
        --use_semantic_pruning \
        --target_subgraph_size 100 \
        --stage1_ratio 10 \
        --path_weight 0.6 \
        --semantic_weight 0.4 \
        --kge_model TransE

Author: GRAIL
"""

import argparse
import os
import sys
import logging

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(description='Extract Subgraphs with TSSP')

    # Dataset
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='Dataset name (e.g., ogbl-biokg-inductive)')
    parser.add_argument('--db_path', type=str, default=None,
                        help='Custom LMDB path (optional)')

    # Subgraph extraction
    parser.add_argument('--hop', '-ho', type=int, default=2,
                        help='Number of hops for subgraph extraction')
    parser.add_argument('--max_nodes_per_hop', '-max_h', type=int, default=200,
                        help='Max nodes sampled per hop (before pruning)')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='Use enclosing subgraph')

    # TSSP - Two-Stage Semantic Pruning
    parser.add_argument('--use_semantic_pruning', '-sp', action='store_true',
                        help='Enable TSSP')
    parser.add_argument('--target_subgraph_size', '-tss', type=int, default=100,
                        help='Target number of nodes after Stage 2 pruning')
    parser.add_argument('--stage1_ratio', '-s1r', type=int, default=10,
                        help='Stage 1 keeps stage1_ratio × target_subgraph_size nodes')
    parser.add_argument('--path_weight', '-pw', type=float, default=0.6,
                        help='Weight for structural/path score (alpha)')
    parser.add_argument('--semantic_weight', '-sw', type=float, default=0.4,
                        help='Weight for semantic score (beta)')
    parser.add_argument('--kge_model', type=str, default='TransE',
                        help='KGE model for semantic embeddings')
    parser.add_argument('--semantic_embeddings_path', '-sep', type=str, default=None,
                        help='Custom path to semantic embeddings')

    # Negative sampling
    parser.add_argument('--num_neg_samples_per_link', '-neg', type=int, default=1,
                        help='Number of negative samples per positive')
    parser.add_argument('--max_links', type=int, default=1000000,
                        help='Max links to process')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='Probability of constrained negative sampling')
    parser.add_argument('--hard_negative_ratio', type=float, default=0.0,
                        help='Ratio of hard negatives (same entity type)')

    # Splits
    parser.add_argument('--splits', type=str, default='train,valid,test',
                        help='Comma-separated splits to process')

    # Other
    parser.add_argument('--num_workers', '-nw', type=int, default=8,
                        help='Number of workers for parallel processing')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU to use (-1 for CPU)')

    args = parser.parse_args()

    # Print configuration
    print("\n" + "="*60)
    print("SUBGRAPH EXTRACTION WITH TSSP")
    print("="*60)
    print(f"\nDataset: {args.dataset}")
    print(f"Hop: {args.hop}")
    print(f"Max nodes per hop: {args.max_nodes_per_hop}")

    if args.use_semantic_pruning:
        print(f"\n[TSSP ENABLED]")
        print(f"  Target subgraph size: {args.target_subgraph_size}")
        print(f"  Stage 1 ratio: {args.stage1_ratio}")
        print(f"  Path weight (α): {args.path_weight}")
        print(f"  Semantic weight (β): {args.semantic_weight}")
        print(f"  KGE model: {args.kge_model}")
    else:
        print(f"\n[TSSP DISABLED] - Using random pruning")

    print("="*60 + "\n")

    # Setup params object
    from argparse import Namespace
    params = Namespace()

    # Copy args to params
    for key, value in vars(args).items():
        setattr(params, key, value)

    # Set paths
    params.main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(params.main_dir, 'data', args.dataset)

    params.file_paths = {
        'train': os.path.join(data_path, 'train.txt'),
        'valid': os.path.join(data_path, 'valid.txt'),
        'test': os.path.join(data_path, 'test.txt')
    }

    # Add required params that datasets.py expects
    params.test_file = 'test'
    params.train_file = 'train'
    params.valid_file = 'valid'

    # Set db_path if not provided
    if not args.db_path:
        max_nodes_str = f"_max{args.max_nodes_per_hop}" if args.max_nodes_per_hop else ""
        semantic_str = ""
        if args.use_semantic_pruning:
            semantic_str = f"_sp_sr{args.stage1_ratio}_tss{args.target_subgraph_size}"

        params.db_path = os.path.join(
            data_path,
            f'subgraphs_hop{args.hop}{max_nodes_str}{semantic_str}'
        )

    logging.info(f"LMDB path: {params.db_path}")

    # Check files exist
    for split, path in params.file_paths.items():
        if not os.path.exists(path):
            logging.error(f"File not found: {path}")
            sys.exit(1)
        else:
            logging.info(f"Found {split}: {path}")

    # Set device
    import torch
    if args.gpu >= 0 and torch.cuda.is_available():
        params.device = torch.device(f'cuda:{args.gpu}')
        logging.info(f"Using GPU: {args.gpu}")
    else:
        params.device = torch.device('cpu')
        logging.info("Using CPU")

    # Parse splits
    splits = [s.strip() for s in args.splits.split(',')]
    logging.info(f"Processing splits: {splits}")

    # Generate subgraphs
    from subgraph_extraction.datasets import generate_subgraph_datasets

    logging.info("\nStarting subgraph extraction...")
    generate_subgraph_datasets(params, splits=splits)

    print("\n" + "="*60)
    print("EXTRACTION COMPLETED!")
    print("="*60)

    # Print cache path
    max_nodes_str = f"_max{args.max_nodes_per_hop}" if args.max_nodes_per_hop else ""
    semantic_str = ""
    if args.use_semantic_pruning:
        semantic_str = f"_semantic_pruning_sr{args.stage1_ratio}_pw{args.path_weight}_sw{args.semantic_weight}_tss{args.target_subgraph_size}"

    cache_path = os.path.join(
        params.main_dir,
        f'data/{args.dataset}/subgraphs_en_{args.enclosing_sub_graph}_neg_{args.num_neg_samples_per_link}_hop_{args.hop}{max_nodes_str}{semantic_str}'
    )
    print(f"\nCache saved to: {cache_path}")
    print("="*60)


if __name__ == '__main__':
    main()
