"""
Update Negative Samples in Existing LMDB
========================================

Script này cập nhật CHỈ negative subgraphs trong LMDB hiện tại
với hard negative sampling (same entity type).

Usage:
    python update_negatives.py \
        --db_path data/ogbl-biokg-full/200_hop2_new \
        --hard_negative_ratio 0.5 \
        --num_workers 8

Author: GRAIL Optimization
"""

import os
import sys
import argparse
import logging
import lmdb
import struct
import time
import json
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
from argparse import Namespace

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.graph_utils import deserialize, serialize, ssp_multigraph_to_dgl, incidence_matrix
from utils.data_utils import process_files
from subgraph_extraction.graph_sampler import sample_mixed_neg, load_entity_to_type


def parse_args():
    parser = argparse.ArgumentParser(description='Update negative samples in LMDB')
    parser.add_argument('--db_path', type=str, required=True,
                        help='Path to existing LMDB database')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (default: parent of db_path)')
    parser.add_argument('--hard_negative_ratio', type=float, default=0.5,
                        help='Ratio of hard negatives (0-1, default: 0.5)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for subgraph extraction')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Batch size for LMDB writes')
    parser.add_argument('--splits', type=str, default='train',
                        help='Splits to update (comma-separated, e.g., train,valid)')
    parser.add_argument('--backup', action='store_true',
                        help='Backup original negatives before updating')
    parser.add_argument('--hop', type=int, default=2,
                        help='Number of hops for subgraph extraction')
    parser.add_argument('--max_nodes_per_hop', type=int, default=200,
                        help='Maximum nodes per hop')
    # Semantic pruning options
    parser.add_argument('--use_semantic_pruning', action='store_true', default=True,
                        help='Use semantic pruning for subgraph extraction (default: True)')
    parser.add_argument('--semantic_embeddings_path', type=str,
                        default='./kge/1760606770.5893373/entity_embedding.npy',
                        help='Path to semantic embeddings file')
    parser.add_argument('--target_subgraph_size', type=int, default=100,
                        help='Target subgraph size after pruning')
    parser.add_argument('--stage1_ratio', type=int, default=10,
                        help='Stage 1 pruning ratio (keep target_size * stage1_ratio nodes)')
    parser.add_argument('--path_weight', type=float, default=0.6,
                        help='Weight for path-based scoring (alpha)')
    parser.add_argument('--semantic_weight', type=float, default=0.4,
                        help='Weight for semantic scoring (beta)')
    return parser.parse_args()


def load_positive_triplets_from_lmdb(db_path, split='train'):
    """Load positive triplets from existing LMDB"""
    logging.info(f"Loading positive triplets from {db_path}/{split}_pos...")

    env = lmdb.open(db_path, readonly=True, max_dbs=6, lock=False)
    db_pos = env.open_db(f'{split}_pos'.encode())

    triplets = []

    with env.begin(db=db_pos) as txn:
        num_graphs = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        logging.info(f"Found {num_graphs:,} positive graphs")

        for idx in tqdm(range(num_graphs), desc="Loading positives"):
            str_id = '{:08}'.format(idx).encode('ascii')
            data = deserialize(txn.get(str_id))
            if data is not None:
                # Extract triplet: (head, tail, relation)
                nodes = data['nodes']
                r_label = data['r_label']
                if len(nodes) >= 2:
                    triplets.append([nodes[0], nodes[1], r_label])

    env.close()
    logging.info(f"Loaded {len(triplets):,} triplets")
    return np.array(triplets)


def load_graph_structure(data_dir):
    """Load graph adjacency matrices"""
    logging.info(f"Loading graph structure from {data_dir}...")

    file_paths = {
        'train': os.path.join(data_dir, 'train.txt'),
        'valid': os.path.join(data_dir, 'valid.txt'),
        'test': os.path.join(data_dir, 'test.txt')
    }

    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(file_paths, None)

    logging.info(f"Entities: {len(entity2id):,}, Relations: {len(relation2id)}")

    return adj_list, entity2id, relation2id, id2entity, id2relation


# Global variables for worker processes
_worker_data = {}


def init_worker(adj_list, params, max_label_value, A_incidence, semantic_embeddings=None):
    """Initialize worker with shared data"""
    global _worker_data
    _worker_data['adj_list'] = adj_list
    _worker_data['params'] = params
    _worker_data['max_label_value'] = max_label_value
    _worker_data['A_incidence'] = A_incidence
    _worker_data['semantic_embeddings'] = semantic_embeddings


def extract_subgraph_worker(args):
    """Extract subgraph for a single negative triplet with semantic pruning"""
    idx, (head, tail, rel) = args

    adj_list = _worker_data['adj_list']
    params = _worker_data['params']
    max_label_value = _worker_data['max_label_value']
    A_incidence = _worker_data['A_incidence']
    semantic_embeddings = _worker_data['semantic_embeddings']

    try:
        # Check if we should use FAST semantic extraction
        use_semantic = getattr(params, 'use_semantic_pruning', False) and semantic_embeddings is not None

        if use_semantic:
            # Use FAST semantic pruning extraction
            from subgraph_extraction.semantic_pruning import fast_subgraph_extraction

            nodes, n_labels, subgraph_size, enc_ratio, num_pruned = fast_subgraph_extraction(
                u=head, v=tail, r_label=rel,
                A_list=adj_list,
                A_incidence=A_incidence,
                h=params.hop,
                max_nodes_per_hop=params.max_nodes_per_hop,
                target_size=params.target_subgraph_size,
                stage1_ratio=params.stage1_ratio,
                embeddings=semantic_embeddings,
                alpha=params.path_weight,
                beta=params.semantic_weight,
                enclosing=params.enclosing_sub_graph
            )
        else:
            # Standard extraction without semantic pruning
            from subgraph_extraction.graph_sampler import subgraph_extraction_labeling

            nodes, n_labels, subgraph_size, enc_ratio, num_pruned = subgraph_extraction_labeling(
                (head, tail), rel, adj_list,
                params.hop,
                params.enclosing_sub_graph,
                params.max_nodes_per_hop,
                A_incidence=A_incidence
            )

        if max_label_value is not None:
            n_labels = np.array([np.minimum(label, max_label_value).tolist() for label in n_labels])

        datum = {
            'nodes': nodes,
            'r_label': rel,
            'g_label': 0,  # Negative label
            'n_labels': n_labels,
            'subgraph_size': subgraph_size,
            'enc_ratio': enc_ratio,
            'num_pruned_nodes': num_pruned
        }

        str_id = '{:08}'.format(idx).encode('ascii')
        serialized = serialize(datum)

        return str_id, serialized, subgraph_size

    except Exception as e:
        logging.warning(f"Error extracting subgraph {idx}: {e}")
        return None


def update_negatives_in_lmdb(db_path, neg_triplets, adj_list, params, max_label_value,
                              num_workers=8, batch_size=10000, split='train',
                              semantic_embeddings=None):
    """Update negative subgraphs in LMDB with semantic pruning support"""

    logging.info(f"Updating {len(neg_triplets):,} negative subgraphs...")

    # Pre-compute incidence matrix
    logging.info("Pre-computing incidence matrix...")
    A_incidence = incidence_matrix(adj_list)
    A_incidence = A_incidence + A_incidence.T
    if hasattr(A_incidence, 'tocsr'):
        A_incidence = A_incidence.tocsr()

    # Log semantic pruning status
    if semantic_embeddings is not None:
        logging.info(f"Semantic pruning ENABLED (embeddings shape: {semantic_embeddings.shape})")
        logging.info(f"  - Target size: {params.target_subgraph_size}")
        logging.info(f"  - Stage1 ratio: {params.stage1_ratio}")
        logging.info(f"  - Path weight: {params.path_weight}, Semantic weight: {params.semantic_weight}")
    else:
        logging.info("Semantic pruning DISABLED")

    # Open LMDB for writing
    env = lmdb.open(db_path, max_dbs=6, map_size=50 * 1024**3)  # 50GB
    db_neg = env.open_db(f'{split}_neg'.encode())

    # Update num_graphs
    with env.begin(write=True, db=db_neg) as txn:
        txn.put('num_graphs'.encode(), len(neg_triplets).to_bytes(
            int.bit_length(len(neg_triplets)), byteorder='little'))

    # Prepare arguments
    args_list = [(idx, (neg_triplets[idx][0], neg_triplets[idx][1], neg_triplets[idx][2]))
                 for idx in range(len(neg_triplets))]

    # Extract subgraphs with multiprocessing
    logging.info(f"Extracting subgraphs with {num_workers} workers...")

    results = []
    total_written = 0

    with mp.Pool(processes=num_workers,
                 initializer=init_worker,
                 initargs=(adj_list, params, max_label_value, A_incidence, semantic_embeddings)) as pool:

        for result in tqdm(pool.imap_unordered(extract_subgraph_worker, args_list, chunksize=64),
                          total=len(args_list), desc="Extracting negatives"):
            if result is not None:
                results.append(result)

            # Batch write to LMDB
            if len(results) >= batch_size:
                with env.begin(write=True, db=db_neg) as txn:
                    for str_id, serialized, _ in results:
                        txn.put(str_id, serialized)
                total_written += len(results)
                logging.info(f"Written {total_written:,}/{len(neg_triplets):,} subgraphs")
                results.clear()

    # Write remaining
    if results:
        with env.begin(write=True, db=db_neg) as txn:
            for str_id, serialized, _ in results:
                txn.put(str_id, serialized)
        total_written += len(results)

    env.close()
    logging.info(f"Completed! Written {total_written:,} negative subgraphs")


def main():
    args = parse_args()

    print("=" * 70)
    print("UPDATE NEGATIVE SAMPLES IN LMDB")
    print("=" * 70)
    print(f"DB Path: {args.db_path}")
    print(f"Hard Negative Ratio: {args.hard_negative_ratio * 100:.0f}%")
    print(f"Semantic Pruning: {'ENABLED' if args.use_semantic_pruning else 'DISABLED'}")
    print(f"Splits: {args.splits}")
    print("=" * 70)

    # Determine data directory
    if args.data_dir is None:
        args.data_dir = os.path.dirname(args.db_path)

    # Check entity_to_type.json exists
    entity_type_path = os.path.join(args.data_dir, 'entity_to_type.json')
    if not os.path.exists(entity_type_path):
        logging.error(f"entity_to_type.json not found at {entity_type_path}")
        logging.error("Hard negative sampling requires entity type mapping!")
        sys.exit(1)

    # Load semantic embeddings if enabled
    semantic_embeddings = None
    if args.use_semantic_pruning:
        if os.path.exists(args.semantic_embeddings_path):
            logging.info(f"Loading semantic embeddings from {args.semantic_embeddings_path}...")
            semantic_embeddings = np.load(args.semantic_embeddings_path)
            logging.info(f"Semantic embeddings shape: {semantic_embeddings.shape}")
        else:
            logging.warning(f"Semantic embeddings not found at {args.semantic_embeddings_path}")
            logging.warning("Falling back to path-based pruning only")

    # Load graph structure
    adj_list, entity2id, relation2id, id2entity, id2relation = load_graph_structure(args.data_dir)

    # Get max_label_value from existing LMDB
    env = lmdb.open(args.db_path, readonly=True, max_dbs=6, lock=False)
    with env.begin() as txn:
        max_n_label_sub = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
        max_n_label_obj = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')
    env.close()
    max_label_value = np.array([max_n_label_sub, max_n_label_obj])
    logging.info(f"Max label value: {max_label_value}")

    # Create params for subgraph extraction
    params = Namespace(
        hop=args.hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        enclosing_sub_graph=True,
        use_semantic_pruning=args.use_semantic_pruning,
        target_subgraph_size=args.target_subgraph_size,
        stage1_ratio=args.stage1_ratio,
        path_weight=args.path_weight,
        semantic_weight=args.semantic_weight
    )

    # Process each split
    splits = [s.strip() for s in args.splits.split(',')]

    for split in splits:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing split: {split}")
        logging.info(f"{'='*60}")

        # Load positive triplets from LMDB
        pos_triplets = load_positive_triplets_from_lmdb(args.db_path, split)

        if len(pos_triplets) == 0:
            logging.warning(f"No positive triplets found for split '{split}', skipping...")
            continue

        # Backup original negatives if requested
        if args.backup:
            backup_path = f"{args.db_path}_backup_{split}_neg"
            logging.info(f"Backing up original negatives to {backup_path}")
            # TODO: Implement backup logic

        # Generate new negative samples with hard negative ratio
        logging.info(f"Generating new negatives with {args.hard_negative_ratio*100:.0f}% hard ratio...")

        pos_edges, neg_edges = sample_mixed_neg(
            adj_list=adj_list,
            edges=pos_triplets,
            data_dir=args.data_dir,
            num_neg_samples_per_link=1,
            max_size=len(pos_triplets),
            hard_negative_ratio=args.hard_negative_ratio
        )

        logging.info(f"Generated {len(neg_edges):,} negative triplets")

        # Update LMDB with new negative subgraphs
        update_negatives_in_lmdb(
            db_path=args.db_path,
            neg_triplets=neg_edges,
            adj_list=adj_list,
            params=params,
            max_label_value=max_label_value,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            split=split,
            semantic_embeddings=semantic_embeddings
        )

    print("\n" + "=" * 70)
    print("UPDATE COMPLETED!")
    print("=" * 70)
    print(f"Updated: {args.db_path}")
    print(f"Hard ratio: {args.hard_negative_ratio * 100:.0f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
