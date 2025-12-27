"""
Script to generate test LMDB cache for OGB evaluation.
Generates both head and tail negatives for fair comparison with OGB leaderboard.
Reuses existing functions from train.py and graph_sampler.py.
"""
import os
import argparse
import logging
import json
import torch
import numpy as np
import lmdb
from tqdm import tqdm

from subgraph_extraction.graph_sampler import links2subgraphs
from utils.data_utils import process_files


def load_ogb_negatives(data_dir, split='test'):
    """Load pre-generated negatives (already converted to global IDs)."""
    converted_path = os.path.join(data_dir, f'{split}_negatives.pt')

    if os.path.exists(converted_path):
        logging.info(f"Loading converted negatives from {converted_path}")
        data = torch.load(converted_path)
        logging.info(f"  Triplets: {len(data['head']):,}")
        logging.info(f"  head_neg shape: {data['head_neg'].shape}")
        logging.info(f"  tail_neg shape: {data['tail_neg'].shape}")
        return data

    raise FileNotFoundError(f"Converted negatives not found at {converted_path}")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Generate test LMDB cache with OGB negatives (head + tail)')

    # Required params
    parser.add_argument("--dataset", "-d", type=str, required=True)
    parser.add_argument("--experiment_name", "-e", type=str, default=None,
                        help="Experiment name (to load relation2id)")

    # Data params
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid"])
    parser.add_argument("--max_links", type=int, default=5000000,
                        help="Maximum number of test links")
    parser.add_argument("--num_neg_samples", type=int, default=500,
                        help="Number of negative samples per side (max 500)")

    # Subgraph extraction params
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--max_nodes_per_hop", type=int, default=500)
    parser.add_argument("--enclosing_sub_graph", type=bool, default=True)

    # Semantic pruning params (reuse from train.py)
    parser.add_argument('--use_semantic_pruning', '-sp', action='store_true')
    parser.add_argument('--semantic_embeddings_path', '-sep', type=str, default=None)
    parser.add_argument('--stage1_ratio', '-sr', type=int, default=10)
    parser.add_argument('--path_weight', '-pw', type=float, default=0.6)
    parser.add_argument('--semantic_weight', '-sw', type=float, default=0.4)
    parser.add_argument('--target_subgraph_size', '-tss', type=int, default=100)

    # GPU params
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--disable_cuda', action='store_true')

    # Output path
    parser.add_argument("--db_path", type=str, default=None)

    params = parser.parse_args()
    params.main_dir = os.path.dirname(os.path.abspath(__file__))

    # Set device
    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')
    logging.info(f"Using device: {params.device}")

    # Load relation2id
    data_dir = os.path.join(params.main_dir, f'data/{params.dataset}')
    relation2id_path = os.path.join(data_dir, 'relation2id.json')

    saved_relation2id = None
    max_label_value = None

    if params.experiment_name:
        model_path = os.path.join('experiments', params.experiment_name, 'best_graph_classifier.pth')
        if os.path.exists(model_path):
            logging.info(f"Loading relation2id from {model_path}")
            model = torch.load(model_path, map_location='cpu')
            saved_relation2id = model.relation2id
            max_label_value = model.gnn.max_label_value
            logging.info(f"Loaded {len(saved_relation2id)} relations, max_label={max_label_value}")

    if saved_relation2id is None and os.path.exists(relation2id_path):
        with open(relation2id_path) as f:
            saved_relation2id = json.load(f)
        logging.info(f"Loaded {len(saved_relation2id)} relations from {relation2id_path}")

    # Load OGB negatives
    logging.info("=" * 60)
    logging.info("Loading OGB pre-generated negatives...")
    ogb_data = load_ogb_negatives(data_dir, params.split)

    num_triplets = len(ogb_data['head'])
    if params.max_links < num_triplets:
        logging.info(f"Limiting to {params.max_links} triplets (from {num_triplets})")
        num_triplets = params.max_links

    num_neg = min(params.num_neg_samples, 500)
    logging.info(f"Using {num_neg} negatives per side (head + tail)")

    # Set file paths and load graph
    params.file_paths = {
        'train': os.path.join(data_dir, 'train.txt'),
        'valid': os.path.join(data_dir, 'valid.txt'),
        'test': os.path.join(data_dir, 'test.txt')
    }

    logging.info("Loading graph...")
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(
        params.file_paths, saved_relation2id
    )
    logging.info(f"  Entities: {len(entity2id):,}")
    logging.info(f"  Relations: {len(relation2id)}")

    # Setup LMDB path
    if params.db_path is None:
        semantic_str = ""
        if params.use_semantic_pruning:
            semantic_str = f"_sp_tss{params.target_subgraph_size}"

        params.db_path = os.path.join(
            data_dir,
            f'{params.split}_ogb_lmdb_hop{params.hop}_neg{num_neg}{semantic_str}'
        )

    logging.info(f"Output LMDB path: {params.db_path}")

    # Prepare positive and negative edges
    logging.info("=" * 60)
    logging.info("Preparing edges with OGB negatives...")

    pos_edges = []
    neg_head_edges = []
    neg_tail_edges = []

    for i in tqdm(range(num_triplets), desc="Preparing edges"):
        head = int(ogb_data['head'][i])
        rel = int(ogb_data['relation'][i])
        tail = int(ogb_data['tail'][i])

        head_negs = ogb_data['head_neg'][i][:num_neg]
        tail_negs = ogb_data['tail_neg'][i][:num_neg]

        # Positive edge
        pos_edges.append([head, tail, rel])

        # Negative head edges (replace head)
        for neg_head in head_negs:
            neg_head_edges.append([int(neg_head), tail, rel])

        # Negative tail edges (replace tail)
        for neg_tail in tail_negs:
            neg_tail_edges.append([head, int(neg_tail), rel])

    pos_edges = np.array(pos_edges)
    neg_head_edges = np.array(neg_head_edges)
    neg_tail_edges = np.array(neg_tail_edges)

    logging.info(f"  Positive edges: {len(pos_edges):,}")
    logging.info(f"  Negative head edges: {len(neg_head_edges):,}")
    logging.info(f"  Negative tail edges: {len(neg_tail_edges):,}")

    # Load semantic embeddings if needed
    semantic_embeddings = None
    if params.use_semantic_pruning and params.semantic_embeddings_path:
        logging.info(f"Loading semantic embeddings from {params.semantic_embeddings_path}")
        semantic_embeddings = np.load(params.semantic_embeddings_path)
        logging.info(f"Loaded embeddings: {semantic_embeddings.shape}")

        if torch.cuda.is_available() and not params.disable_cuda:
            semantic_embeddings = torch.from_numpy(semantic_embeddings).float().to(params.device)
            logging.info(f"Embeddings moved to {params.device}")

    # Set params needed by links2subgraphs
    params.num_neg_samples_per_link = num_neg
    params.constrained_neg_prob = 0

    # Generate subgraphs in 3 stages
    logging.info("=" * 60)
    total_subgraphs = len(pos_edges) + len(neg_head_edges) + len(neg_tail_edges)
    logging.info(f"Total subgraphs to generate: {total_subgraphs:,}")
    logging.info("=" * 60)

    import time
    start_time = time.time()

    # Stage 1: Positive subgraphs
    logging.info("")
    logging.info("=" * 60)
    logging.info(f"[1/3] Generating POSITIVE subgraphs ({len(pos_edges):,} triplets)...")
    logging.info("=" * 60)
    stage1_start = time.time()

    pos_db_path = params.db_path + "_pos"
    params_pos = argparse.Namespace(**vars(params))
    params_pos.db_path = pos_db_path
    params_pos.num_neg_samples_per_link = 0  # No negatives for positive-only

    graphs_pos = {
        params.split: {
            'pos': pos_edges,
            'neg': np.array([]).reshape(0, 3)  # Empty negatives
        }
    }
    links2subgraphs(adj_list, graphs_pos, params_pos, max_label_value, semantic_embeddings)

    stage1_time = time.time() - stage1_start
    logging.info(f"Stage 1 completed in {stage1_time/60:.1f} minutes")

    # Stage 2: Negative head subgraphs
    logging.info("")
    logging.info("=" * 60)
    logging.info(f"[2/3] Generating NEGATIVE HEAD subgraphs ({len(neg_head_edges):,} triplets)...")
    logging.info("=" * 60)
    stage2_start = time.time()

    neg_head_db_path = params.db_path + "_neg_head"
    params_neg_head = argparse.Namespace(**vars(params))
    params_neg_head.db_path = neg_head_db_path
    params_neg_head.num_neg_samples_per_link = 0

    # Treat neg_head as "positive" for extraction (we just want the subgraphs)
    graphs_neg_head = {
        params.split: {
            'pos': neg_head_edges,
            'neg': np.array([]).reshape(0, 3)
        }
    }
    links2subgraphs(adj_list, graphs_neg_head, params_neg_head, max_label_value, semantic_embeddings)

    stage2_time = time.time() - stage2_start
    logging.info(f"Stage 2 completed in {stage2_time/60:.1f} minutes")

    # Stage 3: Negative tail subgraphs
    logging.info("")
    logging.info("=" * 60)
    logging.info(f"[3/3] Generating NEGATIVE TAIL subgraphs ({len(neg_tail_edges):,} triplets)...")
    logging.info("=" * 60)
    stage3_start = time.time()

    neg_tail_db_path = params.db_path + "_neg_tail"
    params_neg_tail = argparse.Namespace(**vars(params))
    params_neg_tail.db_path = neg_tail_db_path
    params_neg_tail.num_neg_samples_per_link = 0

    graphs_neg_tail = {
        params.split: {
            'pos': neg_tail_edges,
            'neg': np.array([]).reshape(0, 3)
        }
    }
    links2subgraphs(adj_list, graphs_neg_tail, params_neg_tail, max_label_value, semantic_embeddings)

    stage3_time = time.time() - stage3_start
    logging.info(f"Stage 3 completed in {stage3_time/60:.1f} minutes")

    total_time = time.time() - start_time

    # Summary
    logging.info("")
    logging.info("=" * 60)
    logging.info("LMDB Generation Complete!")
    logging.info("=" * 60)
    logging.info(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    logging.info(f"  Stage 1 (Positive):     {stage1_time/60:.1f} min")
    logging.info(f"  Stage 2 (Neg Head):     {stage2_time/60:.1f} min")
    logging.info(f"  Stage 3 (Neg Tail):     {stage3_time/60:.1f} min")
    logging.info("")
    logging.info(f"Output files:")
    logging.info(f"  Positive LMDB:     {pos_db_path}")
    logging.info(f"  Negative Head LMDB: {neg_head_db_path}")
    logging.info(f"  Negative Tail LMDB: {neg_tail_db_path}")
    logging.info("")
    logging.info(f"Statistics:")
    logging.info(f"  Positive subgraphs:     {len(pos_edges):,}")
    logging.info(f"  Negative head subgraphs: {len(neg_head_edges):,}")
    logging.info(f"  Negative tail subgraphs: {len(neg_tail_edges):,}")
    logging.info(f"  Total subgraphs:        {total_subgraphs:,}")
    logging.info("=" * 60)
    logging.info("")
    logging.info("To evaluate GraIL with OGB protocol, use:")
    logging.info(f"  python evaluate_grail_ogb.py \\")
    logging.info(f"      --experiment_name <exp_name> \\")
    logging.info(f"      --dataset {params.dataset} \\")
    logging.info(f"      --db_path {params.db_path}")


if __name__ == '__main__':
    main()
