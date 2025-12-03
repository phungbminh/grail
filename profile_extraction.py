"""
Profiling script to find bottleneck in subgraph extraction
"""
import os
import time
import logging
import argparse
import numpy as np
import torch
from subgraph_extraction.datasets import generate_subgraph_datasets
from utils.initialization_utils import initialize_experiment

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile subgraph extraction')

    # Required args
    parser.add_argument("--dataset", type=str, default="ogbl-biokg")
    parser.add_argument("--experiment_name", type=str, default="profile_test")
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--max_links", type=int, default=1000)  # Small sample for quick test
    parser.add_argument("--max_nodes_per_hop", type=int, default=1000)
    parser.add_argument("--enclosing_sub_graph", type=bool, default=True)
    parser.add_argument("--num_neg_samples_per_link", type=int, default=1)
    parser.add_argument("--constrained_neg_prob", type=float, default=0.0)
    parser.add_argument("--add_traspose_rels", type=bool, default=True)

    # Semantic pruning args
    parser.add_argument("--use_semantic_pruning", action='store_true')
    parser.add_argument("--semantic_embeddings_path", type=str, default=None)
    parser.add_argument("--stage1_ratio", type=int, default=10)
    parser.add_argument("--path_weight", type=float, default=0.6)
    parser.add_argument("--semantic_weight", type=float, default=0.4)
    parser.add_argument("--target_subgraph_size", type=int, default=1000)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--disable_cuda", action='store_true')

    params = parser.parse_args()
    initialize_experiment(params, __file__)

    # Setup device
    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device(f'cuda:{params.gpu}')
    else:
        params.device = torch.device('cpu')

    # Setup paths
    params.train_file = "train"
    params.valid_file = "valid"
    params.file_paths = {
        'train': os.path.join(params.main_dir, f'data/{params.dataset}/{params.train_file}.txt'),
        'valid': os.path.join(params.main_dir, f'data/{params.dataset}/{params.valid_file}.txt')
    }

    # Cache path
    max_nodes_str = f"_maxnodes_{params.max_nodes_per_hop}" if params.max_nodes_per_hop else ""
    semantic_str = ""
    if params.use_semantic_pruning:
        semantic_str = f"_semantic_pruning_sr{params.stage1_ratio}_pw{params.path_weight}_sw{params.semantic_weight}_tss{params.target_subgraph_size}"

    params.db_path = os.path.join(
        params.main_dir,
        f'data/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}{max_nodes_str}{semantic_str}_PROFILE'
    )

    # Remove old cache
    if os.path.exists(params.db_path):
        import shutil
        logging.info(f"Removing old cache: {params.db_path}")
        shutil.rmtree(params.db_path)

    print("\n" + "="*80)
    print("PROFILING SUBGRAPH EXTRACTION")
    print("="*80)
    print(f"Dataset: {params.dataset}")
    print(f"Max links: {params.max_links}")
    print(f"Hop: {params.hop}")
    print(f"Max nodes per hop: {params.max_nodes_per_hop}")
    print(f"Semantic pruning: {params.use_semantic_pruning}")
    if params.use_semantic_pruning:
        print(f"Embeddings: {params.semantic_embeddings_path}")
    print(f"Device: {params.device}")
    print("="*80 + "\n")

    # Profile extraction
    start_total = time.time()
    generate_subgraph_datasets(params, splits=['train', 'valid'])
    total_time = time.time() - start_total

    print("\n" + "="*80)
    print(f"TOTAL TIME: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print("="*80)
    print("\nCheck logs above for bottlenecks:")
    print("  1. Negative sampling time")
    print("  2. Extraction time")
    print("  3. Memory usage")
    print("  4. LMDB write time")
    print("="*80 + "\n")
