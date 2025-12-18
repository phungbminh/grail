"""
Ablation Study Script for GraIL

Compares effectiveness of:
1. Relation embedding (r) in decoder: with vs without
2. Pooling strategies: mean vs query_attention
3. Attention heads: single-head vs multi-head

Usage:
    python ablation_study.py --dataset ogbl-biokg --db_path data/ogbl-biokg/100_tssp_hop2 --num_epochs 1
"""

import os
import argparse
import logging
import torch
import numpy as np
from datetime import datetime
from scipy.sparse import SparseEfficiencyWarning
from warnings import simplefilter

from subgraph_extraction.datasets import SubgraphDataset
from utils.initialization_utils import initialize_experiment
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl
from managers.evaluator import Evaluator
from managers.trainer import Trainer

# Import models
from model.dgl.graph_classifier import GraphClassifier
from model.dgl.graph_classifier_ablation import GraphClassifierAblation


def run_experiment(params, config_name, use_rel_emb=True, pool_type='mean', pool_heads=1):
    """Run a single experiment configuration."""

    logging.info(f"\n{'='*60}")
    logging.info(f"Running: {config_name}")
    logging.info(f"  - use_rel_emb: {use_rel_emb}")
    logging.info(f"  - pool_type: {pool_type}")
    logging.info(f"  - pool_heads: {pool_heads}")
    logging.info(f"{'='*60}")

    # Update params for this config
    params.pool_type = pool_type
    params.pool_heads = pool_heads
    params.use_rel_emb = use_rel_emb
    params.experiment_name = f"ablation_{config_name}"

    # Load datasets
    train = SubgraphDataset(params.db_path, 'train_pos', 'train_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.train_file)

    valid = SubgraphDataset(params.db_path, 'valid_pos', 'valid_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.valid_file)

    params.num_rels = train.num_rels
    params.aug_num_rels = train.aug_num_rels
    params.inp_dim = train.n_feat_dim
    params.max_label_value = train.max_n_label

    # Create relation2id from id2relation
    relation2id = {v: k for k, v in train.id2relation.items()}

    # Initialize model (use ablation version)
    model = GraphClassifierAblation(params, relation2id).to(params.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {num_params:,}")

    # Initialize evaluator and trainer
    valid_evaluator = Evaluator(params, model, valid)
    trainer = Trainer(params, model, train, valid_evaluator)

    # Train
    trainer.train()

    # Get final metrics
    result = valid_evaluator.eval()

    return {
        'config': config_name,
        'use_rel_emb': use_rel_emb,
        'pool_type': pool_type,
        'pool_heads': pool_heads,
        'num_params': num_params,
        'auc': result.get('auc', 0),
        'auc_pr': result.get('auc_pr', 0),
        'mrr': result.get('mrr', 0),
        'hits@1': result.get('hits@1', 0),
        'hits@3': result.get('hits@3', 0),
        'hits@10': result.get('hits@10', 0),
    }


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    results = []

    # Define ablation configurations
    configs = [
        # Baseline: mean pooling, with relation embedding
        ('baseline_mean_rel', True, 'mean', 1),

        # Ablation 1: mean pooling, WITHOUT relation embedding
        ('mean_no_rel', False, 'mean', 1),

        # Ablation 2: query_attention single-head, with relation embedding
        ('query_attn_1head_rel', True, 'query_attention', 1),

        # Ablation 3: query_attention single-head, WITHOUT relation embedding
        ('query_attn_1head_no_rel', False, 'query_attention', 1),

        # Ablation 4: query_attention multi-head (4), with relation embedding
        ('query_attn_4head_rel', True, 'query_attention', 4),

        # Ablation 5: query_attention multi-head (4), WITHOUT relation embedding
        ('query_attn_4head_no_rel', False, 'query_attention', 4),
    ]

    for config_name, use_rel_emb, pool_type, pool_heads in configs:
        try:
            result = run_experiment(params, config_name, use_rel_emb, pool_type, pool_heads)
            results.append(result)

            # Print intermediate result
            logging.info(f"\nResult for {config_name}:")
            logging.info(f"  AUC: {result['auc']:.4f}, AUC-PR: {result['auc_pr']:.4f}")
            if result.get('mrr', 0) > 0:
                logging.info(f"  MRR: {result['mrr']:.4f}, Hits@10: {result['hits@10']:.4f}")

        except Exception as e:
            logging.error(f"Error in {config_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary table
    print("\n" + "="*100)
    print("ABLATION STUDY RESULTS")
    print("="*100)
    print(f"{'Config':<30} {'Rel Emb':<10} {'Pool Type':<15} {'Heads':<8} {'Params':<12} {'AUC':<10} {'AUC-PR':<10}")
    print("-"*100)

    for r in results:
        print(f"{r['config']:<30} {str(r['use_rel_emb']):<10} {r['pool_type']:<15} {r['pool_heads']:<8} {r['num_params']:<12,} {r['auc']:<10.4f} {r['auc_pr']:<10.4f}")

    print("="*100)

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    # Effect of relation embedding
    rel_results = [r for r in results if r['use_rel_emb']]
    no_rel_results = [r for r in results if not r['use_rel_emb']]

    if rel_results and no_rel_results:
        avg_rel_auc = np.mean([r['auc'] for r in rel_results])
        avg_no_rel_auc = np.mean([r['auc'] for r in no_rel_results])
        print(f"\n1. Effect of Relation Embedding (r):")
        print(f"   - With rel_emb:    avg AUC = {avg_rel_auc:.4f}")
        print(f"   - Without rel_emb: avg AUC = {avg_no_rel_auc:.4f}")
        print(f"   - Improvement: {(avg_rel_auc - avg_no_rel_auc)*100:.2f}%")

    # Effect of pooling type
    mean_results = [r for r in results if r['pool_type'] == 'mean']
    query_results = [r for r in results if r['pool_type'] == 'query_attention']

    if mean_results and query_results:
        avg_mean_auc = np.mean([r['auc'] for r in mean_results])
        avg_query_auc = np.mean([r['auc'] for r in query_results])
        print(f"\n2. Effect of Pooling Type:")
        print(f"   - Mean pooling:     avg AUC = {avg_mean_auc:.4f}")
        print(f"   - Query attention:  avg AUC = {avg_query_auc:.4f}")
        print(f"   - Improvement: {(avg_query_auc - avg_mean_auc)*100:.2f}%")

    # Effect of multi-head
    single_head = [r for r in results if r['pool_type'] == 'query_attention' and r['pool_heads'] == 1]
    multi_head = [r for r in results if r['pool_type'] == 'query_attention' and r['pool_heads'] > 1]

    if single_head and multi_head:
        avg_single_auc = np.mean([r['auc'] for r in single_head])
        avg_multi_auc = np.mean([r['auc'] for r in multi_head])
        print(f"\n3. Effect of Multi-Head Attention:")
        print(f"   - Single-head (1):  avg AUC = {avg_single_auc:.4f}")
        print(f"   - Multi-head (4):   avg AUC = {avg_multi_auc:.4f}")
        print(f"   - Improvement: {(avg_multi_auc - avg_single_auc)*100:.2f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ablation_results_{timestamp}.txt"
    with open(results_file, 'w') as f:
        f.write("ABLATION STUDY RESULTS\n")
        f.write("="*100 + "\n")
        for r in results:
            f.write(f"{r}\n")
    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Ablation Study for GraIL')

    # Data params
    parser.add_argument("--dataset", "-d", type=str, default="ogbl-biokg")
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, default="train")
    parser.add_argument("--valid_file", type=str, default="valid")

    # Training params
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_every_iter", type=int, default=5000)

    # Model params
    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--num_gcn_layers", type=int, default=3)
    parser.add_argument("--num_bases", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--edge_dropout", type=float, default=0.5)
    parser.add_argument("--gnn_type", type=str, default='rgcn')
    parser.add_argument("--gnn_agg_type", type=str, default='sum')
    parser.add_argument("--pool_dropout", type=float, default=0.1)

    # Other params
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--disable_cuda', action='store_true')
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--max_nodes_per_hop", type=int, default=None)
    parser.add_argument("--num_neg_samples_per_link", type=int, default=1)
    parser.add_argument('--add_traspose_rels', type=bool, default=False)
    parser.add_argument('--enclosing_sub_graph', type=bool, default=True)
    parser.add_argument('--use_kge_embeddings', type=bool, default=False)
    parser.add_argument("--kge_model", type=str, default="TransE")
    parser.add_argument("--rel_emb_dim", type=int, default=32)
    parser.add_argument("--attn_rel_emb_dim", type=int, default=32)
    parser.add_argument('--has_attn', type=bool, default=True)
    parser.add_argument('--add_ht_emb', type=bool, default=True)
    parser.add_argument("--margin", type=float, default=10)
    parser.add_argument("--l2", type=float, default=5e-4)
    parser.add_argument("--clip", type=int, default=1000)
    parser.add_argument("--early_stop", type=int, default=100)

    params = parser.parse_args()

    # Initialize
    params.main_dir = os.path.dirname(os.path.abspath(__file__))
    params.experiment_name = "ablation_study"

    params.file_paths = {
        'train': os.path.join(params.main_dir, f'data/{params.dataset}/{params.train_file}.txt'),
        'valid': os.path.join(params.main_dir, f'data/{params.dataset}/{params.valid_file}.txt')
    }

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device(f'cuda:{params.gpu}')
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl

    main(params)
