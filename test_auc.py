import os
import argparse
import logging
import time
import torch
from scipy.sparse import SparseEfficiencyWarning
import numpy as np

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl
from managers.evaluator import Evaluator

from warnings import simplefilter


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    # Step 1: Load model
    logging.info("=" * 50)
    logging.info("[Step 1/4] Loading model...")
    start_time = time.time()
    graph_classifier = initialize_model(params, None, load_model=True)
    logging.info(f"[Step 1/4] Model loaded in {time.time() - start_time:.2f}s")
    logging.info(f"  Device: {params.device}")

    all_auc = []
    auc_mean = 0
    all_auc_pr = []
    auc_pr_mean = 0

    for r in range(1, params.runs + 1):
        logging.info("=" * 50)
        logging.info(f"[Run {r}/{params.runs}]")

        # Step 2: Build cache path
        max_nodes_str = f"_maxnodes_{params.max_nodes_per_hop}" if params.max_nodes_per_hop else ""
        semantic_str = ""
        if hasattr(params, 'use_semantic_pruning') and params.use_semantic_pruning:
            semantic_str = f"_semantic_pruning_sr{params.stage1_ratio}_pw{params.path_weight}_sw{params.semantic_weight}_tss{params.target_subgraph_size}"

        params.db_path = os.path.join(params.main_dir, f'data/{params.dataset}/test_subgraphs_{params.experiment_name}_{params.constrained_neg_prob}_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}{max_nodes_str}{semantic_str}')

        # Step 2: Generate subgraph datasets
        logging.info("[Step 2/4] Generating test subgraphs...")
        start_time = time.time()
        generate_subgraph_datasets(params, splits=['test'],
                                   saved_relation2id=graph_classifier.relation2id,
                                   max_label_value=graph_classifier.gnn.max_label_value)
        logging.info(f"[Step 2/4] Subgraphs generated in {time.time() - start_time:.2f}s")

        # Step 3: Load test dataset
        logging.info("[Step 3/4] Loading test dataset...")
        start_time = time.time()
        test = SubgraphDataset(params.db_path, 'test_pos', 'test_neg', params.file_paths, graph_classifier.relation2id,
                               add_traspose_rels=params.add_traspose_rels,
                               num_neg_samples_per_link=params.num_neg_samples_per_link,
                               use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                               kge_model=params.kge_model, file_name=params.test_file)
        logging.info(f"[Step 3/4] Dataset loaded in {time.time() - start_time:.2f}s")
        logging.info(f"  Test samples: {len(test):,}")

        # Step 4: Evaluate
        logging.info("[Step 4/4] Evaluating...")
        start_time = time.time()
        test_evaluator = Evaluator(params, graph_classifier, test)
        result = test_evaluator.eval(save=True)
        logging.info(f"[Step 4/4] Evaluation completed in {time.time() - start_time:.2f}s")

        logging.info(f'\nTest Set Performance: {result}')
        all_auc.append(result['auc'])
        auc_mean = auc_mean + (result['auc'] - auc_mean) / r

        all_auc_pr.append(result['auc_pr'])
        auc_pr_mean = auc_pr_mean + (result['auc_pr'] - auc_pr_mean) / r

    # Final results
    logging.info("=" * 50)
    logging.info("FINAL RESULTS:")
    logging.info(f'  AUC:    {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}')
    logging.info(f'  AUC-PR: {np.mean(all_auc_pr):.4f} ± {np.std(all_auc_pr):.4f}')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str, default="Toy",
                        help="Dataset string")
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--test_file", "-t", type=str, default="test",
                        help="Name of file containing test triplets")
    parser.add_argument("--runs", type=int, default=1,
                        help="How many runs to perform for mean and std?")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')

    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=100000,
                        help="Set maximum number of links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=3,
                        help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--model_type', '-m', type=str, choices=['dgl'], default='dgl',
                        help='what format to store subgraphs in for model')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')

    # Graph pooling params
    parser.add_argument('--pool_type', '-pool', type=str, default='mean',
                        choices=['mean', 'sum', 'max', 'attention', 'query_attention'],
                        help='graph pooling strategy: mean, sum, max, attention (global), query_attention (head/tail conditioned)')
    parser.add_argument('--pool_heads', '-ph', type=int, default=1,
                        help='number of attention heads for attention pooling (default: 1, recommended: 4 for multi-head)')
    parser.add_argument('--pool_dropout', '-pd', type=float, default=0.0,
                        help='dropout rate for attention pooling (default: 0.0, recommended: 0.1-0.2 for attention)')

    # Semantic pruning params
    parser.add_argument('--use_semantic_pruning', '-sp', action='store_true',
                        help='enable Two-Stage Semantic Pruning for subgraph extraction')
    parser.add_argument('--semantic_embeddings_path', '-sep', type=str, default=None,
                        help='path to pre-trained semantic embeddings (e.g., TransE entity embeddings)')
    parser.add_argument('--stage1_ratio', '-sr', type=int, default=10,
                        help='ratio for Stage 1 pruning (target_M * stage1_ratio nodes kept after Stage 1)')
    parser.add_argument('--path_weight', '-pw', type=float, default=0.6,
                        help='weight for path length scores in Stage 2 (alpha)')
    parser.add_argument('--semantic_weight', '-sw', type=float, default=0.4,
                        help='weight for semantic similarity scores in Stage 2 (beta)')
    parser.add_argument('--target_subgraph_size', '-tss', type=int, default=1000,
                        help='target subgraph size after pruning (M)')

    params = parser.parse_args()
    initialize_experiment(params, __file__)

    params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'test': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.test_file))
    }

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl

    main(params)
