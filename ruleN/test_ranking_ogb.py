import os
import sys
import random
import argparse
import logging
import json
import time
import multiprocessing as mp

# Add parent directory to path to import utils module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import dgl
import scipy.sparse as ssp
from tqdm import tqdm
import networkx as nx

from ogb.linkproppred import LinkPropPredDataset, Evaluator
from subgraph_extraction.graph_sampler import subgraph_extraction_labeling
from utils.graph_utils import incidence_matrix, remove_nodes, ssp_multigraph_to_dgl
from utils.dgl_utils import _bfs_relational
from utils.data_utils import process_files as process_files_original

# Wrapper to match the signature expected by test code
def process_files(files, saved_relation2id, add_traspose_rels):
    """Wrapper around utils.data_utils.process_files with transpose support"""
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files_original(files, saved_relation2id)

    # Add transpose relations if requested
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list = adj_list + adj_list_t

    # Convert to DGL
    dgl_adj_list = ssp_multigraph_to_dgl(adj_list)

    return adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity

def intialize_worker(model, adj_list, dgl_adj_list, params, node_features, kge_entity2id):
    global model_, adj_list_, dgl_adj_list_, params_, node_features_, kge_entity2id_
    model_, adj_list_, dgl_adj_list_, params_, node_features_, kge_entity2id_ = model, adj_list, dgl_adj_list, params, node_features, kge_entity2id

# Note: subgraph_extraction_labeling is imported from subgraph_extraction.graph_sampler
# We still need node_label here for prepare_features
def node_label(subgraph, max_distance=1):
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels
    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes

def prepare_features(subgraph, n_labels, max_n_label, n_feats=None):
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
    n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
    subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
    head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
    tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
    n_ids = np.zeros(n_nodes)
    n_ids[head_id] = 1
    n_ids[tail_id] = 2
    subgraph.ndata['id'] = torch.FloatTensor(n_ids)
    return subgraph

def get_subgraphs(links, adj_list, dgl_adj_list, max_node_label_value, params, node_features=None, kge_entity2id=None):
    """Extract subgraphs for a batch of links.

    Note: This function is a performance bottleneck. The main issue is that
    subgraph_extraction_labeling() recomputes the incidence matrix for each link,
    which is extremely wasteful. However, we cannot modify the imported function.

    TODO: Consider creating a custom optimized version that pre-computes incidence matrix.
    """
    subgraphs = []
    r_labels = []
    for link in links:
        head, tail, rel = link[0], link[1], link[2]
        # subgraph_extraction_labeling returns 5 values: nodes, labels, subgraph_size, enc_ratio, num_pruned_nodes
        # WARNING: This function recomputes incidence matrix each time (slow!)
        nodes, node_labels, _, _, _ = subgraph_extraction_labeling((head, tail), rel, adj_list, h=params.hop, enclosing_sub_graph=params.enclosing_sub_graph, max_node_label_value=max_node_label_value)

        # DGL 2.2.1: Use node_subgraph instead of deprecated subgraph API
        subgraph_nodes_tensor = torch.LongTensor(nodes)
        subgraph = dgl.node_subgraph(dgl_adj_list, subgraph_nodes_tensor)

        # Copy edge types from parent graph
        subgraph.edata['type'] = dgl_adj_list.edata['type'][subgraph.edata[dgl.EID]]
        subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        # Check if edge exists between roots (handle case where edge doesn't exist)
        try:
            # DGL 2.2.1: Use edge_ids (plural) instead of edge_id
            edges_btw_roots = subgraph.edge_ids(0, 1)
            rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)
            if rel_link.squeeze().nelement() == 0:
                subgraph.add_edges(0, 1)
                subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
                subgraph.edata['label'][-1] = torch.tensor(rel).type(torch.LongTensor)
        except:
            # Edge doesn't exist between node 0 and 1, add it
            subgraph.add_edges(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(rel).type(torch.LongTensor)

        # This part is simplified, assuming no kge_embeddings for now
        n_feats = None
        subgraph = prepare_features(subgraph, node_labels, max_node_label_value, n_feats)
        subgraphs.append(subgraph)
        r_labels.append(rel)
    batched_graph = dgl.batch(subgraphs)
    r_labels = torch.LongTensor(r_labels)
    return (batched_graph, r_labels)

def main(params):
    # Setup device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('=' * 80)
    logging.info(f'Device: {device}')

    # Load trained GraIL model
    logging.info(f'Loading trained GraIL model from: {params.experiment_name}')
    model_path = os.path.join('experiments', params.experiment_name, 'best_graph_classifier.pth')
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    logging.info(f'✓ Model loaded successfully')

    # Load graph structure for subgraph extraction
    # NOTE: This assumes the 'ogbl-biokg' data has been converted to .txt files in data/ogbl-biokg
    logging.info(f'Loading graph structure for subgraph extraction...')
    db_path = os.path.join('./data', params.dataset)
    train_path = os.path.join(db_path, 'train.txt')

    # We only need the training graph to build the adjacencies for subgraph extraction
    # NOTE: Must use 'train' key because utils/data_utils.process_files expects it
    files = {'train': train_path}
    adj_list, dgl_adj_list, _, _, _, _ = process_files(files, model.relation2id, params.add_traspose_rels)
    logging.info(f'✓ Graph structure loaded successfully')

    # Load OGB dataset and evaluator
    logging.info(f'Loading OGB dataset: {params.dataset}...')
    dataset = LinkPropPredDataset(name='ogbl-biokg', root=params.ogb_root)
    split_edge = dataset.get_edge_split()
    evaluator = Evaluator(name='ogbl-biokg')
    logging.info(f'✓ OGB dataset loaded successfully')

    # Select the split to evaluate
    if params.split == 'valid':
        split_data = split_edge['valid']
    elif params.split == 'test':
        split_data = split_edge['test']
    else:
        raise ValueError('Split not recognized: must be "valid" or "test"')

    # Optionally use only a subset of samples for faster evaluation
    total_samples = len(split_data['head'])
    if params.num_samples is not None and params.num_samples < total_samples:
        logging.info(f'Using subset of {params.num_samples:,} samples out of {total_samples:,} for faster evaluation')
        import numpy as np
        indices = np.random.choice(total_samples, params.num_samples, replace=False)
        split_data = {
            'head': split_data['head'][indices],
            'tail': split_data['tail'][indices],
            'relation': split_data['relation'][indices],
            'head_neg': split_data['head_neg'][indices],
            'tail_neg': split_data['tail_neg'][indices]
        }

    # Dataset statistics
    num_samples = len(split_data['head'])
    num_neg_per_sample = split_data['tail_neg'].shape[1]
    logging.info('=' * 80)
    logging.info(f'Evaluation Configuration:')
    logging.info(f'  Split: {params.split}')
    logging.info(f'  Number of samples: {num_samples:,}')
    logging.info(f'  Negative samples per link: {num_neg_per_sample}')
    logging.info(f'  Total predictions per mode: {num_samples * (1 + num_neg_per_sample):,}')
    logging.info(f'  Hop: {params.hop}, Max nodes per hop: {params.max_nodes_per_hop}')
    logging.info('=' * 80)

    # Batch size for evaluation - larger is faster but uses more memory
    # With GPU: can use 64-128, with CPU: use 16-32
    if params.eval_batch_size is not None:
        eval_batch_size = params.eval_batch_size
    else:
        eval_batch_size = 64 if device.type == 'cuda' else 16
    logging.info(f'  Eval batch size: {eval_batch_size}')

    # Evaluate both head-batch and tail-batch modes (like OGB baseline)
    modes = ['tail-batch', 'head-batch']
    all_pos_preds = []
    all_neg_preds = []

    for mode in modes:
        logging.info(f'\n>>> Starting {mode} evaluation...')
        pos_preds = []
        neg_preds = []

        # Get data for current mode
        if mode == 'tail-batch':
            # Predict tail: (head, relation, ?)
            source_nodes = split_data['head']
            target_nodes = split_data['tail']
            target_nodes_neg = split_data['tail_neg']
        else:  # head-batch
            # Predict head: (?, relation, tail)
            source_nodes = split_data['tail']  # Swap: tail becomes source
            target_nodes = split_data['head']  # Swap: head becomes target
            target_nodes_neg = split_data['head_neg']

        relation_types = split_data['relation']

        for i in tqdm(range(0, len(source_nodes), eval_batch_size), desc=f'{mode:12s}', ncols=100):
            batch_src = source_nodes[i:i+eval_batch_size]
            batch_rel = relation_types[i:i+eval_batch_size]
            batch_tgt = target_nodes[i:i+eval_batch_size]
            batch_tgt_neg = target_nodes_neg[i:i+eval_batch_size]

            # Create a list of all links to score in this batch
            all_links_to_score = []
            for j in range(len(batch_src)):
                # For tail-batch: (head, tail, rel)
                # For head-batch: (tail, head, rel) - will be swapped back in scoring
                if mode == 'tail-batch':
                    # Add positive link
                    all_links_to_score.append((batch_src[j], batch_tgt[j], batch_rel[j]))
                    # Add negative links (corrupted tails)
                    for neg_tail in batch_tgt_neg[j]:
                        all_links_to_score.append((batch_src[j], neg_tail, batch_rel[j]))
                else:  # head-batch
                    # Add positive link (swap back to original order)
                    all_links_to_score.append((batch_tgt[j], batch_src[j], batch_rel[j]))
                    # Add negative links (corrupted heads)
                    for neg_head in batch_tgt_neg[j]:
                        all_links_to_score.append((neg_head, batch_src[j], batch_rel[j]))

            # Get subgraphs and scores
            data, _ = get_subgraphs(all_links_to_score, adj_list, dgl_adj_list, model.gnn.max_label_value, params)

            # Move data to device (GPU if available)
            data = data.to(device)

            with torch.no_grad():
                scores = model(data)

            # Reshape scores and append to lists
            scores = scores.view(len(batch_src), -1)  # Shape: (eval_batch_size, 1 + num_neg)
            pos_preds.append(scores[:, 0])
            neg_preds.append(scores[:, 1:])

        # Concatenate predictions for this mode
        mode_pos = torch.cat(pos_preds, dim=0)
        mode_neg = torch.cat(neg_preds, dim=0)
        all_pos_preds.append(mode_pos)
        all_neg_preds.append(mode_neg)
        logging.info(f'✓ {mode} completed: {len(mode_pos):,} predictions generated')

    # Concatenate predictions from both modes (head-batch and tail-batch)
    logging.info('\n>>> Aggregating predictions from both modes...')
    pos_pred = torch.cat(all_pos_preds, dim=0)
    neg_pred = torch.cat(all_neg_preds, dim=0)

    logging.info(f'✓ Total predictions: {len(pos_pred):,}')
    logging.info(f'  - tail-batch: {len(all_pos_preds[0]):,} predictions')
    logging.info(f'  - head-batch: {len(all_pos_preds[1]):,} predictions')

    # Compute metrics
    logging.info('\n>>> Computing evaluation metrics...')
    input_dict = {
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    }

    result = evaluator.eval(input_dict)

    # Print results
    logging.info('=' * 80)
    logging.info(f'FINAL RESULTS ON {params.split.upper()} SPLIT:')
    logging.info('=' * 80)
    for key, value in result.items():
        logging.info(f'  {key:20s}: {value:.6f}')
    logging.info('=' * 80)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='OGB-style ranking evaluation for GraIL')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, required=True, help="Name of the experiment folder to load the model from.")
    parser.add_argument("--dataset", "-d", type=str, default="ogbl-biokg", help="Dataset string (should be ogbl-biokg).")
    parser.add_argument("--ogb_root", type=str, default="reference/ogb/examples/linkproppred/biokg/dataset", help="Root directory for OGB datasets.")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"], help="Which split to evaluate on.")
    
    # GraIL-specific params (should match the training configuration)
    parser.add_argument("--hop", type=int, default=3, help="Enclosing subgraph hop number.")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False, help='Whether to append adj matrix list with symmetric relations.')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True, help='whether to only consider enclosing subgraph')
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None, help="if > 0, upper bound the # nodes per hop by subsampling")

    # Evaluation optimization params
    parser.add_argument("--num_samples", "-n", type=int, default=None, help="Number of samples to evaluate (if None, use all samples). For faster testing, use e.g. 1000")
    parser.add_argument("--eval_batch_size", "-bs", type=int, default=None, help="Batch size for evaluation (if None, auto-set: 64 for GPU, 16 for CPU)")

    params = parser.parse_args()
    main(params)
