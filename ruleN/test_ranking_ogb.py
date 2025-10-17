import os
import random
import argparse
import logging
import json
import time
import multiprocessing as mp

import torch
import numpy as np
import dgl
import scipy.sparse as ssp
from tqdm import tqdm
import networkx as nx

from ogb.linkproppred import LinkPropPredDataset, Evaluator
from utils.dgl_utils import _bfs_relational
from utils.graph_utils import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize

# Helper functions copied from test_ranking.py to make the script self-contained

def process_files(files, saved_relation2id, add_traspose_rels):
    entity2id = {}
    relation2id = saved_relation2id
    triplets = {}
    ent = 0
    for file_type, file_path in files.items():
        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]
        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if triplet[1] in saved_relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], saved_relation2id[triplet[1]]])
        triplets[file_type] = np.array(data)
    id2entity = {v: k for k, v in entity2id.items()}
    adj_list = []
    for i in range(len(saved_relation2id)):
        idx = np.argwhere(triplets['graph'][:, 2] == i)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['graph'][:, 0][idx].squeeze(1), triplets['graph'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list = adj_list + adj_list_t
    dgl_adj_list = ssp_multigraph_to_dgl(adj_list)
    return adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity

def ssp_multigraph_to_dgl(graph, n_feats=None):
    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    for rel, adj in enumerate(graph):
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)
    g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl.from_networkx(g_nx, edge_attrs=['type'])
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)
    return g_dgl

def intialize_worker(model, adj_list, dgl_adj_list, params, node_features, kge_entity2id):
    global model_, adj_list_, dgl_adj_list_, params_, node_features_, kge_entity2id_
    model_, adj_list_, dgl_adj_list_, params_, node_features_, kge_entity2id_ = model, adj_list, dgl_adj_list, params, node_features, kge_entity2id

def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)

def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, node_information=None, max_node_label_value=None):
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T
    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)
    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)
    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]
    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)
    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]
    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])
    return pruned_subgraph_nodes, pruned_labels

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

def get_subgraphs(links, adj_list, dgl_adj_list, max_node_label_value, node_features=None, kge_entity2id=None):
    subgraphs = []
    r_labels = []
    for link in links:
        head, tail, rel = link[0], link[1], link[2]
        nodes, node_labels = subgraph_extraction_labeling((head, tail), rel, adj_list, h=params_.hop, enclosing_sub_graph=params_.enclosing_sub_graph, max_node_label_value=max_node_label_value)
        subgraph = dgl.DGLGraph(dgl_adj_list.subgraph(nodes))
        subgraph.edata['type'] = dgl_adj_list.edata['type'][dgl_adj_list.subgraph(nodes).parent_eid]
        subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)
        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)
        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edge(0, 1)
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
    # Load trained GraIL model
    model_path = os.path.join('experiments', params.experiment_name, 'best_graph_classifier.pth')
    model = torch.load(model_path, map_location='cpu')
    model.eval()

    # Load graph structure for subgraph extraction
    # NOTE: This assumes the 'ogbl-biokg' data has been converted to .txt files in data/ogbl-biokg
    db_path = os.path.join('./data', params.dataset)
    train_path = os.path.join(db_path, 'train.txt')
    
    # We only need the training graph to build the adjacencies for subgraph extraction
    files = {'graph': train_path}
    adj_list, dgl_adj_list, _, _, _, _ = process_files(files, model.relation2id, params.add_traspose_rels)

    # Load OGB dataset and evaluator
    dataset = LinkPropPredDataset(name='ogbl-biokg', root=params.ogb_root)
    split_edge = dataset.get_edge_split()
    evaluator = Evaluator(name='ogbl-biokg')

    # Select the split to evaluate
    if params.split == 'valid':
        source_nodes = split_edge['valid']['head']
        relation_types = split_edge['valid']['relation']
        target_nodes = split_edge['valid']['tail']
        target_nodes_neg = split_edge['valid']['tail_neg']
    elif params.split == 'test':
        source_nodes = split_edge['test']['head']
        relation_types = split_edge['test']['relation']
        target_nodes = split_edge['test']['tail']
        target_nodes_neg = split_edge['test']['tail_neg']
    else:
        raise ValueError('Split not recognized: must be "valid" or "test"')

    pos_preds = []
    neg_preds = []

    # Initialize worker for multiprocessing
    intialize_worker(model, adj_list, dgl_adj_list, params, None, None)

    # Using a smaller batch size for evaluation to avoid memory issues
    eval_batch_size = 16 
    
    with mp.Pool(processes=None, initializer=intialize_worker, initargs=(model, adj_list, dgl_adj_list, params, None, None)) as p:
        for i in tqdm(range(0, len(source_nodes), eval_batch_size)):
            batch_src = source_nodes[i:i+eval_batch_size]
            batch_rel = relation_types[i:i+eval_batch_size]
            batch_tgt = target_nodes[i:i+eval_batch_size]
            batch_tgt_neg = target_nodes_neg[i:i+eval_batch_size]

            pos_links = np.array([batch_src, batch_tgt, batch_rel]).T
            
            # Create a list of all links to score in this batch
            all_links_to_score = []
            for j in range(len(batch_src)):
                # Add positive link
                all_links_to_score.append((batch_src[j], batch_tgt[j], batch_rel[j]))
                # Add negative links
                for neg_tail in batch_tgt_neg[j]:
                    all_links_to_score.append((batch_src[j], neg_tail, batch_rel[j]))

            # Get subgraphs and scores
            data, _ = get_subgraphs(all_links_to_score, adj_list, dgl_adj_list, model.gnn.max_label_value)
            with torch.no_grad():
                scores = model(data)

            # Reshape scores and append to lists
            scores = scores.view(len(batch_src), -1) # Shape: (eval_batch_size, 1 + num_neg) 
            pos_preds.append(scores[:, 0])
            neg_preds.append(scores[:, 1:])

    # Concatenate all predictions and evaluate
    pos_pred = torch.cat(pos_preds, dim=0)
    neg_pred = torch.cat(neg_preds, dim=0)
    
    input_dict = {
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    }
    
    result = evaluator.eval(input_dict)
    
    logging.info(f'Results on {params.split} split:')
    for key, value in result.items():
        logging.info(f'{key}: {value}')


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

    params = parser.parse_args()
    main(params)
