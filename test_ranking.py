import os
import random
import argparse
import logging
import json
import time

import multiprocessing as mp
import scipy.sparse as ssp
from tqdm import tqdm
from collections import deque
import networkx as nx
import torch
import numpy as np
import dgl

# Import from graph_sampler to reuse existing functions
from subgraph_extraction.graph_sampler import (
    subgraph_extraction_labeling,
    get_neighbor_nodes,
    node_label
)
from utils.graph_utils import incidence_matrix


def process_files(files, saved_relation2id, add_traspose_rels):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

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

            # Save the triplets corresponding to only the known relations
            if triplet[1] in saved_relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], saved_relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(saved_relation2id)):
        idx = np.argwhere(triplets['graph'][:, 2] == i)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['graph'][:, 0][idx].squeeze(1), triplets['graph'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    # Add transpose matrices to handle both directions of relations.
    adj_list_aug = adj_list
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t

    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

    return adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation


def intialize_worker(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, A_incidence_precomputed=None, semantic_embeddings=None):
    global model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_, A_incidence_, semantic_embeddings_
    model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_ = model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id

    # OPTIMIZATION: Use pre-computed A_incidence (saves ~1-2s per subgraph)
    if A_incidence_precomputed is not None:
        A_incidence_ = A_incidence_precomputed
    else:
        # Fallback: compute if not provided
        A_incidence_ = incidence_matrix(adj_list)
        A_incidence_ += A_incidence_.T
        if not isinstance(A_incidence_, ssp.csr_matrix):
            A_incidence_ = A_incidence_.tocsr()

    semantic_embeddings_ = semantic_embeddings


def get_neg_samples_replacing_head_tail(test_links, adj_list, num_samples=50):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    for i, (head, tail, rel) in enumerate(zip(heads, tails, rels)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        while len(neg_triplet['head'][0]) < num_samples:
            neg_head = head
            neg_tail = np.random.choice(n)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        while len(neg_triplet['tail'][0]) < num_samples:
            neg_head = np.random.choice(n)
            neg_tail = tail
            # neg_head, neg_tail, rel = np.random.choice(n), np.random.choice(n), np.random.choice(r)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_all(test_links, adj_list):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    print('sampling negative triplets...')
    for i, (head, tail, rel) in tqdm(enumerate(zip(heads, tails, rels)), total=len(heads)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        for neg_tail in range(n):
            neg_head = head

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        for neg_head in range(n):
            neg_tail = tail

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_from_ruleN(ruleN_pred_path, entity2id, saved_relation2id):
    with open(ruleN_pred_path) as f:
        pred_data = [line.split() for line in f.read().split('\n')[:-1]]

    neg_triplets = []
    for i in range(len(pred_data) // 3):
        neg_triplet = {'head': [[], 10000], 'tail': [[], 10000]}
        if pred_data[3 * i][1] in saved_relation2id:
            head, rel, tail = entity2id[pred_data[3 * i][0]], saved_relation2id[pred_data[3 * i][1]], entity2id[pred_data[3 * i][2]]
            for j, new_head in enumerate(pred_data[3 * i + 1][1::2]):
                neg_triplet['head'][0].append([entity2id[new_head], tail, rel])
                if entity2id[new_head] == head:
                    neg_triplet['head'][1] = j
            for j, new_tail in enumerate(pred_data[3 * i + 2][1::2]):
                neg_triplet['tail'][0].append([head, entity2id[new_tail], rel])
                if entity2id[new_tail] == tail:
                    neg_triplet['tail'][1] = j

            neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
            neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

            neg_triplets.append(neg_triplet)

    return neg_triplets



def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl.from_networkx(g_nx, edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def prepare_features(subgraph, n_labels, max_n_label, n_feats=None):
    # One hot encode the node label feature and concat to n_featsure
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
    n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
    subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

    head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
    tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
    n_ids = np.zeros(n_nodes)
    n_ids[head_id] = 1  # head
    n_ids[tail_id] = 2  # tail
    subgraph.ndata['id'] = torch.FloatTensor(n_ids)

    return subgraph


def get_subgraphs(all_links, adj_list, dgl_adj_list, max_node_label_value, id2entity, node_features=None, kge_entity2id=None):
    # dgl_adj_list = ssp_multigraph_to_dgl(adj_list)

    subgraphs = []
    r_labels = []

    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]
        # Use subgraph_extraction_labeling from graph_sampler (returns 5 values)
        nodes, node_labels, _, _, _ = subgraph_extraction_labeling(
            (head, tail), rel, adj_list,
            h=params_.hop,
            enclosing_sub_graph=params_.enclosing_sub_graph,
            max_nodes_per_hop=getattr(params_, 'max_nodes_per_hop', None),
            max_node_label_value=max_node_label_value,
            A_incidence=A_incidence_  # Use cached incidence matrix
        )

        subgraph = dgl.DGLGraph(dgl_adj_list.subgraph(nodes))
        subgraph.edata['type'] = dgl_adj_list.edata['type'][dgl_adj_list.subgraph(nodes).parent_eid]
        subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)

        if rel_link.squeeze().nelement() == 0:
            # subgraph.add_edge(0, 1, {'type': torch.tensor([rel]), 'label': torch.tensor([rel])})
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(rel).type(torch.LongTensor)

        kge_nodes = [kge_entity2id[id2entity[n]] for n in nodes] if kge_entity2id else None
        n_feats = node_features[kge_nodes] if node_features is not None else None
        subgraph = prepare_features(subgraph, node_labels, max_node_label_value, n_feats)

        subgraphs.append(subgraph)
        r_labels.append(rel)

    batched_graph = dgl.batch(subgraphs)
    r_labels = torch.LongTensor(r_labels)

    return (batched_graph, r_labels)


def get_rank(neg_links):
    """Compute rank for a single test link (CPU version for multiprocessing)"""
    head_neg_links = neg_links['head'][0]
    head_target_id = neg_links['head'][1]

    if head_target_id != 10000:
        data = get_subgraphs(head_neg_links, adj_list_, dgl_adj_list_, model_.gnn.max_label_value, id2entity_, node_features_, kge_entity2id_)
        with torch.no_grad():
            head_scores = model_(data).squeeze(1).detach().numpy()
        head_rank = np.argwhere(np.argsort(head_scores)[::-1] == head_target_id) + 1
    else:
        head_scores = np.array([])
        head_rank = 10000

    tail_neg_links = neg_links['tail'][0]
    tail_target_id = neg_links['tail'][1]

    if tail_target_id != 10000:
        data = get_subgraphs(tail_neg_links, adj_list_, dgl_adj_list_, model_.gnn.max_label_value, id2entity_, node_features_, kge_entity2id_)
        with torch.no_grad():
            tail_scores = model_(data).squeeze(1).detach().numpy()
        tail_rank = np.argwhere(np.argsort(tail_scores)[::-1] == tail_target_id) + 1
    else:
        tail_scores = np.array([])
        tail_rank = 10000

    return head_scores, head_rank, tail_scores, tail_rank


def get_rank_gpu(neg_links, model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, A_incidence, device):
    """Compute rank for a single test link (GPU version for single-thread)"""
    head_neg_links = neg_links['head'][0]
    head_target_id = neg_links['head'][1]

    if head_target_id != 10000:
        data = get_subgraphs_gpu(head_neg_links, adj_list, dgl_adj_list, model.gnn.max_label_value, id2entity, params, node_features, kge_entity2id, A_incidence, device)
        with torch.no_grad():
            head_scores = model(data).squeeze(1).cpu().numpy()
        head_rank = np.argwhere(np.argsort(head_scores)[::-1] == head_target_id) + 1
    else:
        head_scores = np.array([])
        head_rank = 10000

    tail_neg_links = neg_links['tail'][0]
    tail_target_id = neg_links['tail'][1]

    if tail_target_id != 10000:
        data = get_subgraphs_gpu(tail_neg_links, adj_list, dgl_adj_list, model.gnn.max_label_value, id2entity, params, node_features, kge_entity2id, A_incidence, device)
        with torch.no_grad():
            tail_scores = model(data).squeeze(1).cpu().numpy()
        tail_rank = np.argwhere(np.argsort(tail_scores)[::-1] == tail_target_id) + 1
    else:
        tail_scores = np.array([])
        tail_rank = 10000

    return head_scores, head_rank, tail_scores, tail_rank


def get_subgraphs_gpu(all_links, adj_list, dgl_adj_list, max_node_label_value, id2entity, params, node_features=None, kge_entity2id=None, A_incidence=None, device=None):
    """GPU version of get_subgraphs"""
    subgraphs = []
    r_labels = []

    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]
        # Use subgraph_extraction_labeling from graph_sampler (returns 5 values)
        nodes, node_labels, _, _, _ = subgraph_extraction_labeling(
            (head, tail), rel, adj_list,
            h=params.hop,
            enclosing_sub_graph=params.enclosing_sub_graph,
            max_nodes_per_hop=getattr(params, 'max_nodes_per_hop', None),
            max_node_label_value=max_node_label_value,
            A_incidence=A_incidence
        )

        subgraph = dgl.DGLGraph(dgl_adj_list.subgraph(nodes))
        subgraph.edata['type'] = dgl_adj_list.edata['type'][dgl_adj_list.subgraph(nodes).parent_eid]
        subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)

        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(rel).type(torch.LongTensor)

        kge_nodes = [kge_entity2id[id2entity[n]] for n in nodes] if kge_entity2id else None
        n_feats = node_features[kge_nodes] if node_features is not None else None
        subgraph = prepare_features(subgraph, node_labels, max_node_label_value, n_feats)

        subgraphs.append(subgraph)
        r_labels.append(rel)

    batched_graph = dgl.batch(subgraphs)
    r_labels = torch.LongTensor(r_labels)

    # Move to GPU if available
    if device is not None and device.type == 'cuda':
        batched_graph = batched_graph.to(device)
        r_labels = r_labels.to(device)

    return (batched_graph, r_labels)


def save_to_file(neg_triplets, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'ranking_head.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['head'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

    with open(os.path.join('./data', params.dataset, 'ranking_tail.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['tail'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


def save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'grail_ranking_head_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score in zip(neg_triplet['head'][0], all_head_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score)]) + '\n')

    with open(os.path.join('./data', params.dataset, 'grail_ranking_tail_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score in zip(neg_triplet['tail'][0], all_tail_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score)]) + '\n')


def save_score_to_file_from_ruleN(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'grail_ruleN_ranking_head_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score in zip(neg_triplet['head'][0], all_head_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score)]) + '\n')

    with open(os.path.join('./data', params.dataset, 'grail_ruleN_ranking_tail_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score in zip(neg_triplet['tail'][0], all_tail_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score)]) + '\n')


def get_kge_embeddings(dataset, kge_model):

    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


def main(params):
    # Step 1: Load model
    logger.info("=" * 50)
    logger.info("[Step 1/6] Loading model...")
    start_time = time.time()

    # Load model to GPU if available
    if params.device.type == 'cuda':
        logger.info(f"Loading model to GPU: {params.device}")
        model = torch.load(params.model_path, map_location=params.device)
        model = model.to(params.device)
    else:
        logger.info("Loading model to CPU")
        model = torch.load(params.model_path, map_location='cpu')

    model.eval()
    logger.info(f"[Step 1/6] Model loaded in {time.time() - start_time:.2f}s")

    # Step 2: Process files
    logger.info("=" * 50)
    logger.info("[Step 2/6] Processing input files...")
    start_time = time.time()
    adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths, model.relation2id, params.add_traspose_rels)
    logger.info(f"[Step 2/6] Files processed in {time.time() - start_time:.2f}s")
    logger.info(f"  - Entities: {len(entity2id):,}")
    logger.info(f"  - Relations: {len(relation2id)}")
    logger.info(f"  - Test links: {len(triplets['links']):,}")

    # Step 3: Load KGE embeddings (if enabled)
    logger.info("=" * 50)
    logger.info("[Step 3/6] Loading KGE embeddings...")
    start_time = time.time()
    node_features, kge_entity2id = get_kge_embeddings(params.dataset, params.kge_model) if params.use_kge_embeddings else (None, None)
    if node_features is not None:
        logger.info(f"[Step 3/6] KGE embeddings loaded in {time.time() - start_time:.2f}s, shape: {node_features.shape}")
    else:
        logger.info(f"[Step 3/6] No KGE embeddings (skipped)")

    # Step 4: Pre-compute A_incidence matrix
    logger.info("=" * 50)
    logger.info("[Step 4/6] Pre-computing A_incidence matrix...")
    start_time = time.time()
    A_incidence_precomputed = incidence_matrix(adj_list)
    A_incidence_precomputed += A_incidence_precomputed.T
    # Convert to CSR for fast BFS
    if not isinstance(A_incidence_precomputed, ssp.csr_matrix):
        A_incidence_precomputed = A_incidence_precomputed.tocsr()
    logger.info(f"[Step 4/6] A_incidence pre-computed in {time.time() - start_time:.2f}s, shape: {A_incidence_precomputed.shape}")

    # Load semantic embeddings if enabled
    semantic_embeddings = None
    if hasattr(params, 'use_semantic_pruning') and params.use_semantic_pruning:
        if params.semantic_embeddings_path:
            logger.info(f"Loading semantic embeddings from {params.semantic_embeddings_path}")
            semantic_embeddings = np.load(params.semantic_embeddings_path)
            logger.info(f"Loaded semantic embeddings: {semantic_embeddings.shape}")

    # Step 5: Generate negative samples
    logger.info("=" * 50)
    logger.info("[Step 5/6] Generating negative samples...")
    start_time = time.time()
    if params.mode == 'sample':
        neg_triplets = get_neg_samples_replacing_head_tail(triplets['links'], adj_list)
        save_to_file(neg_triplets, id2entity, id2relation)
    elif params.mode == 'all':
        neg_triplets = get_neg_samples_replacing_head_tail_all(triplets['links'], adj_list)
    elif params.mode == 'ruleN':
        neg_triplets = get_neg_samples_replacing_head_tail_from_ruleN(params.ruleN_pred_path, entity2id, relation2id)
    logger.info(f"[Step 5/6] Negative samples generated in {time.time() - start_time:.2f}s")
    logger.info(f"  - Total neg triplets: {len(neg_triplets):,}")

    # Step 6: Compute rankings
    logger.info("=" * 50)
    logger.info("[Step 6/6] Computing rankings (this may take a while)...")
    start_time = time.time()

    ranks = []
    all_head_scores = []
    all_tail_scores = []

    # Use GPU single-thread mode if CUDA is available (faster for inference)
    # Use CPU multiprocessing mode otherwise
    if params.device.type == 'cuda':
        logger.info(f"Using GPU mode (single-thread) on {params.device}")
        for i, neg_link in enumerate(tqdm(neg_triplets, desc="Computing ranks (GPU)")):
            head_scores, head_rank, tail_scores, tail_rank = get_rank_gpu(
                neg_link, model, adj_list, dgl_adj_list, id2entity, params,
                node_features, kge_entity2id, A_incidence_precomputed, params.device
            )
            ranks.append(head_rank)
            ranks.append(tail_rank)
            all_head_scores += head_scores.tolist()
            all_tail_scores += tail_scores.tolist()

            # Log progress every 100 links
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(neg_triplets) - i - 1)
                logger.info(f"  Progress: {i+1}/{len(neg_triplets)} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
    else:
        logger.info("Using CPU mode (multiprocessing)")
        # Pass pre-computed A_incidence to workers
        init_args = (model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, A_incidence_precomputed, semantic_embeddings)

        with mp.Pool(processes=None, initializer=intialize_worker, initargs=init_args) as p:
            for head_scores, head_rank, tail_scores, tail_rank in tqdm(p.imap(get_rank, neg_triplets), total=len(neg_triplets), desc="Computing ranks (CPU)"):
                ranks.append(head_rank)
                ranks.append(tail_rank)

                all_head_scores += head_scores.tolist()
                all_tail_scores += tail_scores.tolist()

    logger.info(f"[Step 6/6] Rankings computed in {time.time() - start_time:.2f}s")

    # Save results
    logger.info("=" * 50)
    logger.info("Saving results...")
    if params.mode == 'ruleN':
        save_score_to_file_from_ruleN(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation)
    else:
        save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation)

    isHit1List = [x for x in ranks if x <= 1]
    isHit5List = [x for x in ranks if x <= 5]
    isHit10List = [x for x in ranks if x <= 10]
    hits_1 = len(isHit1List) / len(ranks)
    hits_5 = len(isHit5List) / len(ranks)
    hits_10 = len(isHit10List) / len(ranks)

    mrr = np.mean(1 / np.array(ranks))

    logger.info(f'MRR | Hits@1 | Hits@5 | Hits@10 : {mrr} | {hits_1} | {hits_5} | {hits_10}')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Testing script for hits@10')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="fb_v2_margin_loss",
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str, default="FB237_v2",
                        help="Path to dataset")
    parser.add_argument("--mode", "-m", type=str, default="sample", choices=["sample", "all", "ruleN"],
                        help="Negative sampling mode")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument("--hop", type=int, default=3,
                        help="How many hops to go while eextracting subgraphs?")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='Whether to append adj matrix list with symmetric relations?')

    # GPU params
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')

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

    params.file_paths = {
        'graph': os.path.join('./data', params.dataset, 'train.txt'),
        'links': os.path.join('./data', params.dataset, 'test.txt')
    }

    params.ruleN_pred_path = os.path.join('./data', params.dataset, 'pos_predictions.txt')
    params.model_path = os.path.join('experiments', params.experiment_name, 'best_graph_classifier.pth')

    # Set device (GPU or CPU)
    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    file_handler = logging.FileHandler(os.path.join('experiments', params.experiment_name, f'log_rank_test_{time.time()}.txt'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info(f'Device: {params.device}')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    main(params)
