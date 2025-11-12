from torch.utils.data import Dataset
import timeit
import os
import struct
import logging
import lmdb
import numpy as np
import json
import pickle
import time
import torch
import dgl
from tqdm import tqdm
from utils.graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from utils.data_utils import process_files, save_to_file, plot_rel_dist
from .graph_sampler import *


def process_sampling_task(task):
    """Process a single sampling task - defined at module level for pickle compatibility"""
    from .graph_sampler import sample_fair_neg
    import traceback

    try:
        logging.info(f"Starting sampling for {task['split_name']} ({len(task['triplets']):,} triplets)")
        start_time = time.time()

        pos, neg = sample_fair_neg(
            task['adj_list'],
            task['triplets'],
            task['data_dir'],
            task['num_neg_samples_per_link'],
            max_size=task['max_size'],
            constrained_neg_prob=task['constrained_neg_prob']
        )

        elapsed = time.time() - start_time
        logging.info(f"Completed {task['split_name']} sampling in {elapsed:.1f}s ({len(neg):,} negatives)")

        return task['split_name'], pos, neg
    except Exception as e:
        logging.error(f"Error in process_sampling_task for {task['split_name']}: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise


def generate_subgraph_datasets(params, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None):

    testing = 'test' in splits
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths, saved_relation2id)

    # plot_rel_dist(adj_list, os.path.join(params.main_dir, f'data/{params.dataset}/rel_dist.png'))

    data_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
    if not os.path.isdir(data_path) and not testing:
        with open(data_path, 'w') as f:
            json.dump(relation2id, f)

    graphs = {}

    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}

    # Get data directory for entity type mapping
    data_dir = os.path.join(params.main_dir, f'data/{params.dataset}')

    # OPTIMIZATION: Sample train and valid/test links in parallel with 8 cores
    logging.info("Starting ULTRA-FAST parallel negative sampling with 8 cores...")

    # Prepare data for parallel processing
    sampling_tasks = []
    for split_name, split in graphs.items():
        sampling_tasks.append({
            'split_name': split_name,
            'adj_list': adj_list,
            'triplets': split['triplets'],
            'data_dir': data_dir,
            'num_neg_samples_per_link': params.num_neg_samples_per_link,
            'max_size': split['max_size'],
            'constrained_neg_prob': params.constrained_neg_prob
        })

    # Process all splits in parallel with 8 cores
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import threading
    import queue

    # Progress tracking
    completed_tasks = 0
    total_tasks = len(sampling_tasks)
    start_total = time.time()

    # Process with all available cores for maximum speed
    import multiprocessing
    num_workers = min(multiprocessing.cpu_count(), 15)  # Use up to 15 workers
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_sampling_task, task) for task in sampling_tasks]

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                split_name, pos, neg = future.result()
                graphs[split_name]['pos'] = pos
                graphs[split_name]['neg'] = neg
                completed_tasks += 1

                # Progress update
                elapsed = time.time() - start_total
                eta = (total_tasks - completed_tasks) * elapsed / completed_tasks
                logging.info(f"Progress: {completed_tasks}/{total_tasks} splits completed | ETA: {eta/60:.1f}min")

            except Exception as e:
                import traceback
                logging.error(f"Sampling task failed: {e}")
                logging.error(f"Full traceback: {traceback.format_exc()}")

    logging.info("ALL SAMPLING TASKS COMPLETED!")

    if testing:
        directory = os.path.join(params.main_dir, 'data/{}/'.format(params.dataset))
        save_to_file(directory, f'neg_{params.test_file}_{params.constrained_neg_prob}.txt', graphs['test']['neg'], id2entity, id2relation)

    # Load semantic embeddings if semantic pruning is enabled
    semantic_embeddings = None
    if hasattr(params, 'use_semantic_pruning') and params.use_semantic_pruning:
        if params.semantic_embeddings_path:
            # Load embeddings from specified path
            logging.info(f"Loading semantic embeddings from {params.semantic_embeddings_path}")
            semantic_embeddings = np.load(params.semantic_embeddings_path)
        else:
            # Load default KGE embeddings
            logging.info(f"Loading default {params.kge_model} embeddings for semantic pruning")
            semantic_embeddings, _ = get_kge_embeddings(params.dataset, params.kge_model)

        if semantic_embeddings is not None:
            logging.info(f"Loaded semantic embeddings with shape: {semantic_embeddings.shape}")

    # Ensure subgraph extraction completed successfully before proceeding
    links2subgraphs(adj_list, graphs, params, max_label_value, semantic_embeddings)

    # Verify that the database directory exists
    if not os.path.isdir(params.db_path):
        logging.error(f"LMDB database directory not found: {params.db_path}")
        raise FileNotFoundError(f"LMDB database directory not found: {params.db_path}")


def get_kge_embeddings(dataset, kge_model):

    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, included_relations=None, add_traspose_rels=False, num_neg_samples_per_link=1, use_kge_embeddings=False, dataset='', kge_model='', file_name=''):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.node_features, self.kge_entity2id = get_kge_embeddings(dataset, kge_model) if use_kge_embeddings else (None, None)
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name

        ssp_graph, __, __, __, id2entity, id2relation = process_files(raw_data_paths, included_relations)
        self.num_rels = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        logging.info(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")

        # logging.info('=====================')
        # logging.info(f"Subgraph size stats: \n Avg size {self.avg_subgraph_size}, \n Min size {self.min_subgraph_size}, \n Max size {self.max_subgraph_size}, \n Std {self.std_subgraph_size}")

        # logging.info('=====================')
        # logging.info(f"Enclosed nodes ratio stats: \n Avg size {self.avg_enc_ratio}, \n Min size {self.min_enc_ratio}, \n Max size {self.max_enc_ratio}, \n Std {self.std_enc_ratio}")

        # logging.info('=====================')
        # logging.info(f"# of pruned nodes stats: \n Avg size {self.avg_num_pruned_nodes}, \n Min size {self.min_num_pruned_nodes}, \n Max size {self.max_num_pruned_nodes}, \n Std {self.std_num_pruned_nodes}")

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        # Test loading first subgraph with progress indication
        print("[DATASET] Loading first subgraph to initialize...")
        start_time = time.time()
        self.__getitem__(0)
        init_time = time.time() - start_time
        print(f"[DATASET] First subgraph loaded in {init_time:.4f} seconds")

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            data_pos = deserialize(txn.get(str_id))
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = data_pos['nodes'], data_pos['r_label'], data_pos['g_label'], data_pos['n_label']
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
        subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                data_neg = deserialize(txn.get(str_id))
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = data_neg['nodes'], data_neg['r_label'], data_neg['g_label'], data_neg['n_label']
                subgraphs_neg.append(self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg))
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)

        return subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, g_labels_neg, r_labels_neg

    def __len__(self):
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels):
        subgraph = self.graph.subgraph(nodes)
        subgraph.edata['type'] = self.graph.edata['type'][subgraph.edata[dgl.EID]]
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        # Check if edge exists between roots, handle DGL API
        try:
            edges_btw_roots = subgraph.edge_ids(0, 1)
            rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
            if rel_link.squeeze().nelement() == 0:
                subgraph.add_edges(0, 1)
                subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
                subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)
        except:
            # Edge doesn't exist, add it
            subgraph.add_edges(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)

        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, n_feats)

        return subgraph

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        label_feats[np.arange(n_nodes), n_labels] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph
