"""
Evaluate GraIL model using OGB official evaluator (type-constrained)
Uses pre-extracted subgraphs from LMDB for speed.

Usage: python evaluate_grail_ogb.py --experiment_name exp_name --db_path data/ogbl-biokg-full/test_ogb_lmdb
"""

import argparse
import torch
import numpy as np
import os
import sys
import logging
import lmdb
import pickle
import dgl
from tqdm import tqdm
from ogb.linkproppred import Evaluator
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from model.dgl.graph_classifier import GraphClassifier
from utils.initialization_utils import initialize_model
from utils.graph_utils import ssp_multigraph_to_dgl, deserialize
from utils.data_utils import process_files

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SubgraphDatasetOGB:
    """Dataset for loading pre-extracted subgraphs from LMDB (3 separate files)"""

    def __init__(self, db_path, raw_data_paths, num_neg_samples=500, split='test'):
        # Load from 3 separate LMDB files
        self.pos_env = lmdb.open(db_path + '_pos', readonly=True, max_dbs=3, lock=False)
        self.neg_head_env = lmdb.open(db_path + '_neg_head', readonly=True, max_dbs=3, lock=False)
        self.neg_tail_env = lmdb.open(db_path + '_neg_tail', readonly=True, max_dbs=3, lock=False)

        self.db_pos = self.pos_env.open_db(f'{split}_pos'.encode())
        self.db_neg_head = self.neg_head_env.open_db(f'{split}_pos'.encode())  # stored as pos in each LMDB
        self.db_neg_tail = self.neg_tail_env.open_db(f'{split}_pos'.encode())  # stored as pos in each LMDB

        self.num_neg_samples = num_neg_samples

        # Load graph for subgraph preparation
        ssp_graph, __, __, __, id2entity, id2relation = process_files(raw_data_paths, None)
        self.num_rels = len(ssp_graph)
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.id2entity = id2entity
        self.id2relation = id2relation

        # Get max labels from positive LMDB
        with self.pos_env.begin() as txn:
            self.max_n_label = np.array([0, 0])
            max_sub = txn.get('max_n_label_sub'.encode())
            max_obj = txn.get('max_n_label_obj'.encode())
            if max_sub:
                self.max_n_label[0] = int.from_bytes(max_sub, byteorder='little')
            if max_obj:
                self.max_n_label[1] = int.from_bytes(max_obj, byteorder='little')

        with self.pos_env.begin(db=self.db_pos) as txn:
            self.num_graphs = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        # Get neg counts
        with self.neg_head_env.begin(db=self.db_neg_head) as txn:
            self.num_neg_head = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.neg_tail_env.begin(db=self.db_neg_tail) as txn:
            self.num_neg_tail = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        logging.info(f"Loaded LMDB with {self.num_graphs} positive, {self.num_neg_head} neg_head, {self.num_neg_tail} neg_tail")

    def get_pos(self, index):
        """Get positive subgraph"""
        with self.pos_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            data = deserialize(txn.get(str_id))
            if data is None:
                return None, None
            return self._prepare_subgraph(data), data['r_label']

    def get_neg_heads(self, index):
        """Get negative head subgraphs (500 per positive) - BULK READ"""
        # Bulk read all data first
        base_idx = index * self.num_neg_samples
        data_list = []
        with self.neg_head_env.begin(db=self.db_neg_head) as txn:
            for i in range(self.num_neg_samples):
                str_id = '{:08}'.format(base_idx + i).encode('ascii')
                data_list.append(deserialize(txn.get(str_id)))

        # Parallel prepare subgraphs using ThreadPool
        def prepare(data):
            if data is not None:
                return self._prepare_subgraph(data)
            return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            subgraphs = list(executor.map(prepare, data_list))
        return subgraphs

    def get_neg_tails(self, index):
        """Get negative tail subgraphs (500 per positive) - BULK READ"""
        # Bulk read all data first
        base_idx = index * self.num_neg_samples
        data_list = []
        with self.neg_tail_env.begin(db=self.db_neg_tail) as txn:
            for i in range(self.num_neg_samples):
                str_id = '{:08}'.format(base_idx + i).encode('ascii')
                data_list.append(deserialize(txn.get(str_id)))

        # Parallel prepare subgraphs using ThreadPool
        def prepare(data):
            if data is not None:
                return self._prepare_subgraph(data)
            return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            subgraphs = list(executor.map(prepare, data_list))
        return subgraphs

    def _prepare_subgraph(self, data):
        """Prepare DGL subgraph from LMDB data"""
        nodes = data['nodes']
        r_label = data['r_label']
        n_labels = data['n_label']

        subgraph = self.graph.subgraph(nodes)
        subgraph.edata['type'] = self.graph.edata['type'][subgraph.edata[dgl.EID]]
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        # Add edge between roots if needed
        try:
            edges_btw_roots = subgraph.edge_ids(0, 1)
            rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
            if rel_link.squeeze().nelement() == 0:
                subgraph.add_edges(0, 1)
                subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
                subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)
        except:
            subgraph.add_edges(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)

        # Prepare features
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        subgraph.ndata['feat'] = torch.FloatTensor(label_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1
        n_ids[tail_id] = 2
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        return subgraph


def prefetch_triplet(args):
    """Prefetch a single triplet's data (for parallel loading)"""
    idx, dataset = args
    try:
        subgraph_pos, r_label = dataset.get_pos(idx)
        neg_heads = dataset.get_neg_heads(idx)
        neg_tails = dataset.get_neg_tails(idx)
        return idx, subgraph_pos, r_label, neg_heads, neg_tails
    except Exception as e:
        logging.warning(f"Error prefetching triplet {idx}: {e}")
        return idx, None, None, [], []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='ogbl-biokg-full')
    parser.add_argument('--db_path', type=str, required=True, help='Path to LMDB with OGB negatives (without _pos/_neg_head/_neg_tail suffix)')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'valid'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_test', type=int, default=None, help='Max test samples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for scoring')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--prefetch', type=int, default=4, help='Number of triplets to prefetch')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load data paths
    data_path = os.path.join('data', args.dataset)
    file_paths = {
        'train': os.path.join(data_path, 'train.txt'),
        'valid': os.path.join(data_path, 'valid.txt'),
        'test': os.path.join(data_path, 'test.txt')
    }

    # Load dataset from LMDB
    logging.info(f"Loading LMDB from {args.db_path}...")
    dataset = SubgraphDatasetOGB(args.db_path, file_paths, num_neg_samples=500, split=args.split)

    num_test = dataset.num_graphs
    if args.max_test:
        num_test = min(num_test, args.max_test)

    logging.info(f"Number of test triplets: {num_test}")

    # Load model
    experiment_dir = os.path.join('experiments', args.experiment_name)
    logging.info(f"Loading model from {experiment_dir}...")

    import json
    params_path = os.path.join(experiment_dir, 'params.json')
    if os.path.exists(params_path):
        with open(params_path) as f:
            params_dict = json.load(f)
        from argparse import Namespace
        params = Namespace(**params_dict)
    else:
        raise FileNotFoundError(f"params.json not found in {experiment_dir}")

    params.experiment_name = args.experiment_name
    params.device = device

    # Initialize and load model
    model = initialize_model(params, None, load_model=True)
    model = model.to(device)
    model.eval()
    logging.info("Model loaded successfully")

    # Setup evaluator
    evaluator = Evaluator(name='ogbl-biokg')

    # Evaluate with FULL GPU batching + PREFETCHING
    logging.info("\nStarting evaluation from LMDB (GPU batched + prefetching)...")
    logging.info(f"Using {args.num_workers} workers for prefetching, batch_size={args.batch_size}")

    y_pred_pos_head = []
    y_pred_neg_head = []
    y_pred_pos_tail = []
    y_pred_neg_tail = []

    # Statistics for missing subgraphs
    missing_pos = 0
    missing_neg_head = 0
    missing_neg_tail = 0
    total_neg_head = 0
    total_neg_tail = 0

    # Missing subgraph = no structural path = very unlikely link = very low score
    MISSING_SCORE = -1e9

    # Process in batches of triplets with prefetching
    triplet_batch_size = args.prefetch  # Process multiple triplets at once

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import queue
    import threading

    # Create a prefetch queue
    prefetch_queue = queue.Queue(maxsize=args.prefetch * 2)
    stop_event = threading.Event()

    def prefetch_worker():
        """Background thread for prefetching triplets"""
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            batch_start = 0
            while batch_start < num_test and not stop_event.is_set():
                batch_end = min(batch_start + triplet_batch_size, num_test)
                futures = []
                for idx in range(batch_start, batch_end):
                    future = executor.submit(lambda i: (
                        i,
                        dataset.get_pos(i),
                        dataset.get_neg_heads(i),
                        dataset.get_neg_tails(i)
                    ), idx)
                    futures.append(future)

                for future in futures:
                    if stop_event.is_set():
                        break
                    try:
                        result = future.result(timeout=60)
                        prefetch_queue.put(result)
                    except Exception as e:
                        logging.warning(f"Prefetch error: {e}")

                batch_start = batch_end

        prefetch_queue.put(None)  # Signal end

    # Start prefetch thread
    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()

    pbar = tqdm(total=num_test, desc="Evaluating (prefetched)")
    processed = 0

    while processed < num_test:
        # Get prefetched data
        try:
            data = prefetch_queue.get(timeout=120)
        except queue.Empty:
            logging.warning("Prefetch queue timeout, breaking")
            break

        if data is None:
            break

        idx, (subgraph_pos, r_label), neg_heads, neg_tails = data

        if subgraph_pos is None:
            missing_pos += 1
            processed += 1
            pbar.update(1)
            continue

        # Collect ALL graphs for this triplet
        all_graphs = [subgraph_pos]
        all_labels = [r_label]

        neg_head_valid_indices = []
        neg_tail_valid_indices = []

        for i, sg in enumerate(neg_heads):
            total_neg_head += 1
            if sg is not None:
                all_graphs.append(sg)
                all_labels.append(r_label)
                neg_head_valid_indices.append(i)
            else:
                missing_neg_head += 1

        num_valid_neg_head = len(neg_head_valid_indices)

        for i, sg in enumerate(neg_tails):
            total_neg_tail += 1
            if sg is not None:
                all_graphs.append(sg)
                all_labels.append(r_label)
                neg_tail_valid_indices.append(i)
            else:
                missing_neg_tail += 1

        num_valid_neg_tail = len(neg_tail_valid_indices)

        # BATCH ALL GRAPHS AT ONCE - maximum GPU utilization!
        batched = dgl.batch(all_graphs).to(device)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long, device=device)

        with torch.no_grad():
            all_scores = model((batched, labels_tensor)).squeeze(-1).cpu().tolist()

        # Extract scores
        pos_score = all_scores[0]
        neg_head_scores = all_scores[1:1+num_valid_neg_head]
        neg_tail_scores = all_scores[1+num_valid_neg_head:]

        y_pred_pos_head.append(pos_score)
        y_pred_pos_tail.append(pos_score)

        # Reconstruct neg_head scores with MISSING_SCORE
        full_neg_scores_head = [MISSING_SCORE] * len(neg_heads)
        for j, i in enumerate(neg_head_valid_indices):
            full_neg_scores_head[i] = neg_head_scores[j]
        y_pred_neg_head.append(full_neg_scores_head)

        # Reconstruct neg_tail scores with MISSING_SCORE
        full_neg_scores_tail = [MISSING_SCORE] * len(neg_tails)
        for j, i in enumerate(neg_tail_valid_indices):
            full_neg_scores_tail[i] = neg_tail_scores[j]
        y_pred_neg_tail.append(full_neg_scores_tail)

        processed += 1
        pbar.update(1)

    pbar.close()
    stop_event.set()

    # Log missing subgraph statistics
    logging.info(f"\nSubgraph Statistics:")
    logging.info(f"  Missing positive: {missing_pos:,} / {num_test:,} ({100*missing_pos/num_test:.2f}%)")
    if total_neg_head > 0:
        logging.info(f"  Missing neg head: {missing_neg_head:,} / {total_neg_head:,} ({100*missing_neg_head/total_neg_head:.2f}%)")
    if total_neg_tail > 0:
        logging.info(f"  Missing neg tail: {missing_neg_tail:,} / {total_neg_tail:,} ({100*missing_neg_tail/total_neg_tail:.2f}%)")
    logging.info(f"  (Missing negatives get score={MISSING_SCORE} → helps positive rank higher)")

    # Convert to tensors
    y_pred_pos_head = torch.tensor(y_pred_pos_head)
    y_pred_neg_head = torch.tensor(y_pred_neg_head)
    y_pred_pos_tail = torch.tensor(y_pred_pos_tail)
    y_pred_neg_tail = torch.tensor(y_pred_neg_tail)

    logging.info(f"\nPrediction shapes:")
    logging.info(f"  y_pred_pos_head: {y_pred_pos_head.shape}")
    logging.info(f"  y_pred_neg_head: {y_pred_neg_head.shape}")
    logging.info(f"  y_pred_pos_tail: {y_pred_pos_tail.shape}")
    logging.info(f"  y_pred_neg_tail: {y_pred_neg_tail.shape}")

    # Evaluate using OGB evaluator
    results_head = evaluator.eval({
        'y_pred_pos': y_pred_pos_head,
        'y_pred_neg': y_pred_neg_head,
    })

    results_tail = evaluator.eval({
        'y_pred_pos': y_pred_pos_tail,
        'y_pred_neg': y_pred_neg_tail,
    })

    # Average head and tail predictions
    mrr_head = results_head['mrr_list'].mean().item()
    mrr_tail = results_tail['mrr_list'].mean().item()
    mrr_avg = (mrr_head + mrr_tail) / 2

    hits1_head = results_head['hits@1_list'].mean().item()
    hits1_tail = results_tail['hits@1_list'].mean().item()
    hits1_avg = (hits1_head + hits1_tail) / 2

    hits3_head = results_head['hits@3_list'].mean().item()
    hits3_tail = results_tail['hits@3_list'].mean().item()
    hits3_avg = (hits3_head + hits3_tail) / 2

    hits10_head = results_head['hits@10_list'].mean().item()
    hits10_tail = results_tail['hits@10_list'].mean().item()
    hits10_avg = (hits10_head + hits10_tail) / 2

    print("\n" + "="*60)
    print(f"OGB Evaluation Results for GraIL ({args.experiment_name})")
    print("="*60)
    print(f"Head Prediction - MRR: {mrr_head:.4f}")
    print(f"Tail Prediction - MRR: {mrr_tail:.4f}")
    print(f"Average MRR: {mrr_avg:.4f}")
    print(f"Average Hits@1: {hits1_avg:.4f}")
    print(f"Average Hits@3: {hits3_avg:.4f}")
    print(f"Average Hits@10: {hits10_avg:.4f}")
    print("="*60)

    # Save results
    results_file = os.path.join(experiment_dir, 'ogb_results.json')
    results_dict = {
        'mrr_head': mrr_head,
        'mrr_tail': mrr_tail,
        'mrr_avg': mrr_avg,
        'hits1_avg': hits1_avg,
        'hits3_avg': hits3_avg,
        'hits10_avg': hits10_avg,
        'num_test': num_test
    }
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logging.info(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
