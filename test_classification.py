"""
Classification Metrics Evaluation for GRAIL and KGE
====================================================

Evaluates both GRAIL (subgraph-based) and KGE (embedding-based) models
on CLASSIFICATION metrics: AUC, AUC-PR, Accuracy, F1, etc.

This demonstrates GRAIL's strength in binary classification tasks.

Usage:
    python test_classification.py \
        --grail_experiment hard_neg_50_compgcn \
        --db_path data/ogbl-biokg-full/200_hop2_new \
        --kge_path kge/1760606770.5893373 \
        --max_samples 10000

Author: GRAIL Evaluation
"""

import argparse
import os
import sys
import json
import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
import lmdb
import dgl
from tqdm import tqdm
from sklearn import metrics
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.dgl.graph_classifier import GraphClassifier
from utils.initialization_utils import initialize_model
from utils.graph_utils import ssp_multigraph_to_dgl, deserialize
from utils.data_utils import process_files


class ClassificationDataset:
    """Dataset for classification evaluation from LMDB"""

    def __init__(self, db_path, raw_data_paths, split='test'):
        self.env = lmdb.open(db_path, readonly=True, max_dbs=6, lock=False)
        self.db_pos = self.env.open_db(f'{split}_pos'.encode())
        self.db_neg = self.env.open_db(f'{split}_neg'.encode())

        # Load graph structure
        ssp_graph, __, __, __, id2entity, id2relation = process_files(raw_data_paths, None)
        self.num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.id2entity = id2entity
        self.id2relation = id2relation

        # Get max labels
        with self.env.begin() as txn:
            self.max_n_label = np.array([0, 0])
            max_sub = txn.get('max_n_label_sub'.encode())
            max_obj = txn.get('max_n_label_obj'.encode())
            if max_sub:
                self.max_n_label[0] = int.from_bytes(max_sub, byteorder='little')
            if max_obj:
                self.max_n_label[1] = int.from_bytes(max_obj, byteorder='little')

        # Get counts
        with self.env.begin(db=self.db_pos) as txn:
            self.num_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.env.begin(db=self.db_neg) as txn:
            self.num_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        logging.info(f"Loaded {self.num_pos:,} positive, {self.num_neg:,} negative samples")

    def get_sample(self, idx, is_positive=True):
        """Get a single sample"""
        db = self.db_pos if is_positive else self.db_neg
        with self.env.begin(db=db) as txn:
            str_id = '{:08}'.format(idx).encode('ascii')
            data = deserialize(txn.get(str_id))
            if data is None:
                return None, None, None, None

            nodes = data['nodes']
            r_label = data['r_label']
            n_labels = data['n_labels'] if 'n_labels' in data else data.get('n_label')

            # Head and tail are first two nodes
            head, tail = nodes[0], nodes[1]

            return self._prepare_subgraph(data), r_label, head, tail

    def _prepare_subgraph(self, data):
        """Prepare DGL subgraph"""
        nodes = data['nodes']
        r_label = data['r_label']
        n_labels = data['n_labels'] if 'n_labels' in data else data.get('n_label')

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

        # Prepare node features
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

    def close(self):
        self.env.close()


class KGEScorer:
    """KGE-based scorer using entity and relation embeddings"""

    def __init__(self, kge_path, scoring_method='transe'):
        """
        Args:
            kge_path: Path to KGE embeddings directory
            scoring_method: 'transe', 'distmult', or 'complex'
        """
        self.scoring_method = scoring_method

        # Load embeddings
        entity_emb_path = os.path.join(kge_path, 'entity_embedding.npy')
        relation_emb_path = os.path.join(kge_path, 'relation_embedding.npy')

        logging.info(f"Loading KGE embeddings from {kge_path}...")
        self.entity_emb = np.load(entity_emb_path)
        self.relation_emb = np.load(relation_emb_path)

        logging.info(f"Entity embeddings: {self.entity_emb.shape}")
        logging.info(f"Relation embeddings: {self.relation_emb.shape}")

        # Convert to torch tensors
        self.entity_emb = torch.FloatTensor(self.entity_emb)
        self.relation_emb = torch.FloatTensor(self.relation_emb)

    def to(self, device):
        """Move embeddings to device"""
        self.entity_emb = self.entity_emb.to(device)
        self.relation_emb = self.relation_emb.to(device)
        return self

    def score(self, heads, relations, tails):
        """
        Score triplets using KGE method

        Args:
            heads: tensor of head entity IDs
            relations: tensor of relation IDs
            tails: tensor of tail entity IDs

        Returns:
            scores: tensor of scores (higher = more likely true)
        """
        h_emb = self.entity_emb[heads]
        r_emb = self.relation_emb[relations]
        t_emb = self.entity_emb[tails]

        if self.scoring_method == 'transe':
            # TransE: score = -||h + r - t||
            scores = -torch.norm(h_emb + r_emb - t_emb, p=2, dim=-1)
        elif self.scoring_method == 'distmult':
            # DistMult: score = <h, r, t>
            scores = torch.sum(h_emb * r_emb * t_emb, dim=-1)
        elif self.scoring_method == 'complex':
            # ComplEx (simplified): real and imaginary parts
            dim = h_emb.shape[-1] // 2
            h_re, h_im = h_emb[..., :dim], h_emb[..., dim:]
            r_re, r_im = r_emb[..., :dim], r_emb[..., dim:]
            t_re, t_im = t_emb[..., :dim], t_emb[..., dim:]
            scores = torch.sum(
                h_re * r_re * t_re + h_im * r_re * t_im +
                h_re * r_im * t_im - h_im * r_im * t_re, dim=-1
            )
        else:
            # Default: dot product similarity
            scores = torch.sum(h_emb * t_emb, dim=-1)

        return scores


def evaluate_grail(model, dataset, device, max_samples=None, batch_size=64):
    """Evaluate GRAIL model on classification task"""

    logging.info("\n" + "="*60)
    logging.info("Evaluating GRAIL (Subgraph-based)")
    logging.info("="*60)

    model.eval()

    all_scores = []
    all_labels = []
    all_triplets = []  # For KGE comparison

    num_pos = min(dataset.num_pos, max_samples // 2) if max_samples else dataset.num_pos
    num_neg = min(dataset.num_neg, max_samples // 2) if max_samples else dataset.num_neg

    logging.info(f"Evaluating {num_pos:,} positive, {num_neg:,} negative samples")

    # Collect positive samples
    logging.info("Processing positive samples...")
    pos_graphs = []
    pos_labels = []
    pos_triplets = []

    for idx in tqdm(range(num_pos), desc="Loading positives"):
        subgraph, r_label, head, tail = dataset.get_sample(idx, is_positive=True)
        if subgraph is not None:
            pos_graphs.append(subgraph)
            pos_labels.append(r_label)
            pos_triplets.append((head, r_label, tail))

    # Collect negative samples
    logging.info("Processing negative samples...")
    neg_graphs = []
    neg_labels = []
    neg_triplets = []

    for idx in tqdm(range(num_neg), desc="Loading negatives"):
        subgraph, r_label, head, tail = dataset.get_sample(idx, is_positive=False)
        if subgraph is not None:
            neg_graphs.append(subgraph)
            neg_labels.append(r_label)
            neg_triplets.append((head, r_label, tail))

    # Score in batches
    logging.info("Scoring with GRAIL model...")

    def score_batch(graphs, labels):
        scores = []
        for i in tqdm(range(0, len(graphs), batch_size), desc="Batching", leave=False):
            batch_graphs = graphs[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            batched = dgl.batch(batch_graphs).to(device)
            labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

            with torch.no_grad():
                batch_scores = model((batched, labels_tensor)).squeeze(-1).cpu().tolist()
            scores.extend(batch_scores)
        return scores

    pos_scores = score_batch(pos_graphs, pos_labels)
    neg_scores = score_batch(neg_graphs, neg_labels)

    all_scores = pos_scores + neg_scores
    all_labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    all_triplets = pos_triplets + neg_triplets

    # Compute metrics
    results = compute_classification_metrics(all_scores, all_labels, "GRAIL")

    return results, all_triplets, all_scores, all_labels


def evaluate_kge(kge_scorer, triplets, labels, device, scoring_method='transe'):
    """Evaluate KGE model on classification task"""

    logging.info("\n" + "="*60)
    logging.info(f"Evaluating KGE ({scoring_method.upper()})")
    logging.info("="*60)

    # Extract heads, relations, tails
    heads = torch.tensor([t[0] for t in triplets], dtype=torch.long, device=device)
    relations = torch.tensor([t[1] for t in triplets], dtype=torch.long, device=device)
    tails = torch.tensor([t[2] for t in triplets], dtype=torch.long, device=device)

    logging.info(f"Scoring {len(triplets):,} triplets...")

    with torch.no_grad():
        scores = kge_scorer.score(heads, relations, tails).cpu().numpy()

    # Compute metrics
    results = compute_classification_metrics(scores.tolist(), labels, f"KGE-{scoring_method.upper()}")

    return results, scores.tolist()


def compute_classification_metrics(scores, labels, model_name):
    """Compute classification metrics"""

    scores = np.array(scores)
    labels = np.array(labels)

    # AUC-ROC
    auc_roc = metrics.roc_auc_score(labels, scores)

    # AUC-PR (Average Precision)
    auc_pr = metrics.average_precision_score(labels, scores)

    # Find optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Binary predictions at optimal threshold
    preds = (scores >= optimal_threshold).astype(int)

    # Accuracy
    accuracy = metrics.accuracy_score(labels, preds)

    # Precision, Recall, F1
    precision = metrics.precision_score(labels, preds, zero_division=0)
    recall = metrics.recall_score(labels, preds, zero_division=0)
    f1 = metrics.f1_score(labels, preds, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(labels, preds).ravel()

    # Score statistics
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    results = {
        'model': model_name,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'optimal_threshold': optimal_threshold,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'pos_score_mean': float(np.mean(pos_scores)),
        'pos_score_std': float(np.std(pos_scores)),
        'neg_score_mean': float(np.mean(neg_scores)),
        'neg_score_std': float(np.std(neg_scores)),
        'score_separation': float(np.mean(pos_scores) - np.mean(neg_scores))
    }

    return results


def print_results(results):
    """Pretty print results"""
    print("\n" + "="*60)
    print(f"Results for {results['model']}")
    print("="*60)
    print(f"{'Metric':<25} {'Value':>15}")
    print("-"*40)
    print(f"{'AUC-ROC':<25} {results['auc_roc']:>15.4f}")
    print(f"{'AUC-PR':<25} {results['auc_pr']:>15.4f}")
    print(f"{'Accuracy':<25} {results['accuracy']:>15.4f}")
    print(f"{'Precision':<25} {results['precision']:>15.4f}")
    print(f"{'Recall':<25} {results['recall']:>15.4f}")
    print(f"{'F1 Score':<25} {results['f1']:>15.4f}")
    print("-"*40)
    print(f"{'Optimal Threshold':<25} {results['optimal_threshold']:>15.4f}")
    print(f"{'True Positives':<25} {results['true_positives']:>15,}")
    print(f"{'True Negatives':<25} {results['true_negatives']:>15,}")
    print(f"{'False Positives':<25} {results['false_positives']:>15,}")
    print(f"{'False Negatives':<25} {results['false_negatives']:>15,}")
    print("-"*40)
    print(f"{'Pos Score Mean':<25} {results['pos_score_mean']:>15.4f}")
    print(f"{'Pos Score Std':<25} {results['pos_score_std']:>15.4f}")
    print(f"{'Neg Score Mean':<25} {results['neg_score_mean']:>15.4f}")
    print(f"{'Neg Score Std':<25} {results['neg_score_std']:>15.4f}")
    print(f"{'Score Separation':<25} {results['score_separation']:>15.4f}")
    print("="*60)


def print_comparison(grail_results, kge_results_list):
    """Print comparison table"""

    print("\n" + "="*80)
    print("COMPARISON: GRAIL vs KGE")
    print("="*80)

    # Header
    models = ['GRAIL'] + [r['model'] for r in kge_results_list]
    header = f"{'Metric':<20}"
    for m in models:
        header += f"{m:>15}"
    print(header)
    print("-"*80)

    # Metrics to compare
    compare_metrics = ['auc_roc', 'auc_pr', 'accuracy', 'precision', 'recall', 'f1', 'score_separation']
    metric_names = ['AUC-ROC', 'AUC-PR', 'Accuracy', 'Precision', 'Recall', 'F1', 'Score Sep.']

    all_results = [grail_results] + kge_results_list

    for metric, name in zip(compare_metrics, metric_names):
        row = f"{name:<20}"
        values = [r[metric] for r in all_results]
        best_idx = np.argmax(values)
        for i, v in enumerate(values):
            if i == best_idx:
                row += f"{v:>14.4f}*"  # Mark best with *
            else:
                row += f"{v:>15.4f}"
        print(row)

    print("="*80)
    print("* = Best performance")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Classification Metrics Evaluation')
    parser.add_argument('--grail_experiment', type=str, default='hard_neg_50_compgcn',
                        help='GRAIL experiment name')
    parser.add_argument('--db_path', type=str, default='data/ogbl-biokg-full/200_hop2_new',
                        help='Path to LMDB database')
    parser.add_argument('--dataset', type=str, default='ogbl-biokg-full',
                        help='Dataset name')
    parser.add_argument('--kge_path', type=str, default='kge/1760606770.5893373',
                        help='Path to KGE embeddings')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help='Data split to evaluate')
    parser.add_argument('--max_samples', type=int, default=20000,
                        help='Maximum samples to evaluate (None for all)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for GRAIL scoring')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--kge_methods', type=str, default='transe,distmult',
                        help='KGE scoring methods (comma-separated)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Data paths
    data_path = os.path.join('data', args.dataset)
    file_paths = {
        'train': os.path.join(data_path, 'train.txt'),
        'valid': os.path.join(data_path, 'valid.txt'),
        'test': os.path.join(data_path, 'test.txt')
    }

    # Load dataset
    logging.info(f"Loading dataset from {args.db_path}...")
    dataset = ClassificationDataset(args.db_path, file_paths, split=args.split)

    # Load GRAIL model
    experiment_dir = os.path.join('experiments', args.grail_experiment)
    logging.info(f"Loading GRAIL model from {experiment_dir}...")

    params_path = os.path.join(experiment_dir, 'params.json')
    with open(params_path) as f:
        params_dict = json.load(f)

    from argparse import Namespace
    params = Namespace(**params_dict)
    params.experiment_name = args.grail_experiment
    params.device = device

    grail_model = initialize_model(params, None, load_model=True)
    grail_model = grail_model.to(device)
    grail_model.eval()

    # Evaluate GRAIL
    start_time = time.time()
    grail_results, triplets, grail_scores, labels = evaluate_grail(
        grail_model, dataset, device,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    grail_time = time.time() - start_time
    grail_results['eval_time'] = grail_time
    print_results(grail_results)

    # Evaluate KGE methods
    kge_results_list = []
    kge_methods = [m.strip() for m in args.kge_methods.split(',')]

    for method in kge_methods:
        try:
            kge_scorer = KGEScorer(args.kge_path, scoring_method=method)
            kge_scorer.to(device)

            start_time = time.time()
            kge_results, kge_scores = evaluate_kge(
                kge_scorer, triplets, labels, device,
                scoring_method=method
            )
            kge_time = time.time() - start_time
            kge_results['eval_time'] = kge_time

            print_results(kge_results)
            kge_results_list.append(kge_results)
        except Exception as e:
            logging.error(f"Error evaluating KGE-{method}: {e}")

    # Print comparison
    if kge_results_list:
        print_comparison(grail_results, kge_results_list)

    # Save results
    output_file = args.output or os.path.join(experiment_dir, 'classification_results.json')
    all_results = {
        'grail': grail_results,
        'kge': {r['model']: r for r in kge_results_list},
        'config': {
            'db_path': args.db_path,
            'split': args.split,
            'max_samples': args.max_samples,
            'num_samples': len(labels),
            'num_positive': sum(labels),
            'num_negative': len(labels) - sum(labels)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"\nResults saved to {output_file}")

    dataset.close()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"GRAIL AUC-ROC: {grail_results['auc_roc']:.4f} (Time: {grail_results['eval_time']:.1f}s)")
    for r in kge_results_list:
        print(f"{r['model']} AUC-ROC: {r['auc_roc']:.4f} (Time: {r['eval_time']:.1f}s)")
    print("="*80)


if __name__ == '__main__':
    main()
