"""
Quick Ablation Test - Fast comparison of model configurations

Runs a quick test on a small subset of data to compare:
1. Effect of relation embedding (r)
2. Effect of query attention pooling
3. Effect of multi-head attention

Usage:
    python quick_ablation_test.py --db_path data/ogbl-biokg/100_tssp_hop2 --num_samples 500
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score
import time
from tqdm import tqdm

from subgraph_extraction.datasets import SubgraphDataset
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl
from model.dgl.graph_classifier_ablation import GraphClassifierAblation


def evaluate(model, dataloader, device):
    """Quick evaluation on a subset."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            data_pos, targets_pos, data_neg, targets_neg = move_batch_to_device_dgl(batch, device)

            # Positive samples
            score_pos = model(data_pos).squeeze()
            all_preds.extend(torch.sigmoid(score_pos).cpu().numpy())
            all_labels.extend([1] * len(score_pos))

            # Negative samples
            score_neg = model(data_neg).squeeze()
            all_preds.extend(torch.sigmoid(score_neg).cpu().numpy())
            all_labels.extend([0] * len(score_neg))

    auc = roc_auc_score(all_labels, all_preds)
    auc_pr = average_precision_score(all_labels, all_preds)

    return auc, auc_pr


def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs, config_name):
    """Train for one epoch with progress bar."""
    model.train()
    total_loss = 0
    num_batches = 0

    criterion = nn.MarginRankingLoss(margin=10)

    pbar = tqdm(dataloader, desc=f"[{config_name}] Epoch {epoch}/{total_epochs}",
                leave=False, ncols=100)

    for batch in pbar:
        data_pos, targets_pos, data_neg, targets_neg = move_batch_to_device_dgl(batch, device)

        optimizer.zero_grad()

        score_pos = model(data_pos)
        score_neg = model(data_neg)

        loss = criterion(score_pos, score_neg, torch.ones_like(score_pos))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})

    return total_loss / num_batches


def run_config(params, config_name, use_rel_emb, pool_type, pool_heads, gnn_type, train_loader, valid_loader):
    """Run a single configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"  gnn={gnn_type}, use_rel_emb={use_rel_emb}, pool_type={pool_type}, pool_heads={pool_heads}")
    print(f"{'='*60}")

    # Update params
    params.use_rel_emb = use_rel_emb
    params.pool_type = pool_type
    params.pool_heads = pool_heads
    params.gnn_type = gnn_type

    # Initialize model
    model = GraphClassifierAblation(params, params.relation2id).to(params.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    # Training
    start_time = time.time()
    best_auc = 0
    best_auc_pr = 0

    for epoch in range(params.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, params.device,
                                  epoch+1, params.num_epochs, config_name)
        val_auc, val_auc_pr = evaluate(model, valid_loader, params.device)

        if val_auc > best_auc:
            best_auc = val_auc
            best_auc_pr = val_auc_pr

        print(f"  Epoch {epoch+1}: loss={train_loss:.4f}, AUC={val_auc:.4f}, AUC-PR={val_auc_pr:.4f}")

    train_time = time.time() - start_time

    # Get attention weights if using query_attention
    attention_stats = None
    head_diversity = None
    if pool_type == 'query_attention':
        model.eval()
        with torch.no_grad():
            batch = next(iter(valid_loader))
            data_pos, _, _, _ = move_batch_to_device_dgl(batch, params.device)
            try:
                attn_weights, node_info = model.get_attention_analysis(data_pos)

                # Analyze attention distribution
                if isinstance(attn_weights, list):
                    # Multi-head - analyze each head separately
                    head_stats = []
                    for i, w in enumerate(attn_weights):
                        w_squeezed = w.squeeze()
                        h_entropy = -torch.sum(w_squeezed * torch.log(w_squeezed + 1e-10)).item() / len(w_squeezed)
                        head_stats.append({
                            'mean': w_squeezed.mean().item(),
                            'std': w_squeezed.std().item(),
                            'entropy': h_entropy
                        })

                    # Calculate head diversity (cosine similarity between heads)
                    head_vectors = [w.squeeze() for w in attn_weights]
                    diversity_scores = []
                    for i in range(len(head_vectors)):
                        for j in range(i+1, len(head_vectors)):
                            cos_sim = torch.nn.functional.cosine_similarity(
                                head_vectors[i].unsqueeze(0),
                                head_vectors[j].unsqueeze(0)
                            ).item()
                            diversity_scores.append(1 - cos_sim)  # diversity = 1 - similarity

                    head_diversity = np.mean(diversity_scores) if diversity_scores else 0

                    all_weights = torch.cat([w.squeeze() for w in attn_weights])
                    attention_stats = {
                        'mean': all_weights.mean().item(),
                        'std': all_weights.std().item(),
                        'entropy': -torch.sum(all_weights * torch.log(all_weights + 1e-10)).item() / len(all_weights),
                        'head_stats': head_stats,
                        'head_diversity': head_diversity
                    }
                    print(f"  Attention: entropy={attention_stats['entropy']:.4f}, head_diversity={head_diversity:.4f}")
                    for i, hs in enumerate(head_stats):
                        print(f"    Head {i}: entropy={hs['entropy']:.4f}, std={hs['std']:.4f}")
                else:
                    all_weights = attn_weights.squeeze()
                    attention_stats = {
                        'mean': all_weights.mean().item(),
                        'std': all_weights.std().item(),
                        'entropy': -torch.sum(all_weights * torch.log(all_weights + 1e-10)).item() / len(all_weights)
                    }
                    print(f"  Attention: entropy={attention_stats['entropy']:.4f}, std={attention_stats['std']:.4f}")

            except Exception as e:
                print(f"  Could not get attention stats: {e}")

    return {
        'config': config_name,
        'gnn_type': gnn_type,
        'use_rel_emb': use_rel_emb,
        'pool_type': pool_type,
        'pool_heads': pool_heads,
        'num_params': num_params,
        'best_auc': best_auc,
        'best_auc_pr': best_auc_pr,
        'train_time': train_time,
        'attention_stats': attention_stats,
        'head_diversity': head_diversity
    }


def main(args):
    logging.basicConfig(level=logging.WARNING)  # Reduce logging noise

    print("="*60)
    print("QUICK ABLATION TEST")
    print("="*60)

    # Setup params
    params = argparse.Namespace()
    params.main_dir = os.path.dirname(os.path.abspath(__file__))
    params.dataset = args.dataset
    params.db_path = args.db_path

    params.file_paths = {
        'train': os.path.join(params.main_dir, f'data/{args.dataset}/train.txt'),
        'valid': os.path.join(params.main_dir, f'data/{args.dataset}/valid.txt')
    }

    # Model params
    params.emb_dim = args.emb_dim
    params.num_gcn_layers = args.num_gcn_layers
    params.num_bases = args.num_bases
    params.dropout = 0.0
    params.edge_dropout = 0.5
    params.gnn_type = 'rgcn'
    params.gnn_agg_type = 'sum'
    params.has_attn = True
    params.add_ht_emb = True
    params.rel_emb_dim = 32
    params.attn_rel_emb_dim = 32
    params.pool_dropout = 0.1
    params.comp_fn = 'sub'  # CompGCN default composition function

    # Training params
    params.lr = args.lr
    params.num_epochs = args.num_epochs
    params.batch_size = args.batch_size
    params.num_neg_samples_per_link = 1
    params.add_traspose_rels = False
    params.use_kge_embeddings = False
    params.kge_model = 'TransE'

    # Device
    if torch.cuda.is_available() and not args.cpu:
        params.device = torch.device(f'cuda:{args.gpu}')
    else:
        params.device = torch.device('cpu')
    print(f"Device: {params.device}")

    # Load datasets
    print(f"\nLoading data from: {args.db_path}")
    train_dataset = SubgraphDataset(
        args.db_path, 'train_pos', 'train_neg', params.file_paths,
        add_traspose_rels=False, num_neg_samples_per_link=1,
        use_kge_embeddings=False, dataset=args.dataset,
        kge_model='TransE', file_name='train'
    )

    valid_dataset = SubgraphDataset(
        args.db_path, 'valid_pos', 'valid_neg', params.file_paths,
        add_traspose_rels=False, num_neg_samples_per_link=1,
        use_kge_embeddings=False, dataset=args.dataset,
        kge_model='TransE', file_name='valid'
    )

    # Set model params from dataset
    params.num_rels = train_dataset.num_rels
    params.aug_num_rels = train_dataset.aug_num_rels
    params.inp_dim = train_dataset.n_feat_dim
    params.max_label_value = train_dataset.max_n_label

    # Create relation2id from id2relation
    params.relation2id = {v: k for k, v in train_dataset.id2relation.items()}

    # Create subset for quick testing
    num_train = min(args.num_samples, len(train_dataset))
    num_valid = min(args.num_samples // 2, len(valid_dataset))

    train_indices = np.random.choice(len(train_dataset), num_train, replace=False)
    valid_indices = np.random.choice(len(valid_dataset), num_valid, replace=False)

    train_subset = Subset(train_dataset, train_indices)
    valid_subset = Subset(valid_dataset, valid_indices)

    print(f"Using {num_train} train samples, {num_valid} valid samples")

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_dgl
    )
    valid_loader = DataLoader(
        valid_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_dgl
    )

    # Run configurations
    # Format: (config_name, use_rel_emb, pool_type, pool_heads, gnn_type)
    configs = [
        # Baseline: Mean pooling
        ('1_mean_rel', True, 'mean', 1, 'compgcn'),
        ('2_mean_no_rel', False, 'mean', 1, 'compgcn'),

        # Query attention - Single head (đầy đủ đối chứng)
        ('3_query_1h_rel', True, 'query_attention', 1, 'compgcn'),
        ('4_query_1h_no_rel', False, 'query_attention', 1, 'compgcn'),

        # Query attention - Multi head (đầy đủ đối chứng)
        ('5_query_4h_rel', True, 'query_attention', 4, 'compgcn'),
        ('6_query_4h_no_rel', False, 'query_attention', 4, 'compgcn'),

        # Query attention - 2 heads (để so sánh)
        ('7_query_2h_rel', True, 'query_attention', 2, 'compgcn'),
        ('8_query_2h_no_rel', False, 'query_attention', 2, 'compgcn'),
    ]

    results = []
    for config_name, use_rel_emb, pool_type, pool_heads, gnn_type in configs:
        try:
            result = run_config(params, config_name, use_rel_emb, pool_type, pool_heads, gnn_type, train_loader, valid_loader)
            results.append(result)
        except Exception as e:
            print(f"Error in {config_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "="*120)
    print("SUMMARY")
    print("="*120)
    print(f"{'Config':<25} {'GNN':<10} {'Rel':<5} {'Pool':<18} {'Heads':<6} {'Params':<10} {'AUC':<8} {'AUC-PR':<8} {'Time':<8}")
    print("-"*120)

    for r in results:
        rel_str = 'Yes' if r['use_rel_emb'] else 'No'
        print(f"{r['config']:<25} {r['gnn_type']:<10} {rel_str:<5} {r['pool_type']:<18} {r['pool_heads']:<6} {r['num_params']:<10,} {r['best_auc']:<8.4f} {r['best_auc_pr']:<8.4f} {r['train_time']:<8.1f}s")

    print("="*120)

    # Analysis
    if not results:
        print("No results to analyze!")
        return

    print("\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)

    # Best config
    best = max(results, key=lambda x: x['best_auc'])
    print(f"\n1. BEST CONFIGURATION: {best['config']}")
    print(f"   AUC: {best['best_auc']:.4f}, AUC-PR: {best['best_auc_pr']:.4f}")

    # 2. Effect of relation embedding - DETAILED by pool type
    print(f"\n2. EFFECT OF RELATION EMBEDDING (by pooling type):")
    for pool_type in ['mean', 'query_attention']:
        with_rel = [r for r in results if r['use_rel_emb'] and r['pool_type'] == pool_type]
        without_rel = [r for r in results if not r['use_rel_emb'] and r['pool_type'] == pool_type]
        if with_rel and without_rel:
            for wr in with_rel:
                # Find matching config without rel
                matching = [r for r in without_rel if r['pool_heads'] == wr['pool_heads']]
                if matching:
                    wor = matching[0]
                    diff = (wr['best_auc'] - wor['best_auc']) * 100
                    symbol = "✓" if diff > 0 else "✗"
                    heads_str = f"{wr['pool_heads']}h" if pool_type == 'query_attention' else ""
                    print(f"   {pool_type} {heads_str}: rel={wr['best_auc']:.4f} vs no_rel={wor['best_auc']:.4f} → {diff:+.2f}% {symbol}")

    # 3. Effect of query attention vs mean
    print(f"\n3. EFFECT OF QUERY ATTENTION (vs mean pooling):")
    mean_pool = [r for r in results if r['pool_type'] == 'mean']
    query_pool = [r for r in results if r['pool_type'] == 'query_attention']
    if mean_pool and query_pool:
        # Compare with same rel_emb setting
        for use_rel in [True, False]:
            mean_r = [r for r in mean_pool if r['use_rel_emb'] == use_rel]
            query_r = [r for r in query_pool if r['use_rel_emb'] == use_rel]
            if mean_r and query_r:
                avg_mean = np.mean([r['best_auc'] for r in mean_r])
                avg_query = np.mean([r['best_auc'] for r in query_r])
                diff = (avg_query - avg_mean) * 100
                rel_str = "with_rel" if use_rel else "no_rel"
                symbol = "✓" if diff > 0 else "✗"
                print(f"   {rel_str}: mean={avg_mean:.4f} vs query_attn={avg_query:.4f} → {diff:+.2f}% {symbol}")

    # 4. Effect of number of heads
    print(f"\n4. EFFECT OF NUMBER OF HEADS:")
    for use_rel in [True, False]:
        rel_str = "with_rel" if use_rel else "no_rel"
        heads_results = {}
        for r in results:
            if r['pool_type'] == 'query_attention' and r['use_rel_emb'] == use_rel:
                heads_results[r['pool_heads']] = r['best_auc']
        if heads_results:
            print(f"   {rel_str}:", end="")
            for h in sorted(heads_results.keys()):
                print(f" {h}h={heads_results[h]:.4f}", end="")
            print()

    # 5. Head diversity analysis
    print(f"\n5. HEAD DIVERSITY ANALYSIS:")
    multi_head_results = [r for r in results if r['pool_heads'] > 1 and r.get('head_diversity') is not None]
    if multi_head_results:
        for r in multi_head_results:
            diversity = r.get('head_diversity', 0)
            # Low diversity = heads learning similar things (bad)
            # High diversity = heads learning different things (good)
            status = "GOOD (diverse)" if diversity > 0.1 else "WARNING (similar patterns)"
            print(f"   {r['config']}: diversity={diversity:.4f} - {status}")

            # Show per-head entropy if available
            if r.get('attention_stats') and r['attention_stats'].get('head_stats'):
                for i, hs in enumerate(r['attention_stats']['head_stats']):
                    print(f"      Head {i}: entropy={hs['entropy']:.4f}")
    else:
        print("   No multi-head results with diversity data")

    # 6. Summary table for easy comparison
    print(f"\n6. COMPARISON MATRIX:")
    print(f"   {'Config':<20} {'AUC':<8} {'Params':<10} {'Time':<8}")
    print(f"   {'-'*46}")
    for r in sorted(results, key=lambda x: -x['best_auc']):
        print(f"   {r['config']:<20} {r['best_auc']:<8.4f} {r['num_params']:<10,} {r['train_time']:<8.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick Ablation Test')
    parser.add_argument('--dataset', type=str, default='ogbl-biokg')
    parser.add_argument('--db_path', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples for quick test')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs (use 5-10 for thorough test)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--num_gcn_layers', type=int, default=3)
    parser.add_argument('--num_bases', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (try 0.001 for larger models)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    args = parser.parse_args()
    main(args)
