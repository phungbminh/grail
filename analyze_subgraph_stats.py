#!/usr/bin/env python3
"""
Script to analyze subgraph statistics in LMDB cache
Provides detailed statistics on node counts, edge counts, and other metrics
"""
import lmdb
import pickle
import numpy as np
import sys
import os
from tqdm import tqdm
from collections import defaultdict


def analyze_subgraph_stats(db_path, splits=['train', 'valid'], sample_limit=None):
    """
    Analyze subgraph statistics from LMDB database

    Args:
        db_path: Path to LMDB database
        splits: List of splits to analyze (train, valid, test)
        sample_limit: Limit analysis to first N samples (None = all)
    """

    if not os.path.exists(db_path):
        print(f"❌ Database path does not exist: {db_path}")
        return

    print(f"📊 Analyzing subgraph statistics: {db_path}")
    print("=" * 80)

    env = lmdb.open(db_path, readonly=True, max_dbs=10, lock=False)

    all_stats = {}

    for split in splits:
        print(f"\n{'='*80}")
        print(f"📈 Analyzing {split.upper()} split")
        print(f"{'='*80}")

        split_stats = {
            'pos': {'node_counts': [], 'missing': 0, 'valid': 0},
            'neg': {'node_counts': [], 'missing': 0, 'valid': 0}
        }/

        # Analyze positive and negative samples
        for sample_type in ['pos', 'neg']:
            db_name = f'{split}_{sample_type}'

            try:
                db = env.open_db(db_name.encode())

                with env.begin(db=db) as txn:
                    stats = txn.stat()
                    total_entries = stats['entries']

                    # Determine how many to analyze
                    if sample_limit and sample_limit < total_entries:
                        analyze_count = sample_limit
                        print(f"\n🔍 {sample_type.upper()}: Analyzing {analyze_count:,} / {total_entries:,} samples")
                    else:
                        analyze_count = total_entries
                        print(f"\n🔍 {sample_type.upper()}: Analyzing all {total_entries:,} samples")

                    cursor = txn.cursor()
                    cursor.first()

                    for idx in tqdm(range(analyze_count), desc=f"Processing {sample_type}"):
                        if not cursor.key():
                            break

                        data = cursor.value()

                        # Check for missing data
                        if data is None:
                            split_stats[sample_type]['missing'] += 1
                            cursor.next()
                            continue

                        try:
                            # Deserialize
                            data_tuple = pickle.loads(data)

                            # Handle different formats
                            if len(data_tuple) == 7:
                                nodes = data_tuple[0]
                            elif len(data_tuple) == 4:
                                nodes = data_tuple[0]
                            else:
                                print(f"⚠️  Unknown format: {len(data_tuple)} fields")
                                cursor.next()
                                continue

                            # Get node count
                            num_nodes = len(nodes) if hasattr(nodes, '__len__') else 0
                            split_stats[sample_type]['node_counts'].append(num_nodes)
                            split_stats[sample_type]['valid'] += 1

                        except Exception as e:
                            split_stats[sample_type]['missing'] += 1

                        cursor.next()

            except Exception as e:
                print(f"⚠️  Could not analyze {db_name}: {e}")
                continue

        # Print statistics for this split
        print(f"\n{'─'*80}")
        print(f"📊 {split.upper()} STATISTICS")
        print(f"{'─'*80}")

        for sample_type in ['pos', 'neg']:
            stats = split_stats[sample_type]
            node_counts = np.array(stats['node_counts'])

            print(f"\n{sample_type.upper()} samples:")
            print(f"  Valid samples:   {stats['valid']:,}")
            print(f"  Missing samples: {stats['missing']:,}")

            if len(node_counts) > 0:
                print(f"\n  Node count statistics:")
                print(f"    Min:        {node_counts.min():,} nodes")
                print(f"    Max:        {node_counts.max():,} nodes")
                print(f"    Mean:       {node_counts.mean():.2f} nodes")
                print(f"    Median:     {np.median(node_counts):.2f} nodes")
                print(f"    Std Dev:    {node_counts.std():.2f}")

                # Percentiles
                print(f"\n  Percentiles:")
                for p in [10, 25, 50, 75, 90, 95, 99]:
                    print(f"    {p}th:       {np.percentile(node_counts, p):.0f} nodes")

                # Distribution bins
                print(f"\n  Distribution:")
                bins = [0, 2, 4, 6, 8, 10, 20, 50, 100, float('inf')]
                bin_labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-20', '20-50', '50-100', '100+']

                hist, _ = np.histogram(node_counts, bins=bins)
                for label, count in zip(bin_labels, hist):
                    percentage = 100 * count / len(node_counts)
                    bar = '█' * int(percentage / 2)
                    print(f"    {label:>12} nodes: {count:6,} ({percentage:5.2f}%) {bar}")

        all_stats[split] = split_stats

    env.close()

    # Overall summary
    print(f"\n{'='*80}")
    print(f"📋 OVERALL SUMMARY")
    print(f"{'='*80}")

    for split in splits:
        if split not in all_stats:
            continue

        stats = all_stats[split]
        pos_nodes = np.array(stats['pos']['node_counts'])
        neg_nodes = np.array(stats['neg']['node_counts'])

        print(f"\n{split.upper()}:")
        print(f"  Positive: {len(pos_nodes):,} samples, "
              f"{pos_nodes.min():,}-{pos_nodes.max():,} nodes, "
              f"avg {pos_nodes.mean():.1f}")
        print(f"  Negative: {len(neg_nodes):,} samples, "
              f"{neg_nodes.min():,}-{neg_nodes.max():,} nodes, "
              f"avg {neg_nodes.mean():.1f}")

        if stats['pos']['missing'] > 0 or stats['neg']['missing'] > 0:
            print(f"  ⚠️  Missing: {stats['pos']['missing']} pos, {stats['neg']['missing']} neg")

    print(f"\n{'='*80}")

    return all_stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze subgraph statistics in LMDB cache',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all splits (full scan)
  python analyze_subgraph_stats.py --db_path data/ogbl-biokg/subgraphs_en_True_neg_1_hop_2_maxnodes_400

  # Quick analysis (first 10,000 samples only)
  python analyze_subgraph_stats.py --db_path data/ogbl-biokg/subgraphs_... --limit 10000

  # Analyze only training split
  python analyze_subgraph_stats.py --db_path data/ogbl-biokg/subgraphs_... --splits train
        """
    )

    parser.add_argument('--db_path', type=str, required=True,
                        help='Path to LMDB database')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'valid'],
                        help='Which splits to analyze (train, valid, test)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit analysis to first N samples (for quick check)')

    args = parser.parse_args()

    analyze_subgraph_stats(args.db_path, args.splits, args.limit)
