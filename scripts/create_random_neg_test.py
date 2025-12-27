"""
Create Test Set with Random Negatives for Fair GRAIL vs KGE Comparison
======================================================================

Tạo test set với RANDOM negatives (cross-type) thay vì same-type negatives.
Điều này cho phép so sánh công bằng giữa GRAIL và KGE.

Usage:
    python scripts/create_random_neg_test.py \
        --data_path data/ogbl-biokg-full \
        --output_path data/ogbl-biokg-full/test_random_neg \
        --num_negatives 1 \
        --seed 42

Author: GRAIL Evaluation
"""

import argparse
import os
import sys
import json
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_triplets(filepath):
    """Load triplets từ file"""
    triplets = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                h, r, t = parts[0], parts[1], parts[2]
                triplets.append((h, r, t))
    return triplets


def save_triplets(triplets, filepath):
    """Save triplets ra file"""
    with open(filepath, 'w') as f:
        for h, r, t in triplets:
            f.write(f"{h}\t{r}\t{t}\n")


def load_entity_types(data_path):
    """Load entity type mapping"""
    entity_to_type = {}

    # Try loading from entity_to_type.json
    type_file = os.path.join(data_path, 'entity_to_type.json')
    if os.path.exists(type_file):
        with open(type_file, 'r') as f:
            entity_to_type = json.load(f)
            # Convert keys to int if needed
            entity_to_type = {int(k) if k.isdigit() else k: v for k, v in entity_to_type.items()}
        print(f"  Loaded entity types from {type_file}")
        return entity_to_type

    # Try loading from entity_dict.json
    dict_file = os.path.join(data_path, 'entity_dict.json')
    if os.path.exists(dict_file):
        with open(dict_file, 'r') as f:
            entity_dict = json.load(f)
        # entity_dict format: {"type": [count, start_id]}
        for etype, (count, start_id) in entity_dict.items():
            for i in range(count):
                entity_to_type[start_id + i] = etype
        print(f"  Loaded entity types from {dict_file}")
        return entity_to_type

    print("  ⚠️  No entity type file found")
    return entity_to_type


def build_graph_set(triplets):
    """Build set of existing triplets for fast lookup"""
    return set(triplets)


def verify_negatives_are_random(positives, negatives, entity_to_type):
    """Verify negatives are truly random (cross-type)"""
    if not entity_to_type:
        print("  ⚠️  Cannot verify types (no type mapping)")
        return True

    same_type_count = 0
    diff_type_count = 0

    for (ph, pr, pt), (nh, nr, nt) in zip(positives, negatives):
        # Check tail type (assuming tail corruption)
        pt_type = entity_to_type.get(int(pt) if str(pt).isdigit() else pt, 'unknown')
        nt_type = entity_to_type.get(int(nt) if str(nt).isdigit() else nt, 'unknown')

        if pt_type == nt_type:
            same_type_count += 1
        else:
            diff_type_count += 1

    total = same_type_count + diff_type_count
    same_ratio = same_type_count / total if total > 0 else 0

    print(f"\n  Negative type distribution:")
    print(f"    Same type:      {same_type_count:,} ({100*same_type_count/total:.1f}%)")
    print(f"    Different type: {diff_type_count:,} ({100*diff_type_count/total:.1f}%)")

    # For random sampling, same-type should be roughly proportional to type distribution
    # Not 100% like OGB
    if same_ratio > 0.5:
        print(f"  ⚠️  WARNING: High same-type ratio ({100*same_ratio:.1f}%)")
    else:
        print(f"  ✅ Good random distribution")

    return True


def verify_negatives_not_in_graph(negatives, graph_set):
    """Verify no negative triplet exists in graph"""
    in_graph = 0
    for neg in negatives:
        if neg in graph_set:
            in_graph += 1

    if in_graph > 0:
        print(f"  ❌ ERROR: {in_graph} negatives exist in graph!")
        return False

    print(f"  ✅ All {len(negatives):,} negatives are not in graph")
    return True


def verify_entity_coverage(positives, negatives, all_entities):
    """Verify all entities in pos/neg are valid"""
    invalid = 0
    for h, r, t in positives + negatives:
        if h not in all_entities or t not in all_entities:
            invalid += 1

    if invalid > 0:
        print(f"  ❌ ERROR: {invalid} triplets have invalid entities!")
        return False

    print(f"  ✅ All entities are valid")
    return True


def sample_random_negatives(positives, all_entities, graph_set, num_negatives=1,
                           corruption='tail', max_tries=100):
    """
    Sample random negative triplets

    Args:
        positives: List of positive triplets
        all_entities: Set of all entities
        graph_set: Set of existing triplets
        num_negatives: Number of negatives per positive
        corruption: 'head', 'tail', or 'both'
        max_tries: Maximum attempts to find valid negative

    Returns:
        List of negative triplets
    """
    negatives = []
    entity_list = list(all_entities)
    failed = 0

    for h, r, t in tqdm(positives, desc="Sampling negatives"):
        for _ in range(num_negatives):
            found = False
            for _ in range(max_tries):
                if corruption == 'tail':
                    # Corrupt tail
                    new_t = random.choice(entity_list)
                    neg = (h, r, new_t)
                elif corruption == 'head':
                    # Corrupt head
                    new_h = random.choice(entity_list)
                    neg = (new_h, r, t)
                else:  # both
                    # Randomly choose head or tail
                    if random.random() < 0.5:
                        new_t = random.choice(entity_list)
                        neg = (h, r, new_t)
                    else:
                        new_h = random.choice(entity_list)
                        neg = (new_h, r, t)

                # Check if negative is valid (not in graph)
                if neg not in graph_set:
                    negatives.append(neg)
                    found = True
                    break

            if not found:
                failed += 1
                # Use last attempted negative anyway
                negatives.append(neg)

    if failed > 0:
        print(f"  ⚠️  WARNING: {failed} negatives may exist in graph (max_tries reached)")

    return negatives


def create_random_neg_test(args):
    """Main function để tạo random neg test set"""

    print("\n" + "="*60)
    print("CREATING TEST SET WITH RANDOM NEGATIVES")
    print("="*60)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"\nRandom seed: {args.seed}")

    # =========================================
    # STEP 1: Load data
    # =========================================
    print("\n[STEP 1] Loading data...")

    train_file = os.path.join(args.data_path, 'train.txt')
    valid_file = os.path.join(args.data_path, 'valid.txt')
    test_file = os.path.join(args.data_path, 'test.txt')

    train_triplets = load_triplets(train_file)
    valid_triplets = load_triplets(valid_file)
    test_triplets = load_triplets(test_file)

    all_triplets = train_triplets + valid_triplets + test_triplets

    print(f"  Train: {len(train_triplets):,} triplets")
    print(f"  Valid: {len(valid_triplets):,} triplets")
    print(f"  Test:  {len(test_triplets):,} triplets")
    print(f"  Total: {len(all_triplets):,} triplets")

    # Get all entities
    all_entities = set()
    for h, r, t in all_triplets:
        all_entities.add(h)
        all_entities.add(t)

    print(f"  Entities: {len(all_entities):,}")

    # Get all relations
    all_relations = set(r for h, r, t in all_triplets)
    print(f"  Relations: {len(all_relations):,}")

    # =========================================
    # STEP 2: Load entity types
    # =========================================
    print("\n[STEP 2] Loading entity types...")
    entity_to_type = load_entity_types(args.data_path)

    if entity_to_type:
        types = set(entity_to_type.values())
        print(f"  Found {len(types)} entity types: {types}")

    # =========================================
    # STEP 3: Build graph set
    # =========================================
    print("\n[STEP 3] Building graph set for negative filtering...")
    graph_set = build_graph_set(all_triplets)
    print(f"  Graph set size: {len(graph_set):,} triplets")

    # =========================================
    # STEP 4: Sample positives
    # =========================================
    print(f"\n[STEP 4] Selecting positive triplets...")

    if args.max_positives and args.max_positives < len(test_triplets):
        positives = random.sample(test_triplets, args.max_positives)
        print(f"  Sampled {len(positives):,} positives from test set")
    else:
        positives = test_triplets
        print(f"  Using all {len(positives):,} test triplets as positives")

    # =========================================
    # STEP 5: Sample random negatives
    # =========================================
    print(f"\n[STEP 5] Sampling random negatives (corruption={args.corruption})...")

    negatives = sample_random_negatives(
        positives,
        all_entities,
        graph_set,
        num_negatives=args.num_negatives,
        corruption=args.corruption,
        max_tries=args.max_tries
    )

    print(f"  Generated {len(negatives):,} negative triplets")

    # =========================================
    # STEP 6: Verify data integrity
    # =========================================
    print("\n[STEP 6] Verifying data integrity...")

    # Verify negatives not in graph
    if not verify_negatives_not_in_graph(negatives, graph_set):
        print("  ⚠️  Some negatives may be in graph, proceeding anyway...")

    # Verify entity coverage
    verify_entity_coverage(positives, negatives, all_entities)

    # Verify random distribution
    verify_negatives_are_random(positives, negatives, entity_to_type)

    # =========================================
    # STEP 7: Create output directory and save
    # =========================================
    print(f"\n[STEP 7] Saving to {args.output_path}...")

    os.makedirs(args.output_path, exist_ok=True)

    # Save positives
    pos_file = os.path.join(args.output_path, 'positives.txt')
    save_triplets(positives, pos_file)
    print(f"  ✅ Saved positives.txt ({len(positives):,} triplets)")

    # Save negatives
    neg_file = os.path.join(args.output_path, 'negatives.txt')
    save_triplets(negatives, neg_file)
    print(f"  ✅ Saved negatives.txt ({len(negatives):,} triplets)")

    # Save combined (for easy loading)
    # Format: positive triplets first, then negatives
    # Or interleaved: pos1, neg1, pos2, neg2, ...
    combined = []
    labels = []
    for i, pos in enumerate(positives):
        combined.append(pos)
        labels.append(1)
        # Add corresponding negatives
        start_idx = i * args.num_negatives
        end_idx = start_idx + args.num_negatives
        for neg in negatives[start_idx:end_idx]:
            combined.append(neg)
            labels.append(0)

    combined_file = os.path.join(args.output_path, 'test_combined.txt')
    with open(combined_file, 'w') as f:
        for (h, r, t), label in zip(combined, labels):
            f.write(f"{h}\t{r}\t{t}\t{label}\n")
    print(f"  ✅ Saved test_combined.txt ({len(combined):,} triplets)")

    # Save metadata
    metadata = {
        'data_path': args.data_path,
        'num_positives': len(positives),
        'num_negatives_total': len(negatives),
        'num_negatives_per_positive': args.num_negatives,
        'corruption': args.corruption,
        'seed': args.seed,
        'negative_type': 'random_cross_type'
    }

    with open(os.path.join(args.output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ Saved metadata.json")

    # =========================================
    # STEP 8: Final verification
    # =========================================
    print("\n[STEP 8] Final verification...")

    # Reload and verify counts
    reload_pos = load_triplets(pos_file)
    reload_neg = load_triplets(neg_file)

    assert len(reload_pos) == len(positives), "Positive count mismatch!"
    assert len(reload_neg) == len(negatives), "Negative count mismatch!"

    print(f"  ✅ All files verified successfully!")

    # =========================================
    # STEP 9: Statistics
    # =========================================
    print("\n" + "="*50)
    print("STATISTICS")
    print("="*50)
    print(f"  Positives:          {len(positives):,}")
    print(f"  Negatives:          {len(negatives):,}")
    print(f"  Neg per positive:   {args.num_negatives}")
    print(f"  Total samples:      {len(combined):,}")
    print(f"  Corruption mode:    {args.corruption}")

    if entity_to_type:
        # Count type distribution in negatives
        type_counts = defaultdict(int)
        for h, r, t in negatives:
            t_type = entity_to_type.get(int(t) if str(t).isdigit() else t, 'unknown')
            type_counts[t_type] += 1

        print(f"\n  Negative tail type distribution:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {t}: {c:,} ({100*c/len(negatives):.1f}%)")

    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "="*60)
    print("RANDOM NEGATIVE TEST SET CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nOutput directory: {args.output_path}")
    print(f"\nFiles created:")
    print(f"  - positives.txt      ({len(positives):,} triplets)")
    print(f"  - negatives.txt      ({len(negatives):,} triplets)")
    print(f"  - test_combined.txt  ({len(combined):,} triplets with labels)")
    print(f"  - metadata.json")
    print(f"\nNext steps:")
    print(f"  1. Extract subgraphs for this test set")
    print(f"  2. Evaluate GRAIL and KGE models")
    print("="*60)

    return True


def main():
    parser = argparse.ArgumentParser(description='Create Random Negative Test Set')
    parser.add_argument('--data_path', type=str, default='data/ogbl-biokg-full',
                        help='Path to data directory')
    parser.add_argument('--output_path', type=str, default='data/ogbl-biokg-full/test_random_neg',
                        help='Output path for random neg test set')
    parser.add_argument('--num_negatives', type=int, default=1,
                        help='Number of negatives per positive')
    parser.add_argument('--max_positives', type=int, default=None,
                        help='Max positives to use (None = all)')
    parser.add_argument('--corruption', type=str, default='both',
                        choices=['head', 'tail', 'both'],
                        help='Which entity to corrupt')
    parser.add_argument('--max_tries', type=int, default=100,
                        help='Max tries to find valid negative')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    success = create_random_neg_test(args)

    if not success:
        print("\n❌ FAILED to create random negative test set!")
        sys.exit(1)


if __name__ == '__main__':
    main()
