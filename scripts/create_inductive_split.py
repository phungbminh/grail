"""
Create Inductive Data Split for GRAIL Evaluation
=================================================

Tạo train/valid/test splits cho inductive evaluation:
- Train/Valid: Chỉ chứa "seen" entities
- Test: Chứa ít nhất 1 "unseen" entity

Usage:
    python scripts/create_inductive_split.py \
        --data_path data/ogbl-biokg-full \
        --output_path data/ogbl-biokg-inductive \
        --unseen_ratio 0.2 \
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


def get_entity_degree(triplets):
    """Tính degree cho mỗi entity"""
    degree = defaultdict(int)
    for h, r, t in triplets:
        degree[h] += 1
        degree[t] += 1
    return degree


def verify_no_overlap(seen_entities, unseen_entities):
    """Verify không có overlap giữa seen và unseen"""
    overlap = seen_entities & unseen_entities
    if overlap:
        print(f"❌ ERROR: Found {len(overlap)} overlapping entities!")
        return False
    print(f"✅ No overlap between seen and unseen entities")
    return True


def verify_triplet_entities(triplets, allowed_entities, split_name):
    """Verify tất cả entities trong triplets đều nằm trong allowed set"""
    invalid_count = 0
    for h, r, t in triplets:
        if h not in allowed_entities or t not in allowed_entities:
            invalid_count += 1

    if invalid_count > 0:
        print(f"❌ ERROR: {split_name} has {invalid_count} triplets with invalid entities!")
        return False
    print(f"✅ {split_name}: All {len(triplets):,} triplets have valid entities")
    return True


def verify_test_has_unseen(triplets, unseen_entities):
    """Verify mỗi triplet trong test có ít nhất 1 unseen entity"""
    invalid_count = 0
    for h, r, t in triplets:
        if h not in unseen_entities and t not in unseen_entities:
            invalid_count += 1

    if invalid_count > 0:
        print(f"❌ ERROR: Test has {invalid_count} triplets without unseen entities!")
        return False
    print(f"✅ Test: All {len(triplets):,} triplets have at least 1 unseen entity")
    return True


def verify_relations_coverage(train_triplets, test_triplets):
    """Verify test relations đều xuất hiện trong train"""
    train_relations = set(r for h, r, t in train_triplets)
    test_relations = set(r for h, r, t in test_triplets)

    missing = test_relations - train_relations
    if missing:
        print(f"⚠️  WARNING: {len(missing)} relations in test not in train: {missing}")
        return False
    print(f"✅ All {len(test_relations)} test relations appear in train")
    return True


def verify_graph_connectivity(triplets, entities, split_name):
    """Verify graph connectivity (basic check)"""
    # Build adjacency
    neighbors = defaultdict(set)
    for h, r, t in triplets:
        neighbors[h].add(t)
        neighbors[t].add(h)

    # Check entities with no connections
    isolated = [e for e in entities if e not in neighbors]
    if isolated:
        print(f"⚠️  WARNING: {split_name} has {len(isolated)} isolated entities")
    else:
        print(f"✅ {split_name}: No isolated entities")

    return True


def print_statistics(name, triplets, entities):
    """In thống kê chi tiết"""
    relations = set(r for h, r, t in triplets)
    heads = set(h for h, r, t in triplets)
    tails = set(t for h, r, t in triplets)

    print(f"\n{'='*50}")
    print(f"Statistics: {name}")
    print(f"{'='*50}")
    print(f"  Triplets:  {len(triplets):,}")
    print(f"  Entities:  {len(entities):,}")
    print(f"  Relations: {len(relations):,}")
    print(f"  Heads:     {len(heads):,}")
    print(f"  Tails:     {len(tails):,}")
    print(f"{'='*50}")


def create_inductive_split(args):
    """Main function để tạo inductive split"""

    print("\n" + "="*60)
    print("CREATING INDUCTIVE DATA SPLIT")
    print("="*60)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"\nRandom seed: {args.seed}")

    # =========================================
    # STEP 1: Load original data
    # =========================================
    print("\n[STEP 1] Loading original data...")

    train_file = os.path.join(args.data_path, 'train.txt')
    valid_file = os.path.join(args.data_path, 'valid.txt')
    test_file = os.path.join(args.data_path, 'test.txt')

    if not all(os.path.exists(f) for f in [train_file, valid_file, test_file]):
        print("❌ ERROR: Missing data files!")
        return False

    original_train = load_triplets(train_file)
    original_valid = load_triplets(valid_file)
    original_test = load_triplets(test_file)

    all_triplets = original_train + original_valid + original_test

    print(f"  Original train: {len(original_train):,} triplets")
    print(f"  Original valid: {len(original_valid):,} triplets")
    print(f"  Original test:  {len(original_test):,} triplets")
    print(f"  Total:          {len(all_triplets):,} triplets")

    # Get all entities
    all_entities = set()
    for h, r, t in all_triplets:
        all_entities.add(h)
        all_entities.add(t)

    print(f"  Total entities: {len(all_entities):,}")

    # =========================================
    # STEP 2: Calculate entity degrees
    # =========================================
    print("\n[STEP 2] Calculating entity degrees...")

    # Degree based on train only (more realistic)
    degree = get_entity_degree(original_train)

    # Sort entities by degree
    entities_by_degree = sorted(all_entities, key=lambda e: degree.get(e, 0))

    print(f"  Min degree: {min(degree.values()) if degree else 0}")
    print(f"  Max degree: {max(degree.values()) if degree else 0}")
    print(f"  Mean degree: {np.mean(list(degree.values())) if degree else 0:.2f}")

    # =========================================
    # STEP 3: Split entities into seen/unseen
    # =========================================
    print(f"\n[STEP 3] Splitting entities (unseen_ratio={args.unseen_ratio})...")

    num_unseen = int(len(all_entities) * args.unseen_ratio)
    num_seen = len(all_entities) - num_unseen

    # Strategy: Chọn unseen từ entities có degree TRUNG BÌNH
    # (không quá cao - quá quan trọng, không quá thấp - quá ít triplets)

    # Lấy entities có degree > 0 (xuất hiện trong train)
    entities_in_train = [e for e in entities_by_degree if degree.get(e, 0) > 0]

    # Chọn từ phần giữa của distribution
    mid_start = len(entities_in_train) // 4
    mid_end = 3 * len(entities_in_train) // 4
    mid_entities = entities_in_train[mid_start:mid_end]

    if len(mid_entities) < num_unseen:
        # Nếu không đủ, lấy thêm từ các entities khác
        mid_entities = entities_in_train

    # Random sample unseen entities
    unseen_entities = set(random.sample(mid_entities, min(num_unseen, len(mid_entities))))
    seen_entities = all_entities - unseen_entities

    print(f"  Seen entities:   {len(seen_entities):,} ({100*len(seen_entities)/len(all_entities):.1f}%)")
    print(f"  Unseen entities: {len(unseen_entities):,} ({100*len(unseen_entities)/len(all_entities):.1f}%)")

    # Verify no overlap
    if not verify_no_overlap(seen_entities, unseen_entities):
        return False

    # =========================================
    # STEP 4: Create new splits
    # =========================================
    print("\n[STEP 4] Creating new train/valid/test splits...")

    # Train: triplets where BOTH h and t are seen
    new_train = []
    for h, r, t in original_train:
        if h in seen_entities and t in seen_entities:
            new_train.append((h, r, t))

    # Valid: triplets where BOTH h and t are seen (from original valid)
    new_valid = []
    for h, r, t in original_valid:
        if h in seen_entities and t in seen_entities:
            new_valid.append((h, r, t))

    # Test: triplets where AT LEAST ONE of h or t is unseen
    new_test = []
    for h, r, t in all_triplets:
        if h in unseen_entities or t in unseen_entities:
            new_test.append((h, r, t))

    print(f"  New train: {len(new_train):,} triplets")
    print(f"  New valid: {len(new_valid):,} triplets")
    print(f"  New test:  {len(new_test):,} triplets")

    # Check if we have enough data
    if len(new_train) < 1000:
        print("❌ ERROR: New train set too small!")
        return False
    if len(new_test) < 100:
        print("❌ ERROR: New test set too small!")
        return False

    # =========================================
    # STEP 5: Verify data integrity
    # =========================================
    print("\n[STEP 5] Verifying data integrity...")

    # Verify train/valid only have seen entities
    if not verify_triplet_entities(new_train, seen_entities, "Train"):
        return False
    if not verify_triplet_entities(new_valid, seen_entities, "Valid"):
        return False

    # Verify test has unseen entities
    if not verify_test_has_unseen(new_test, unseen_entities):
        return False

    # Verify relations coverage
    verify_relations_coverage(new_train, new_test)

    # Verify connectivity
    train_entities = set()
    for h, r, t in new_train:
        train_entities.add(h)
        train_entities.add(t)
    verify_graph_connectivity(new_train, train_entities, "Train")

    # =========================================
    # STEP 6: Analyze test set composition
    # =========================================
    print("\n[STEP 6] Analyzing test set composition...")

    head_unseen = 0
    tail_unseen = 0
    both_unseen = 0

    for h, r, t in new_test:
        h_unseen = h in unseen_entities
        t_unseen = t in unseen_entities

        if h_unseen and t_unseen:
            both_unseen += 1
        elif h_unseen:
            head_unseen += 1
        elif t_unseen:
            tail_unseen += 1

    print(f"  Head unseen only: {head_unseen:,} ({100*head_unseen/len(new_test):.1f}%)")
    print(f"  Tail unseen only: {tail_unseen:,} ({100*tail_unseen/len(new_test):.1f}%)")
    print(f"  Both unseen:      {both_unseen:,} ({100*both_unseen/len(new_test):.1f}%)")

    # =========================================
    # STEP 7: Save data
    # =========================================
    print(f"\n[STEP 7] Saving to {args.output_path}...")

    os.makedirs(args.output_path, exist_ok=True)

    # Save triplets
    save_triplets(new_train, os.path.join(args.output_path, 'train.txt'))
    save_triplets(new_valid, os.path.join(args.output_path, 'valid.txt'))
    save_triplets(new_test, os.path.join(args.output_path, 'test.txt'))

    # Save entity lists
    with open(os.path.join(args.output_path, 'seen_entities.txt'), 'w') as f:
        for e in sorted(seen_entities):
            f.write(f"{e}\n")

    with open(os.path.join(args.output_path, 'unseen_entities.txt'), 'w') as f:
        for e in sorted(unseen_entities):
            f.write(f"{e}\n")

    # Save metadata
    metadata = {
        'original_data_path': args.data_path,
        'unseen_ratio': args.unseen_ratio,
        'seed': args.seed,
        'num_seen_entities': len(seen_entities),
        'num_unseen_entities': len(unseen_entities),
        'num_train_triplets': len(new_train),
        'num_valid_triplets': len(new_valid),
        'num_test_triplets': len(new_test),
        'test_composition': {
            'head_unseen_only': head_unseen,
            'tail_unseen_only': tail_unseen,
            'both_unseen': both_unseen
        }
    }

    with open(os.path.join(args.output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("  ✅ Saved train.txt")
    print("  ✅ Saved valid.txt")
    print("  ✅ Saved test.txt")
    print("  ✅ Saved seen_entities.txt")
    print("  ✅ Saved unseen_entities.txt")
    print("  ✅ Saved metadata.json")

    # =========================================
    # STEP 8: Final verification
    # =========================================
    print("\n[STEP 8] Final verification...")

    # Reload and verify
    reload_train = load_triplets(os.path.join(args.output_path, 'train.txt'))
    reload_valid = load_triplets(os.path.join(args.output_path, 'valid.txt'))
    reload_test = load_triplets(os.path.join(args.output_path, 'test.txt'))

    assert len(reload_train) == len(new_train), "Train mismatch!"
    assert len(reload_valid) == len(new_valid), "Valid mismatch!"
    assert len(reload_test) == len(new_test), "Test mismatch!"

    print("  ✅ All files verified successfully!")

    # Print final statistics
    print_statistics("New Train", new_train, seen_entities)
    print_statistics("New Valid", new_valid, seen_entities)
    print_statistics("New Test", new_test, unseen_entities)

    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "="*60)
    print("INDUCTIVE SPLIT CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nOutput directory: {args.output_path}")
    print(f"\nNext steps:")
    print(f"  1. Extract subgraphs:")
    print(f"     python subgraph_extraction/datasets.py \\")
    print(f"         --dataset ogbl-biokg-inductive \\")
    print(f"         --hop 2 --max_nodes_per_hop 100")
    print(f"\n  2. Train GRAIL:")
    print(f"     python train.py --dataset ogbl-biokg-inductive ...")
    print("="*60)

    return True


def main():
    parser = argparse.ArgumentParser(description='Create Inductive Data Split')
    parser.add_argument('--data_path', type=str, default='data/ogbl-biokg-full',
                        help='Path to original data')
    parser.add_argument('--output_path', type=str, default='data/ogbl-biokg-inductive',
                        help='Path to save inductive split')
    parser.add_argument('--unseen_ratio', type=float, default=0.2,
                        help='Ratio of unseen entities (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    success = create_inductive_split(args)

    if not success:
        print("\n❌ FAILED to create inductive split!")
        sys.exit(1)


if __name__ == '__main__':
    main()
