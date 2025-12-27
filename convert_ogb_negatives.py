"""
Convert OGB pre-generated negatives to global entity IDs.
"""
import os
import torch
import numpy as np

def main():
    print("=" * 60)
    print("Converting OGB negatives to global IDs")
    print("=" * 60)

    # Entity type offsets (same as convert_ogb_to_grail.py)
    entity_types = ['disease', 'drug', 'function', 'protein', 'sideeffect']
    entity_counts = {
        'disease': 10687,
        'drug': 10533,
        'function': 45085,
        'protein': 17499,
        'sideeffect': 9969
    }

    type_to_offset = {}
    offset = 0
    for etype in entity_types:
        type_to_offset[etype] = offset
        offset += entity_counts[etype]

    print("\nEntity type offsets:")
    for etype in entity_types:
        print(f"  {etype}: offset={type_to_offset[etype]}")

    # Load and convert each split
    ogb_path = './ogb_data/ogbl_biokg/split/random'
    output_dir = './data/ogbl-biokg-full'

    for split in ['test', 'valid']:
        print(f"\n{'=' * 40}")
        print(f"Converting {split} negatives...")

        # Load OGB split
        split_data = torch.load(os.path.join(ogb_path, f'{split}.pt'))

        heads = split_data['head']
        tails = split_data['tail']
        relations = split_data['relation']
        head_types = split_data['head_type']
        tail_types = split_data['tail_type']
        head_neg = split_data['head_neg']  # (N, 500)
        tail_neg = split_data['tail_neg']  # (N, 500)

        num_samples = len(heads)
        print(f"  Samples: {num_samples:,}")
        print(f"  head_neg shape: {head_neg.shape}")
        print(f"  tail_neg shape: {tail_neg.shape}")

        # Convert positives to global IDs
        heads_global = np.array([
            type_to_offset[head_types[i]] + int(heads[i])
            for i in range(num_samples)
        ])
        tails_global = np.array([
            type_to_offset[tail_types[i]] + int(tails[i])
            for i in range(num_samples)
        ])

        # Convert negatives to global IDs
        # head_neg contains negative HEADS (same type as original head)
        # tail_neg contains negative TAILS (same type as original tail)
        print("  Converting head_neg to global IDs...")
        head_neg_global = np.zeros_like(head_neg)
        for i in range(num_samples):
            offset = type_to_offset[head_types[i]]
            head_neg_global[i] = head_neg[i] + offset

        print("  Converting tail_neg to global IDs...")
        tail_neg_global = np.zeros_like(tail_neg)
        for i in range(num_samples):
            offset = type_to_offset[tail_types[i]]
            tail_neg_global[i] = tail_neg[i] + offset

        # Save converted data
        converted_data = {
            'head': heads_global,
            'tail': tails_global,
            'relation': np.array(relations),
            'head_type': head_types,
            'tail_type': tail_types,
            'head_neg': head_neg_global,
            'tail_neg': tail_neg_global,
        }

        output_file = os.path.join(output_dir, f'{split}_negatives.pt')
        torch.save(converted_data, output_file)
        print(f"  Saved to {output_file}")

        # Verify
        print(f"\n  Verification:")
        print(f"    heads_global: min={heads_global.min()}, max={heads_global.max()}")
        print(f"    tails_global: min={tails_global.min()}, max={tails_global.max()}")
        print(f"    head_neg_global: min={head_neg_global.min()}, max={head_neg_global.max()}")
        print(f"    tail_neg_global: min={tail_neg_global.min()}, max={tail_neg_global.max()}")

        # Sample
        print(f"\n  Sample (first triplet):")
        print(f"    Original: ({head_types[0]}:{heads[0]}, r{relations[0]}, {tail_types[0]}:{tails[0]})")
        print(f"    Global: ({heads_global[0]}, r{relations[0]}, {tails_global[0]})")
        print(f"    head_neg[0][:5] (local): {head_neg[0][:5]}")
        print(f"    head_neg[0][:5] (global): {head_neg_global[0][:5]}")
        print(f"    tail_neg[0][:5] (local): {tail_neg[0][:5]}")
        print(f"    tail_neg[0][:5] (global): {tail_neg_global[0][:5]}")

    print(f"\n{'=' * 60}")
    print("Conversion complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
