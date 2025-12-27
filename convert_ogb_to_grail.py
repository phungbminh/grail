"""
Convert OGB ogbl-biokg to GraIL format with CORRECT global entity IDs.
OGB uses per-type local IDs, we need to convert to global IDs.
"""
import os
import json
import numpy as np
from ogb.linkproppred import LinkPropPredDataset

def main():
    print("=" * 60)
    print("Converting OGB ogbl-biokg to GraIL format")
    print("=" * 60)

    # Load OGB dataset
    print("\nLoading OGB dataset...")
    dataset = LinkPropPredDataset(name='ogbl-biokg', root='./ogb_data')
    split_edge = dataset.get_edge_split()

    # Entity type order (alphabetical - OGB convention)
    entity_types = ['disease', 'drug', 'function', 'protein', 'sideeffect']

    # Entity counts from OGB
    entity_counts = {
        'disease': 10687,
        'drug': 10533,
        'function': 45085,
        'protein': 17499,
        'sideeffect': 9969
    }

    # Calculate global ID offsets
    type_to_offset = {}
    offset = 0
    for etype in entity_types:
        type_to_offset[etype] = offset
        offset += entity_counts[etype]

    total_entities = sum(entity_counts.values())
    print(f"\nTotal entities: {total_entities:,}")
    print("\nEntity type offsets:")
    for etype in entity_types:
        print(f"  {etype}: offset={type_to_offset[etype]}, count={entity_counts[etype]}, range=[{type_to_offset[etype]}, {type_to_offset[etype] + entity_counts[etype] - 1}]")

    # Create output directory
    output_dir = './data/ogbl-biokg-full'
    os.makedirs(output_dir, exist_ok=True)

    # Relation mapping
    # Get unique relations from train
    relations = sorted(set(split_edge['train']['relation'].tolist()))
    relation2id = {r: r for r in relations}  # Keep original IDs
    print(f"\nRelations: {len(relations)}")

    # Convert each split
    for split_name in ['train', 'valid', 'test']:
        print(f"\n{'=' * 40}")
        print(f"Converting {split_name} split...")

        split_data = split_edge[split_name]
        heads = split_data['head']
        tails = split_data['tail']
        relations = split_data['relation']
        head_types = split_data['head_type']
        tail_types = split_data['tail_type']

        # Convert to global IDs
        output_file = os.path.join(output_dir, f'{split_name}.txt')
        with open(output_file, 'w') as f:
            for i in range(len(heads)):
                # Get local IDs
                head_local = int(heads[i])
                tail_local = int(tails[i])
                rel = int(relations[i])

                # Get types
                head_type = head_types[i]
                tail_type = tail_types[i]

                # Convert to global IDs
                head_global = type_to_offset[head_type] + head_local
                tail_global = type_to_offset[tail_type] + tail_local

                # Write triplet
                f.write(f"{head_global}\t{rel}\t{tail_global}\n")

        print(f"  Written {len(heads):,} triplets to {output_file}")

    # Create entities.dict
    print(f"\n{'=' * 40}")
    print("Creating entities.dict...")
    entities_file = os.path.join(output_dir, 'entities.dict')
    with open(entities_file, 'w') as f:
        for i in range(total_entities):
            f.write(f"{i}\t{i}\n")
    print(f"  Written {total_entities:,} entities")

    # Create relations.dict
    print("Creating relations.dict...")
    relations_file = os.path.join(output_dir, 'relations.dict')
    with open(relations_file, 'w') as f:
        for r in range(51):  # OGB has 51 relations
            f.write(f"{r}\t{r}\n")
    print(f"  Written 51 relations")

    # Create entity_dict.json for type-constrained evaluation
    print("Creating entity_dict.json...")
    entity_dict = {}
    for etype in entity_types:
        entity_dict[etype] = [entity_counts[etype], type_to_offset[etype]]

    entity_dict_file = os.path.join(output_dir, 'entity_dict.json')
    with open(entity_dict_file, 'w') as f:
        json.dump(entity_dict, f, indent=2)
    print(f"  Saved entity_dict.json")

    # Create entity_to_type.json for type-constrained sampling
    print("Creating entity_to_type.json...")
    entity_to_type = {}
    for etype in entity_types:
        offset = type_to_offset[etype]
        count = entity_counts[etype]
        for i in range(count):
            entity_to_type[offset + i] = etype

    entity_to_type_file = os.path.join(output_dir, 'entity_to_type.json')
    with open(entity_to_type_file, 'w') as f:
        json.dump(entity_to_type, f)
    print(f"  Saved entity_to_type.json ({len(entity_to_type):,} mappings)")

    # Create relation2id.json
    print("Creating relation2id.json...")
    relation2id = {str(r): r for r in range(51)}
    relation2id_file = os.path.join(output_dir, 'relation2id.json')
    with open(relation2id_file, 'w') as f:
        json.dump(relation2id, f, indent=2)
    print(f"  Saved relation2id.json")

    # Verify conversion
    print(f"\n{'=' * 60}")
    print("Verifying conversion...")

    # Check train.txt
    with open(os.path.join(output_dir, 'train.txt')) as f:
        lines = f.readlines()

    print(f"  train.txt: {len(lines):,} triplets")

    # Check entity ID range
    max_entity = 0
    for line in lines[:10000]:
        parts = line.strip().split('\t')
        h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
        max_entity = max(max_entity, h, t)

    print(f"  Max entity ID (sample): {max_entity}")
    print(f"  Expected max: {total_entities - 1}")

    # Show sample triplets
    print("\n  Sample triplets (first 5):")
    for line in lines[:5]:
        parts = line.strip().split('\t')
        h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
        # Determine types
        h_type = None
        t_type = None
        for etype in entity_types:
            offset = type_to_offset[etype]
            count = entity_counts[etype]
            if offset <= h < offset + count:
                h_type = etype
            if offset <= t < offset + count:
                t_type = etype
        print(f"    ({h_type}:{h}, rel:{r}, {t_type}:{t})")

    print(f"\n{'=' * 60}")
    print("Conversion complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
