"""
Download ogbl-biokg from OGB and create entity_dict.json for type-constrained evaluation.
"""
import os
import json
import numpy as np
from collections import defaultdict

def main():
    print("=" * 60)
    print("Downloading ogbl-biokg from OGB...")
    print("=" * 60)

    try:
        from ogb.linkproppred import LinkPropPredDataset
    except ImportError:
        print("Installing ogb...")
        os.system("pip install ogb")
        from ogb.linkproppred import LinkPropPredDataset

    # Download dataset
    dataset = LinkPropPredDataset(name='ogbl-biokg', root='./ogb_data')

    # Get graph data
    split_edge = dataset.get_edge_split()

    print("\n" + "=" * 60)
    print("Dataset Info:")
    print("=" * 60)

    # Check entity types
    train_edge = split_edge['train']
    valid_edge = split_edge['valid']
    test_edge = split_edge['test']

    print(f"\nTrain edges: {len(train_edge['head']):,}")
    print(f"Valid edges: {len(valid_edge['head']):,}")
    print(f"Test edges: {len(test_edge['head']):,}")

    # Get entity type information
    print("\n" + "=" * 60)
    print("Entity Types:")
    print("=" * 60)

    # In ogbl-biokg, entities are stored by type
    # train_edge has 'head_type', 'tail_type' fields
    head_types = set(train_edge['head_type'])
    tail_types = set(train_edge['tail_type'])
    all_types = head_types | tail_types

    print(f"Entity types: {sorted(all_types)}")

    # Count entities per type from OGB structure
    # OGB biokg has fixed entity counts
    entity_counts = {
        'disease': 10687,
        'drug': 10533,
        'function': 45085,
        'protein': 17499,
        'sideeffect': 9969
    }

    for etype in sorted(entity_counts.keys()):
        print(f"  {etype}: {entity_counts[etype]:,} entities")

    # Get relation types
    print("\n" + "=" * 60)
    print("Relation Types:")
    print("=" * 60)

    relations = set(train_edge['relation'])
    print(f"Number of relations: {len(relations)}")
    print(f"Relation IDs: {sorted(relations)[:10]}... (showing first 10)")

    # Create entity_dict.json
    # Format: {type_name: [count, offset]}
    print("\n" + "=" * 60)
    print("Creating entity_dict.json...")
    print("=" * 60)

    # Sort types alphabetically (OGB convention)
    sorted_types = sorted(entity_counts.keys())

    entity_dict = {}
    offset = 0
    for etype in sorted_types:
        count = entity_counts[etype]
        entity_dict[etype] = [count, offset]
        print(f"  {etype}: count={count}, offset={offset}")
        offset += count

    total_entities = sum(entity_counts.values())
    print(f"\nTotal entities: {total_entities:,}")

    # Save entity_dict.json
    output_path = './data/ogbl-biokg/entity_dict.json'
    with open(output_path, 'w') as f:
        json.dump(entity_dict, f, indent=2)
    print(f"\nSaved to: {output_path}")

    # Also create a mapping file for reference
    print("\n" + "=" * 60)
    print("Creating entity type mapping...")
    print("=" * 60)

    # Build entity_to_type mapping
    entity_to_type = {}
    for etype, (count, offset) in entity_dict.items():
        for i in range(count):
            entity_to_type[offset + i] = etype

    mapping_path = './data/ogbl-biokg/entity_to_type.json'
    with open(mapping_path, 'w') as f:
        json.dump(entity_to_type, f)
    print(f"Saved entity_to_type mapping to: {mapping_path}")

    # Verify with your current data
    print("\n" + "=" * 60)
    print("Verifying with your current data...")
    print("=" * 60)

    # Check if your entities.dict matches
    entities_dict_path = './data/ogbl-biokg/entities.dict'
    if os.path.exists(entities_dict_path):
        with open(entities_dict_path, 'r') as f:
            your_entities = len(f.readlines())
        print(f"Your entities.dict: {your_entities:,} entities")
        print(f"OGB full dataset: {total_entities:,} entities")

        if your_entities != total_entities:
            print(f"\nWARNING: Your data has {your_entities} entities, OGB has {total_entities}")
            print("Your data might be a subset. Type-constrained evaluation may need adjustment.")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return entity_dict


if __name__ == '__main__':
    main()
