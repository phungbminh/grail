"""
Create entity_dict.json for type-constrained evaluation on ogbl-biokg.
Downloads entity type information from OGB dataset.
"""
import os
import json
import numpy as np

def create_entity_dict_from_ogb():
    """Create entity_dict.json from OGB dataset."""

    try:
        from ogb.linkproppred import LinkPropPredDataset
    except ImportError:
        print("Installing ogb...")
        os.system("pip install ogb")
        from ogb.linkproppred import LinkPropPredDataset

    print("Loading ogbl-biokg dataset from OGB...")
    dataset = LinkPropPredDataset(name='ogbl-biokg')

    # Get entity type information
    # OGB stores entities with type prefixes
    split_edge = dataset.get_edge_split()

    # The dataset has entity_dict with type -> (num_entities, offset)
    # But we need to map our numeric IDs to types

    # Check if dataset has entity_dict attribute
    if hasattr(dataset, 'entity_dict'):
        entity_dict = dataset.entity_dict
        print(f"Found entity_dict: {entity_dict}")
    else:
        # OGB biokg entities are ordered by type
        # disease, drug, function, protein, sideeffect (alphabetical order)
        entity_counts = {
            'disease': 10687,
            'drug': 10533,
            'function': 45085,
            'protein': 17499,
            'sideeffect': 9969
        }

        # Calculate offsets (entities are stored in alphabetical order by type)
        entity_dict = {}
        offset = 0
        for entity_type in sorted(entity_counts.keys()):
            count = entity_counts[entity_type]
            entity_dict[entity_type] = [count, offset]
            offset += count

        print(f"Created entity_dict: {entity_dict}")

    return entity_dict


def create_entity_dict_from_mapping():
    """
    Create entity_dict.json by analyzing the entities.dict file
    and matching with known OGB structure.
    """
    data_dir = './data/ogbl-biokg'

    # OGB biokg entity counts (from official dataset)
    # Entities are ordered alphabetically by type, then by entity name within type
    entity_counts = {
        'disease': 10687,
        'drug': 10533,
        'function': 45085,
        'protein': 17499,
        'sideeffect': 9969
    }

    # In OGB biokg, entities are stored in this order (alphabetical by type name)
    type_order = ['disease', 'drug', 'function', 'protein', 'sideeffect']

    # Build entity_dict with [count, offset] format
    entity_dict = {}
    offset = 0
    for entity_type in type_order:
        count = entity_counts[entity_type]
        entity_dict[entity_type] = [count, offset]
        print(f"  {entity_type}: count={count}, offset={offset}, range=[{offset}, {offset+count-1}]")
        offset += count

    total = sum(entity_counts.values())
    print(f"\nTotal entities: {total}")

    # Verify with entities.dict
    entities_dict_path = os.path.join(data_dir, 'entities.dict')
    if os.path.exists(entities_dict_path):
        with open(entities_dict_path, 'r') as f:
            num_entities = len(f.readlines())
        print(f"Entities in entities.dict: {num_entities}")

        if num_entities != total:
            print(f"WARNING: Mismatch! entities.dict has {num_entities}, expected {total}")
            print("The entity ordering might be different. Type-constrained evaluation may not work correctly.")

    return entity_dict


def main():
    data_dir = './data/ogbl-biokg'
    output_path = os.path.join(data_dir, 'entity_dict.json')

    print("=" * 60)
    print("Creating entity_dict.json for type-constrained evaluation")
    print("=" * 60)

    # Try to create from local mapping first (faster)
    print("\nCreating entity_dict from OGB structure...")
    entity_dict = create_entity_dict_from_mapping()

    # Save to file
    print(f"\nSaving to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(entity_dict, f, indent=2)

    print("\nDone! entity_dict.json created.")
    print("\nEntity type ranges:")
    for entity_type, (count, offset) in entity_dict.items():
        print(f"  {entity_type}: IDs {offset} to {offset + count - 1} ({count} entities)")

    # Verify format
    print("\nVerifying format...")
    with open(output_path, 'r') as f:
        loaded = json.load(f)
    print(f"Loaded entity_dict with {len(loaded)} types")

    return entity_dict


if __name__ == '__main__':
    main()
