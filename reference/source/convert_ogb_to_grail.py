import torch
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json


def create_entity_type_mapping(split_edge):
    """
    Create entity type mapping from OGB data.
    Returns entity_dict mapping entity types to ID ranges.
    """
    # Collect all unique entity types
    entity_types = set()
    for split in ['train', 'valid', 'test']:
        if split in split_edge:
            data = split_edge[split]
            entity_types.update(data['head_type'])
            entity_types.update(data['tail_type'])

    # Create entity to type mapping
    entity_to_type = {}
    for split in ['train', 'valid', 'test']:
        if split in split_edge:
            data = split_edge[split]
            for i in range(len(data['head'])):
                head_id = data['head'][i].item()
                tail_id = data['tail'][i].item()
                head_type = data['head_type'][i]
                tail_type = data['tail_type'][i]

                entity_to_type[head_id] = head_type
                entity_to_type[tail_id] = tail_type

    # Group entities by type
    type_to_entities = defaultdict(list)
    for entity_id, entity_type in entity_to_type.items():
        type_to_entities[entity_type].append(entity_id)

    # Create entity_dict with type ranges
    entity_dict = {}
    cur_idx = 0
    for entity_type in sorted(type_to_entities.keys()):
        entity_dict[entity_type] = (cur_idx, cur_idx + len(type_to_entities[entity_type]))
        cur_idx += len(type_to_entities[entity_type])

    return entity_dict, entity_to_type


def convert_split_with_types(split_name, input_dir, output_dir, entity_to_type):
    """
    Loads a .pt file for a given split, extracts the triplets with entity types,
    and writes them to a .txt file in the format GraIL expects.
    Format: head\trelation\ttail\thead_type\ttail_type
    """
    input_file = os.path.join(input_dir, f'{split_name}.pt')
    output_file = os.path.join(output_dir, f'{split_name}.txt')

    print(f"Processing {input_file} -> {output_file}...")

    try:
        # Load the .pt file, allowing it to unpickle NumPy objects
        data = torch.load(input_file, weights_only=False)

        # Extract the relevant tensors
        heads = data['head']
        relations = data['relation']
        tails = data['tail']
        head_types = data['head_type']
        tail_types = data['tail_type']

        # Write to the output .txt file with type information
        with open(output_file, 'w') as f:
            for i in tqdm(range(len(heads)), desc=f"Writing {split_name}"):
                h = heads[i].item()
                r = relations[i].item()
                t = tails[i].item()
                ht = head_types[i]
                tt = tail_types[i]
                f.write(f"{h}\t{r}\t{t}\t{ht}\t{tt}\n")

        print(f"Successfully converted {split_name} split with entity types.")

    except Exception as e:
        print(f"An error occurred while processing {split_name}: {e}")


def convert_split(split_name, input_dir, output_dir):
    """
    Legacy function for backward compatibility.
    """
    convert_split_with_types(split_name, input_dir, output_dir, None)

if __name__ == '__main__':
    # Define base paths
    project_root = '/Users/minhbui/Personal/Project/Master/PAPERS/RASG/source/grail'
    input_directory = os.path.join(project_root, 'reference/ogb/examples/linkproppred/biokg/dataset/ogbl_biokg/split/random')
    output_directory = os.path.join(project_root, 'data/ogbl-biokg')

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # First load all splits to create entity type mapping
    print("Creating entity type mapping...")
    from ogb.linkproppred import LinkPropPredDataset
    dataset = LinkPropPredDataset(name='ogbl-biokg')
    split_edge = dataset.get_edge_split()

    entity_dict, entity_to_type = create_entity_type_mapping(split_edge)
    print(f"Entity type mapping created for {len(entity_dict)} entity types:")
    for entity_type, (start, end) in entity_dict.items():
        print(f"  {entity_type}: {start}-{end} ({end-start} entities)")

    # Save entity_dict for use in GraIL
    entity_dict_file = os.path.join(output_directory, 'entity_dict.json')
    with open(entity_dict_file, 'w') as f:
        # Convert tuple keys to strings for JSON serialization
        json_dict = {k: v for k, v in entity_dict.items()}
        json.dump(json_dict, f, indent=2)
    print(f"Entity type mapping saved to {entity_dict_file}")

    # Convert all three splits with entity types
    print("\nConverting splits with entity types...")
    convert_split_with_types('train', input_directory, output_directory, entity_to_type)
    convert_split_with_types('valid', input_directory, output_directory, entity_to_type)
    convert_split_with_types('test', input_directory, output_directory, entity_to_type)

    print("\nData conversion complete with entity type information.")
