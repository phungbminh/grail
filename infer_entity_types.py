"""
Infer entity types from relation patterns in ogbl-biokg subset.
Uses OGB relation type information to determine entity types.
"""
import os
import json
import numpy as np
from collections import defaultdict

# OGB relation definitions: (head_type, relation_name, tail_type)
# From triplet-type-list.csv
OGB_RELATION_TYPES = [
    ('disease', 'disease-protein', 'protein'),
    ('drug', 'drug-disease', 'disease'),
    ('drug', 'drug-drug_acquired_metabolic_disease', 'drug'),
    ('drug', 'drug-drug_bacterial_infectious_disease', 'drug'),
    ('drug', 'drug-drug_benign_neoplasm', 'drug'),
    ('drug', 'drug-drug_cancer', 'drug'),
    ('drug', 'drug-drug_cardiovascular_system_disease', 'drug'),
    ('drug', 'drug-drug_chromosomal_disease', 'drug'),
    ('drug', 'drug-drug_cognitive_disorder', 'drug'),
    ('drug', 'drug-drug_cryptorchidism', 'drug'),
    ('drug', 'drug-drug_developmental_disorder_of_mental_health', 'drug'),
    ('drug', 'drug-drug_endocrine_system_disease', 'drug'),
    ('drug', 'drug-drug_fungal_infectious_disease', 'drug'),
    ('drug', 'drug-drug_gastrointestinal_system_disease', 'drug'),
    ('drug', 'drug-drug_hematopoietic_system_disease', 'drug'),
    ('drug', 'drug-drug_hematopoietic_system_diseases', 'drug'),
    ('drug', 'drug-drug_hypospadias', 'drug'),
    ('drug', 'drug-drug_immune_system_disease', 'drug'),
    ('drug', 'drug-drug_inherited_metabolic_disorder', 'drug'),
    ('drug', 'drug-drug_integumentary_system_disease', 'drug'),
    ('drug', 'drug-drug_irritable_bowel_syndrome', 'drug'),
    ('drug', 'drug-drug_monogenic_disease', 'drug'),
    ('drug', 'drug-drug_musculoskeletal_system_disease', 'drug'),
    ('drug', 'drug-drug_nervous_system_disease', 'drug'),
    ('drug', 'drug-drug_orofacial_cleft', 'drug'),
    ('drug', 'drug-drug_parasitic_infectious_disease', 'drug'),
    ('drug', 'drug-drug_personality_disorder', 'drug'),
    ('drug', 'drug-drug_polycystic_ovary_syndrome', 'drug'),
    ('drug', 'drug-drug_pre-malignant_neoplasm', 'drug'),
    ('drug', 'drug-drug_psoriatic_arthritis', 'drug'),
    ('drug', 'drug-drug_reproductive_system_disease', 'drug'),
    ('drug', 'drug-drug_respiratory_system_disease', 'drug'),
    ('drug', 'drug-drug_sexual_disorder', 'drug'),
    ('drug', 'drug-drug_sleep_disorder', 'drug'),
    ('drug', 'drug-drug_somatoform_disorder', 'drug'),
    ('drug', 'drug-drug_struct_sim', 'drug'),
    ('drug', 'drug-drug_substance-related_disorder', 'drug'),
    ('drug', 'drug-drug_thoracic_disease', 'drug'),
    ('drug', 'drug-drug_urinary_system_disease', 'drug'),
    ('drug', 'drug-drug_viral_infectious_disease', 'drug'),
    ('drug', 'drug-protein', 'protein'),
    ('drug', 'drug-sideeffect', 'sideeffect'),
    ('function', 'function-function', 'function'),
    ('protein', 'protein-function', 'function'),
    ('protein', 'protein-protein_activation', 'protein'),
    ('protein', 'protein-protein_binding', 'protein'),
    ('protein', 'protein-protein_catalysis', 'protein'),
    ('protein', 'protein-protein_expression', 'protein'),
    ('protein', 'protein-protein_inhibition', 'protein'),
    ('protein', 'protein-protein_ptmod', 'protein'),
    ('protein', 'protein-protein_reaction', 'protein'),
]


def main():
    data_dir = './data/ogbl-biokg'

    print("=" * 60)
    print("Inferring entity types from relation patterns")
    print("=" * 60)

    # Read relations.dict to get relation ID mapping
    relation_id_to_idx = {}
    with open(os.path.join(data_dir, 'relations.dict'), 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation_id_to_idx[int(parts[0])] = int(parts[1])

    print(f"Loaded {len(relation_id_to_idx)} relations")

    # Since we don't have relation names in user's data, we need to match by order
    # OGB relations are ordered alphabetically - check if user's relations match
    # For now, assume relation IDs 0-50 correspond to OGB_RELATION_TYPES in order

    # Read train data to collect entity-relation patterns
    print("\nReading train.txt...")
    entity_head_rels = defaultdict(set)  # entity -> set of (rel, position='head')
    entity_tail_rels = defaultdict(set)  # entity -> set of (rel, position='tail')

    with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                head, rel, tail = int(parts[0]), int(parts[1]), int(parts[2])
                entity_head_rels[head].add(rel)
                entity_tail_rels[tail].add(rel)

    print(f"  Found {len(entity_head_rels)} entities as head")
    print(f"  Found {len(entity_tail_rels)} entities as tail")

    # Read entities.dict to get all entity IDs
    entities = set()
    with open(os.path.join(data_dir, 'entities.dict'), 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                entities.add(int(parts[0]))

    print(f"  Total entities: {len(entities)}")

    # Build relation ID to (head_type, tail_type) mapping
    # Assume OGB relation order matches user's relation order
    rel_to_types = {}
    for i, (head_type, rel_name, tail_type) in enumerate(OGB_RELATION_TYPES):
        rel_to_types[i] = (head_type, tail_type)

    print(f"\nBuilt relation-to-types mapping for {len(rel_to_types)} relations")

    # Infer entity types
    print("\nInferring entity types...")
    entity_to_type = {}
    conflicts = 0
    unassigned = 0

    for entity in entities:
        possible_types = set()

        # Check relations where this entity is head
        for rel in entity_head_rels.get(entity, []):
            if rel in rel_to_types:
                head_type, _ = rel_to_types[rel]
                possible_types.add(head_type)

        # Check relations where this entity is tail
        for rel in entity_tail_rels.get(entity, []):
            if rel in rel_to_types:
                _, tail_type = rel_to_types[rel]
                possible_types.add(tail_type)

        if len(possible_types) == 1:
            entity_to_type[entity] = list(possible_types)[0]
        elif len(possible_types) > 1:
            # Conflict - multiple possible types
            # This can happen if entity participates in relations with different types
            # Choose by majority or first found
            type_counts = defaultdict(int)
            for rel in entity_head_rels.get(entity, []):
                if rel in rel_to_types:
                    head_type, _ = rel_to_types[rel]
                    type_counts[head_type] += 1
            for rel in entity_tail_rels.get(entity, []):
                if rel in rel_to_types:
                    _, tail_type = rel_to_types[rel]
                    type_counts[tail_type] += 1

            best_type = max(type_counts.keys(), key=lambda t: type_counts[t])
            entity_to_type[entity] = best_type
            conflicts += 1
        else:
            # No relations found for this entity
            unassigned += 1

    print(f"  Assigned types: {len(entity_to_type)}")
    print(f"  Conflicts (multiple types): {conflicts}")
    print(f"  Unassigned (no relations): {unassigned}")

    # Count entities per type
    type_counts = defaultdict(list)
    for entity, etype in entity_to_type.items():
        type_counts[etype].append(entity)

    print("\nEntity type distribution:")
    for etype in sorted(type_counts.keys()):
        print(f"  {etype}: {len(type_counts[etype])} entities")

    # Build entity_dict.json with [count, offset] format
    # Sort types alphabetically, then assign offsets
    print("\n" + "=" * 60)
    print("Creating entity_dict.json...")
    print("=" * 60)

    sorted_types = sorted(type_counts.keys())
    entity_dict = {}
    offset = 0

    # Also create entity_to_global_id mapping
    entity_to_global_id = {}  # local_id -> global_id in sorted order
    global_id = 0

    for etype in sorted_types:
        entities_of_type = sorted(type_counts[etype])
        count = len(entities_of_type)
        entity_dict[etype] = [count, offset]
        print(f"  {etype}: count={count}, offset={offset}, range=[{offset}, {offset+count-1}]")

        for local_id in entities_of_type:
            entity_to_global_id[local_id] = global_id
            global_id += 1

        offset += count

    total = sum(len(v) for v in type_counts.values())
    print(f"\nTotal typed entities: {total}")

    # Handle unassigned entities
    if unassigned > 0:
        unassigned_entities = [e for e in entities if e not in entity_to_type]
        print(f"\nWARNING: {len(unassigned_entities)} entities have no type assigned!")
        print(f"  First 10 unassigned: {unassigned_entities[:10]}")
        # Assign to 'unknown' type
        entity_dict['unknown'] = [len(unassigned_entities), offset]
        for local_id in sorted(unassigned_entities):
            entity_to_type[local_id] = 'unknown'
            entity_to_global_id[local_id] = global_id
            global_id += 1

    # Save entity_dict.json
    output_path = os.path.join(data_dir, 'entity_dict.json')
    with open(output_path, 'w') as f:
        json.dump(entity_dict, f, indent=2)
    print(f"\nSaved entity_dict.json to {output_path}")

    # Save entity_to_type mapping
    entity_to_type_path = os.path.join(data_dir, 'entity_to_type.json')
    with open(entity_to_type_path, 'w') as f:
        json.dump({str(k): v for k, v in entity_to_type.items()}, f)
    print(f"Saved entity_to_type.json to {entity_to_type_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return entity_dict, entity_to_type


if __name__ == '__main__':
    main()
