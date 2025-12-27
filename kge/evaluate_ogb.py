"""
Evaluate KGE model using OGB official evaluator (type-constrained)
Usage: python evaluate_ogb.py --model TransE --checkpoint_dir ../experiments/kge_baselines/TransE_ogbl-biokg-full
"""

import argparse
import torch
import numpy as np
from ogb.linkproppred import LinkPropPredDataset, Evaluator
from tqdm import tqdm
import os
import sys


def load_model(checkpoint_dir, num_entities, num_relations):
    """Load trained KGE model from checkpoint"""
    # Load config
    config_path = os.path.join(checkpoint_dir, 'config.json')
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Try to infer from checkpoint
        config = None

    # Find checkpoint file
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Infer model parameters from checkpoint
    entity_embedding = checkpoint['model_state_dict']['entity_embedding']
    relation_embedding = checkpoint['model_state_dict']['relation_embedding']

    hidden_dim = entity_embedding.shape[1]
    double_entity = entity_embedding.shape[1] != relation_embedding.shape[1]
    double_relation = 'relation_embedding' in checkpoint['model_state_dict'] and \
                      relation_embedding.shape[1] > hidden_dim

    return checkpoint, hidden_dim, entity_embedding, relation_embedding


def score_triplets(entity_emb, relation_emb, head_idx, rel_idx, tail_idx, model_name='TransE', gamma=9.0):
    """Score triplets using KGE scoring function"""
    head = entity_emb[head_idx]
    relation = relation_emb[rel_idx]
    tail = entity_emb[tail_idx]

    if model_name == 'TransE':
        # TransE: ||h + r - t||
        score = gamma - torch.norm(head + relation - tail, p=1, dim=-1)
    elif model_name == 'DistMult':
        # DistMult: <h, r, t>
        score = (head * relation * tail).sum(dim=-1)
    elif model_name == 'ComplEx':
        # ComplEx uses complex embeddings
        dim = head.shape[-1] // 2
        re_head, im_head = head[..., :dim], head[..., dim:]
        re_relation, im_relation = relation[..., :dim], relation[..., dim:]
        re_tail, im_tail = tail[..., :dim], tail[..., dim:]

        score = (re_head * re_relation * re_tail +
                 im_head * re_relation * im_tail +
                 re_head * im_relation * im_tail -
                 im_head * im_relation * re_tail).sum(dim=-1)
    elif model_name == 'RotatE':
        # RotatE uses rotation in complex space
        dim = head.shape[-1] // 2
        re_head, im_head = head[..., :dim], head[..., dim:]
        re_tail, im_tail = tail[..., :dim], tail[..., dim:]

        # Relation as rotation
        phase = relation / (10.0 / np.pi)  # embedding_range
        re_relation = torch.cos(phase)
        im_relation = torch.sin(phase)

        re_score = re_head * re_relation - im_head * im_relation - re_tail
        im_score = re_head * im_relation + im_head * re_relation - im_tail

        score = gamma - torch.norm(torch.stack([re_score, im_score], dim=0), dim=0).sum(dim=-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE'])
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--gamma', type=float, default=9.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load OGB dataset
    print("Loading OGB dataset...")
    dataset = LinkPropPredDataset(name='ogbl-biokg')
    split_edge = dataset.get_edge_split()

    # Get entity type offsets for global ID conversion
    # OGB biokg has 5 entity types with their counts
    entity_types = ['disease', 'drug', 'function', 'protein', 'sideeffect']
    type_counts = {
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
        offset += type_counts[etype]

    print(f"Entity type offsets: {type_to_offset}")
    print(f"Total entities: {offset}")

    # Load model checkpoint
    print(f"Loading {args.model} from {args.checkpoint_dir}...")
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint'), map_location='cpu')

    entity_emb = checkpoint['model_state_dict']['entity_embedding'].to(device)
    relation_emb = checkpoint['model_state_dict']['relation_embedding'].to(device)

    print(f"Entity embedding shape: {entity_emb.shape}")
    print(f"Relation embedding shape: {relation_emb.shape}")

    # Setup evaluator
    evaluator = Evaluator(name='ogbl-biokg')

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_triplets = split_edge['test']

    # Get positive triplets and convert to global IDs
    head_type = test_triplets['head_type']
    tail_type = test_triplets['tail_type']
    head_local = test_triplets['head']
    tail_local = test_triplets['tail']
    relation = test_triplets['relation']

    # Pre-generated negatives from OGB
    head_neg = test_triplets['head_neg']  # Shape: [num_test, 500]
    tail_neg = test_triplets['tail_neg']  # Shape: [num_test, 500]

    num_test = len(head_local)
    print(f"Number of test triplets: {num_test}")
    print(f"Negative samples per triplet: {head_neg.shape[1]}")

    # Prepare results
    y_pred_pos_head = []
    y_pred_neg_head = []
    y_pred_pos_tail = []
    y_pred_neg_tail = []

    batch_size = 1000

    with torch.no_grad():
        for start in tqdm(range(0, num_test, batch_size), desc="Evaluating"):
            end = min(start + batch_size, num_test)

            # Convert to global IDs
            h_global = torch.tensor([type_to_offset[head_type[i]] + head_local[i]
                                     for i in range(start, end)], device=device)
            t_global = torch.tensor([type_to_offset[tail_type[i]] + tail_local[i]
                                     for i in range(start, end)], device=device)
            r = torch.tensor(relation[start:end], device=device)

            # Score positive triplets (for head prediction: score (?, r, t))
            pos_score_head = score_triplets(entity_emb, relation_emb,
                                            h_global, r, t_global,
                                            args.model, args.gamma)

            # Score positive triplets (for tail prediction: score (h, r, ?))
            pos_score_tail = score_triplets(entity_emb, relation_emb,
                                            h_global, r, t_global,
                                            args.model, args.gamma)

            y_pred_pos_head.append(pos_score_head.cpu())
            y_pred_pos_tail.append(pos_score_tail.cpu())

            # Score negative heads
            batch_neg_scores_head = []
            for i in range(start, end):
                # Convert negative heads to global IDs (same type as positive head)
                h_neg_global = torch.tensor([type_to_offset[head_type[i]] + neg_id
                                             for neg_id in head_neg[i]], device=device)
                t_rep = t_global[i - start].expand(len(h_neg_global))
                r_rep = r[i - start].expand(len(h_neg_global))

                neg_scores = score_triplets(entity_emb, relation_emb,
                                           h_neg_global, r_rep, t_rep,
                                           args.model, args.gamma)
                batch_neg_scores_head.append(neg_scores.cpu())

            y_pred_neg_head.append(torch.stack(batch_neg_scores_head))

            # Score negative tails
            batch_neg_scores_tail = []
            for i in range(start, end):
                # Convert negative tails to global IDs (same type as positive tail)
                t_neg_global = torch.tensor([type_to_offset[tail_type[i]] + neg_id
                                             for neg_id in tail_neg[i]], device=device)
                h_rep = h_global[i - start].expand(len(t_neg_global))
                r_rep = r[i - start].expand(len(t_neg_global))

                neg_scores = score_triplets(entity_emb, relation_emb,
                                           h_rep, r_rep, t_neg_global,
                                           args.model, args.gamma)
                batch_neg_scores_tail.append(neg_scores.cpu())

            y_pred_neg_tail.append(torch.stack(batch_neg_scores_tail))

    # Concatenate results
    y_pred_pos_head = torch.cat(y_pred_pos_head)  # [num_test]
    y_pred_neg_head = torch.cat(y_pred_neg_head)  # [num_test, 500]
    y_pred_pos_tail = torch.cat(y_pred_pos_tail)  # [num_test]
    y_pred_neg_tail = torch.cat(y_pred_neg_tail)  # [num_test, 500]

    print(f"\nPrediction shapes:")
    print(f"  y_pred_pos_head: {y_pred_pos_head.shape}")
    print(f"  y_pred_neg_head: {y_pred_neg_head.shape}")
    print(f"  y_pred_pos_tail: {y_pred_pos_tail.shape}")
    print(f"  y_pred_neg_tail: {y_pred_neg_tail.shape}")

    # Evaluate using OGB evaluator
    results = evaluator.eval({
        'y_pred_pos': y_pred_pos_head,
        'y_pred_neg': y_pred_neg_head,
    })

    results_tail = evaluator.eval({
        'y_pred_pos': y_pred_pos_tail,
        'y_pred_neg': y_pred_neg_tail,
    })

    # Average head and tail predictions (standard protocol)
    mrr_head = results['mrr_list'].mean().item()
    mrr_tail = results_tail['mrr_list'].mean().item()
    mrr_avg = (mrr_head + mrr_tail) / 2

    hits1_head = results['hits@1_list'].mean().item()
    hits1_tail = results_tail['hits@1_list'].mean().item()
    hits1_avg = (hits1_head + hits1_tail) / 2

    hits3_head = results['hits@3_list'].mean().item()
    hits3_tail = results_tail['hits@3_list'].mean().item()
    hits3_avg = (hits3_head + hits3_tail) / 2

    hits10_head = results['hits@10_list'].mean().item()
    hits10_tail = results_tail['hits@10_list'].mean().item()
    hits10_avg = (hits10_head + hits10_tail) / 2

    print("\n" + "="*60)
    print(f"OGB Evaluation Results for {args.model}")
    print("="*60)
    print(f"Head Prediction - MRR: {mrr_head:.4f}")
    print(f"Tail Prediction - MRR: {mrr_tail:.4f}")
    print(f"Average MRR: {mrr_avg:.4f}")
    print(f"Average Hits@1: {hits1_avg:.4f}")
    print(f"Average Hits@3: {hits3_avg:.4f}")
    print(f"Average Hits@10: {hits10_avg:.4f}")
    print("="*60)

    # Compare with OGB leaderboard
    print("\nOGB Leaderboard Reference:")
    print("  TransE:  MRR=0.7452, Hits@10=0.9457")
    print("  RotatE:  MRR=0.7989, Hits@10=0.9640")
    print("  ComplEx: MRR=0.8095, Hits@10=0.9495")


if __name__ == '__main__':
    main()
