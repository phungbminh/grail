#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np
import torch

from torch.utils.data import Dataset


def load_entity_dict(data_path):
    """
    Load entity_dict.json for type-constrained negative sampling (OGB standard).
    Returns:
        entity_to_type: dict mapping entity_id -> type_name
        type_to_entities: dict mapping type_name -> numpy array of entity_ids
    """
    entity_dict_path = os.path.join(data_path, 'entity_dict.json')
    if not os.path.exists(entity_dict_path):
        return None, None

    with open(entity_dict_path, 'r') as f:
        entity_dict = json.load(f)

    entity_to_type = {}
    type_to_entities = {}

    for entity_type, (count, offset) in entity_dict.items():
        entities = np.arange(offset, offset + count, dtype=np.int64)
        type_to_entities[entity_type] = entities
        for eid in range(offset, offset + count):
            entity_to_type[eid] = entity_type

    return entity_to_type, type_to_entities


class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode,
                 entity_to_type=None, type_to_entities=None):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

        # Type-constrained sampling (OGB standard)
        self.entity_to_type = entity_to_type
        self.type_to_entities = type_to_entities
        self.use_type_constraint = entity_to_type is not None
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        # Type-constrained sampling (OGB standard)
        if self.use_type_constraint:
            if self.mode == 'head-batch':
                # Sample from entities of the same type as head
                head_type = self.entity_to_type[head]
                candidate_entities = self.type_to_entities[head_type]
            elif self.mode == 'tail-batch':
                # Sample from entities of the same type as tail
                tail_type = self.entity_to_type[tail]
                candidate_entities = self.type_to_entities[tail_type]
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            n_candidates = len(candidate_entities)
        else:
            candidate_entities = None
            n_candidates = self.nentity

        while negative_sample_size < self.negative_sample_size:
            if self.use_type_constraint:
                # Sample indices from candidate entities
                sample_indices = np.random.randint(n_candidates, size=self.negative_sample_size*2)
                negative_sample = candidate_entities[sample_indices]
            else:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)

            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.from_numpy(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode,
                 entity_to_type=None, type_to_entities=None):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

        # Type-constrained evaluation (OGB standard)
        self.entity_to_type = entity_to_type
        self.type_to_entities = type_to_entities
        self.use_type_constraint = entity_to_type is not None

        # Pre-compute true triples index for fast lookup (OPTIMIZED)
        self.true_head = {}  # (relation, tail) -> set of heads
        self.true_tail = {}  # (head, relation) -> set of tails

        for h, r, t in all_true_triples:
            if (r, t) not in self.true_head:
                self.true_head[(r, t)] = set()
            self.true_head[(r, t)].add(h)

            if (h, r) not in self.true_tail:
                self.true_tail[(h, r)] = set()
            self.true_tail[(h, r)].add(t)

        # Pre-compute all entity indices as numpy array
        self.all_entities = np.arange(nentity, dtype=np.int64)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.use_type_constraint:
            # Type-constrained evaluation (OGB standard)
            if self.mode == 'head-batch':
                head_type = self.entity_to_type[head]
                candidate_entities = self.type_to_entities[head_type]
                n_candidates = len(candidate_entities)

                # Filter bias for type-constrained candidates only
                filter_bias = np.zeros(n_candidates, dtype=np.float32)
                true_heads = self.true_head.get((relation, tail), set())

                # Find position of target and true entities in candidates
                candidate_set = set(candidate_entities)
                for th in true_heads:
                    if th in candidate_set:
                        pos = np.searchsorted(candidate_entities, th)
                        filter_bias[pos] = -1.0

                # Unmask the actual head
                head_pos = np.searchsorted(candidate_entities, head)
                filter_bias[head_pos] = 0.0

                negative_sample = candidate_entities.copy()

            elif self.mode == 'tail-batch':
                tail_type = self.entity_to_type[tail]
                candidate_entities = self.type_to_entities[tail_type]
                n_candidates = len(candidate_entities)

                # Filter bias for type-constrained candidates only
                filter_bias = np.zeros(n_candidates, dtype=np.float32)
                true_tails = self.true_tail.get((head, relation), set())

                # Find position of target and true entities in candidates
                candidate_set = set(candidate_entities)
                for tt in true_tails:
                    if tt in candidate_set:
                        pos = np.searchsorted(candidate_entities, tt)
                        filter_bias[pos] = -1.0

                # Unmask the actual tail
                tail_pos = np.searchsorted(candidate_entities, tail)
                filter_bias[tail_pos] = 0.0

                negative_sample = candidate_entities.copy()
            else:
                raise ValueError('negative batch mode %s not supported' % self.mode)
        else:
            # Standard evaluation (rank against all entities)
            filter_bias = np.zeros(self.nentity, dtype=np.float32)

            if self.mode == 'head-batch':
                true_heads = self.true_head.get((relation, tail), set())
                if true_heads:
                    filter_bias[list(true_heads)] = -1.0
                filter_bias[head] = 0.0
                negative_sample = self.all_entities.copy()
            elif self.mode == 'tail-batch':
                true_tails = self.true_tail.get((head, relation), set())
                if true_tails:
                    filter_bias[list(true_tails)] = -1.0
                filter_bias[tail] = 0.0
                negative_sample = self.all_entities.copy()
            else:
                raise ValueError('negative batch mode %s not supported' % self.mode)

        positive_sample = torch.LongTensor((head, relation, tail))
        negative_sample = torch.from_numpy(negative_sample)
        filter_bias = torch.from_numpy(filter_bias)

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        mode = data[0][3]

        # Check if all negative samples have the same size
        neg_sizes = [_[1].size(0) for _ in data]
        if len(set(neg_sizes)) == 1:
            # Same size - use standard stacking
            negative_sample = torch.stack([_[1] for _ in data], dim=0)
            filter_bias = torch.stack([_[2] for _ in data], dim=0)
        else:
            # Variable sizes (type-constrained) - pad to max size
            max_size = max(neg_sizes)
            batch_size = len(data)

            negative_sample = torch.zeros(batch_size, max_size, dtype=torch.long)
            filter_bias = torch.full((batch_size, max_size), -float('inf'), dtype=torch.float32)

            for i, (_, neg, fb, _) in enumerate(data):
                size = neg.size(0)
                negative_sample[i, :size] = neg
                filter_bias[i, :size] = fb

        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data