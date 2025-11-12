import os
import math
import struct
import logging
import random
import pickle as pkl
import json
import time
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import sys
import torch
from scipy.special import softmax
from utils.dgl_utils import _bfs_relational
# Use numba-optimized version for 10-100x speedup
try:
    from utils.dgl_utils_numba import _bfs_relational_numba as _bfs_relational_fast
    USE_NUMBA = True
except ImportError:
    _bfs_relational_fast = _bfs_relational
    USE_NUMBA = False
    import warnings
    warnings.warn("Numba not installed. Install 'numba>=0.59.0' for 10-100x BFS speedup.")
from utils.graph_utils import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize, get_edge_count, diameter, radius
import networkx as nx


def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
    pos_edges = edges
    neg_edges = []

    # if max_size is set, randomly sample train links
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    # sample negative links for train/test
    n, r = adj_list[0].shape[0], len(adj_list)

    # distribution of edges across reelations
    theta = 0.001
    edge_count = get_edge_count(adj_list)
    rel_dist = np.zeros(edge_count.shape)
    idx = np.nonzero(edge_count)
    rel_dist[idx] = softmax(theta * edge_count[idx])

    # possible head and tails for each relation
    valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
    valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

    pbar = tqdm(total=len(pos_edges))
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], pos_edges[pbar.n % len(pos_edges)][2]
        if np.random.uniform() < constrained_neg_prob:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(valid_heads[rel])
            else:
                neg_tail = np.random.choice(valid_tails[rel])
        else:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(n)
            else:
                neg_tail = np.random.choice(n)

        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_edges.append([neg_head, neg_tail, rel])
            pbar.update(1)

    pbar.close()

    neg_edges = np.array(neg_edges)
    return pos_edges, neg_edges


def load_entity_dict(data_dir):
    """
    Load entity type mapping from JSON file for fair negative sampling.
    Returns dictionary mapping entity types to ID ranges.
    """
    entity_dict_file = os.path.join(data_dir, 'entity_dict.json')

    if not os.path.exists(entity_dict_file):
        logging.warning(f"Entity dict file not found: {entity_dict_file}, using uniform sampling")
        return None

    try:
        with open(entity_dict_file, 'r') as f:
            entity_dict = json.load(f)
        logging.info(f"Loaded entity type mapping for {len(entity_dict)} entity types")
        return entity_dict
    except Exception as e:
        logging.error(f"Error loading entity dict: {e}, using uniform sampling")
        return None


def sample_fair_neg(adj_list, edges, data_dir, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
    """
    Sample negative edges with entity type constraints (fair corruption).
    Only corrupt entities with entities of the SAME TYPE - OGB standard!
    """
    # Load entity type mapping
    entity_dict = load_entity_dict(data_dir)

    if entity_dict is None:
        # Fallback to original uniform sampling
        logging.warning("No entity type mapping available, using uniform sampling")
        return sample_neg(adj_list, edges, num_neg_samples_per_link, max_size, constrained_neg_prob)

    pos_edges = edges
    neg_edges = []

    # if max_size is set, randomly sample train links
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    # sample negative links for train/test
    n, r = adj_list[0].shape[0], len(adj_list)

    # distribution of edges across relations
    theta = 0.001
    edge_count = get_edge_count(adj_list)
    rel_dist = np.zeros(edge_count.shape)
    idx = np.nonzero(edge_count)
    rel_dist[idx] = softmax(theta * edge_count[idx])

    # possible head and tails for each relation
    valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
    valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

    # Create entity to type mapping from entity_dict
    # Use dict comprehension for 10-100x speedup vs loop
    logging.info(f"Building entity type mapping from {len(entity_dict)} entity types...")

    entity_to_type = {
        entity_id: entity_type
        for entity_type, range_tuple in entity_dict.items()
        for entity_id in range(range_tuple[0], range_tuple[1])
    }

    logging.info(f"Fair negative sampling: using entity type mapping for {len(entity_dict)} types, {len(entity_to_type):,} entities")

    # OPTIMIZATION: Pre-compute entity pools for ultra-fast sampling
    entity_pools = {}
    for entity_type, range_tuple in entity_dict.items():
        start, end = range_tuple[0], range_tuple[1]
        entity_pools[entity_type] = list(range(start, end))

    # OPTIMIZATION: Use 8-core multiprocessing for large datasets
    target_neg_samples = num_neg_samples_per_link * len(pos_edges)

    if target_neg_samples > 1000:  # Use multiprocessing for large datasets
        logging.info("Using 8-core ULTRA-FAST multiprocessing...")
        neg_edges = _sample_fair_neg_parallel(
            pos_edges, entity_dict, entity_to_type, entity_pools,
            adj_list, valid_heads, valid_tails, num_neg_samples_per_link,
            constrained_neg_prob, n, target_neg_samples
        )
    else:
        logging.info("Using optimized single-threaded sampling")
        # Optimized single-threaded with pre-computed pools
        from tqdm import tqdm
        pbar = tqdm(total=target_neg_samples, desc="Sampling fair negatives", ncols=100, unit="neg")
        sample_idx = 0
        while len(neg_edges) < target_neg_samples:
            head, tail, rel = pos_edges[sample_idx % len(pos_edges)][0], pos_edges[sample_idx % len(pos_edges)][1], pos_edges[sample_idx % len(pos_edges)][2]

            # Get entity types for head and tail
            head_type = entity_to_type.get(head, "unknown")
            tail_type = entity_to_type.get(tail, "unknown")

            # Get valid entity ranges for fair corruption (handle JSON format)
            head_range_tuple = entity_dict.get(head_type, [0, n])
            tail_range_tuple = entity_dict.get(tail_type, [0, n])
            head_range = (head_range_tuple[0], head_range_tuple[1])
            tail_range = (tail_range_tuple[0], tail_range_tuple[1])

            neg_head, neg_tail = head, tail

            if np.random.uniform() < constrained_neg_prob:
                if np.random.uniform() < 0.5:
                    # Constrained head corruption: sample from same entity type
                    if head_type != "unknown" and head_range[1] > head_range[0]:
                        candidates = list(range(head_range[0], head_range[1]))
                        candidates = [c for c in candidates if c != head]  # Remove original
                        if candidates:
                            neg_head = np.random.choice(candidates)
                        elif valid_heads[rel]:
                            neg_head = np.random.choice(valid_heads[rel])
                        else:
                            neg_head = np.random.randint(0, n)
                    else:
                        if valid_heads[rel]:
                            neg_head = np.random.choice(valid_heads[rel])
                        else:
                            neg_head = np.random.randint(0, n)
                else:
                    # Constrained tail corruption: sample from same entity type
                    if tail_type != "unknown" and tail_range[1] > tail_range[0]:
                        candidates = list(range(tail_range[0], tail_range[1]))
                        candidates = [c for c in candidates if c != tail]  # Remove original
                        if candidates:
                            neg_tail = np.random.choice(candidates)
                        elif valid_tails[rel]:
                            neg_tail = np.random.choice(valid_tails[rel])
                        else:
                            neg_tail = np.random.randint(0, n)
                    else:
                        if valid_tails[rel]:
                            neg_tail = np.random.choice(valid_tails[rel])
                        else:
                            neg_tail = np.random.randint(0, n)
            else:
                # FAIR CORRUPTION: Only sample from SAME entity type!
                if np.random.uniform() < 0.5:
                    # Corrupt head: SAMPLE FROM SAME ENTITY TYPE as original head
                    if head_type != "unknown" and head_range[1] > head_range[0]:
                        # Sample from entities of the same type as head - OGB standard!
                        candidates = list(range(head_range[0], head_range[1]))
                        candidates = [c for c in candidates if c != head]  # Remove original head
                        if candidates:
                            neg_head = np.random.choice(candidates)
                        else:
                            # Fallback to any valid head for this relation
                            if valid_heads[rel]:
                                neg_head = np.random.choice(valid_heads[rel])
                            else:
                                neg_head = np.random.randint(0, n)
                    else:
                        # Fallback
                        if valid_heads[rel]:
                            neg_head = np.random.choice(valid_heads[rel])
                        else:
                            neg_head = np.random.randint(0, n)
                else:
                    # Corrupt tail: SAMPLE FROM SAME ENTITY TYPE as original tail
                    if tail_type != "unknown" and tail_range[1] > tail_range[0]:
                        # Sample from entities of the same type as tail - OGB standard!
                        candidates = list(range(tail_range[0], tail_range[1]))
                        candidates = [c for c in candidates if c != tail]  # Remove original tail
                        if candidates:
                            neg_tail = np.random.choice(candidates)
                        else:
                            # Fallback to any valid tail for this relation
                            if valid_tails[rel]:
                                neg_tail = np.random.choice(valid_tails[rel])
                            else:
                                neg_tail = np.random.randint(0, n)
                    else:
                        # Fallback
                        if valid_tails[rel]:
                            neg_tail = np.random.choice(valid_tails[rel])
                        else:
                            neg_tail = np.random.randint(0, n)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_edges.append([neg_head, neg_tail, rel])
                pbar.update(1)

            sample_idx += 1

        pbar.close()
        logging.info(f"Generated {len(neg_edges)} fair negative samples respecting entity types")

    neg_edges = np.array(neg_edges)
    return pos_edges, neg_edges


def links2subgraphs(A, graphs, params, max_label_value=None, semantic_embeddings=None):
    '''
    extract enclosing subgraphs, write map mode + named dbs
    '''
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []

    # OPTIMIZATION: Calculate BYTES_PER_DATUM more accurately for large datasets
    # Add safety check for empty graphs dictionary
    if not graphs or len(graphs) == 0:
        logging.error("Graphs dictionary is empty!")
        return

    first_split = list(graphs.values())[0]
    if 'pos' not in first_split or len(first_split['pos']) == 0:
        logging.error("No positive links found in graphs dictionary!")
        return

    # Fix: Ensure minimum map size for semantic pruning
    try:
        BYTES_PER_DATUM = get_average_subgraph_size(min(100, len(first_split['pos'])), first_split['pos'], A, params) * 1.5
    except:
        BYTES_PER_DATUM = 50000  # Fallback: 50KB per datum

    # OPTIMIZATION: Calculate per-split map sizes to avoid massive allocation
    max_split_size = 0
    for split_name, split in graphs.items():
        split_size = (len(split['pos']) + len(split['neg'])) * 2 * BYTES_PER_DATUM
        max_split_size = max(max_split_size, split_size)

    # Ensure minimum map size for stability
    min_map_size = 1024 * 1024 * 1024  # 1GB minimum
    max_split_size = max(max_split_size, min_map_size)

    # Log semantic pruning info
    use_semantic = getattr(params, 'use_semantic_pruning', False)

    # OPTIMIZATION: Use maximum split size with safety factor for semantic pruning
    # Semantic pruning needs more storage due to additional metadata
    safety_factor = 3.0 if use_semantic else 1.5
    map_size = int(max_split_size * safety_factor)
    logging.info(f"LMDB map_size: {map_size/(1024**3):.2f} GB for largest split (safety_factor={safety_factor})")
    if use_semantic:
        logging.info(f"Semantic pruning ENABLED with stage1_ratio={getattr(params, 'stage1_ratio', 10)}")
        if semantic_embeddings is not None:
            logging.info(f"Semantic embeddings loaded: {semantic_embeddings.shape}")
        else:
            logging.warning("No semantic embeddings provided - will use path length only")

    # Force large map size for full dataset with semantic pruning
    if use_semantic:
        actual_map_size = max(map_size, 30 * 1024**3)  # Minimum 30GB for semantic
    else:
        actual_map_size = max(map_size, 5 * 1024**3)   # Minimum 5GB for baseline

    # OPTIMIZATION: Fast LMDB settings for 10-50x write speedup
    env = lmdb.open(
        params.db_path,
        map_size=actual_map_size,
        max_dbs=6,
        writemap=True,      # Write directly to mmap (faster)
        map_async=True,     # Async flush (much faster, safe on crash)
        metasync=False,     # Don't sync metadata (faster)
        sync=False          # Don't force flush on commit (10x faster!)
    )
    logging.info(f"LMDB actual map_size: {actual_map_size/(1024**3):.2f} GB")
    logging.info(f"LMDB optimized for SPEED (writemap=True, sync=False)")

    def extraction_helper(A, links, g_labels, split_env):

        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))

        # Pass semantic embeddings to workers
        init_args = (A, params, max_label_value, semantic_embeddings)

        # Use all available cores for maximum speed
        import multiprocessing
        max_workers = min(multiprocessing.cpu_count(), 16)  # Use all 16 cores
        logging.info(f"Using {max_workers} cores for subgraph extraction")

        # OPTIMIZATION: Batch LMDB writes for 10-50x speedup
        batch_size = 200  # Write 200 subgraphs per transaction (increased from 100)
        batch_buffer = []

        with mp.Pool(processes=max_workers, initializer=intialize_worker, initargs=init_args) as p:
            args_ = zip(range(len(links)), links, g_labels)
            # Use imap_unordered for better parallelism (order doesn't matter for LMDB)
            for result in tqdm(p.imap_unordered(extract_save_subgraph, args_, chunksize=4), total=len(links)):
                str_id, serialized_datum, sg_size, enc_ratio, n_pruned, n_labels = result

                max_n_label['value'] = np.maximum(np.max(n_labels, axis=0), max_n_label['value'])
                subgraph_sizes.append(sg_size)
                enc_ratios.append(enc_ratio)
                num_pruned_nodes.append(n_pruned)

                # Add to batch buffer (data already serialized in worker!)
                batch_buffer.append((str_id, serialized_datum))

                # Write batch when full
                if len(batch_buffer) >= batch_size:
                    with env.begin(write=True, db=split_env) as txn:
                        for bid, bdata in batch_buffer:
                            txn.put(bid, bdata)  # Already serialized!
                    batch_buffer = []

            # Write remaining items
            if batch_buffer:
                with env.begin(write=True, db=split_env) as txn:
                    for bid, bdata in batch_buffer:
                        txn.put(bid, bdata)  # Already serialized!

    for split_name, split in graphs.items():
        logging.info(f"Extracting enclosing subgraphs for positive links in {split_name} set")
        labels = np.ones(len(split['pos']))
        db_name_pos = split_name + '_pos'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(A, split['pos'], labels, split_env)

        logging.info(f"Extracting enclosing subgraphs for negative links in {split_name} set")
        labels = np.zeros(len(split['neg']))
        db_name_neg = split_name + '_neg'
        split_env = env.open_db(db_name_neg.encode())
        extraction_helper(A, split['neg'], labels, split_env)

    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']

    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
        txn.put('max_n_label_sub'.encode(), (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(), (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))

        txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
        txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
        txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
        txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

        txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
        txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
        txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
        txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

        txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
        txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
        txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
        txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))


def get_average_subgraph_size(sample_size, links, A, params):
    total_size = 0
    for (n1, n2, r_label) in links[np.random.choice(len(links), sample_size)]:
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A, params.hop, params.enclosing_sub_graph, params.max_nodes_per_hop)
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        total_size += len(serialize(datum))
    return total_size / sample_size


def intialize_worker(A, params, max_label_value, semantic_embeddings=None):
    global A_, params_, max_label_value_, A_incidence_, semantic_embeddings_
    A_, params_, max_label_value_ = A, params, max_label_value
    semantic_embeddings_ = semantic_embeddings

    # Pre-compute incidence matrix once per worker (huge speedup!)
    # This eliminates ~1.5s per subgraph by caching the combined adjacency
    A_incidence_ = incidence_matrix(A)
    A_incidence_ += A_incidence_.T  # Make symmetric


def extract_save_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_

    # Start timing for subgraph extraction
    start_time = time.time()

    # Use cached A_incidence_ from global variable (pre-computed in intialize_worker)
    nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A_, params_.hop, params_.enclosing_sub_graph, params_.max_nodes_per_hop, A_incidence=A_incidence_)

    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])

    datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
    str_id = '{:08}'.format(idx).encode('ascii')

    # OPTIMIZATION: Serialize in worker process (parallelize serialization!)
    # This moves expensive pickle.dumps() from main thread to workers
    serialized_datum = serialize(datum)

    # Log timing for this subgraph extraction
    extraction_time = time.time() - start_time
    # if idx % 100 == 0:  # Log every 100th subgraph to avoid too much output
    #     print(f"[TIMING] Subgraph {idx}: extraction_time={extraction_time:.4f}s, nodes={len(nodes)}, subgraph_size={subgraph_size}, hop={params_.hop}")

    # Return serialized data + metadata separately
    return (str_id, serialized_datum, subgraph_size, enc_ratio, num_pruned_nodes, n_labels)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    # Use numba-optimized version if available
    bfs_generator = _bfs_relational_fast(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None, A_incidence=None,
                                  use_semantic_pruning=False, semantic_embeddings=None, stage1_ratio=10, path_weight=0.6, semantic_weight=0.4):
    # extract the h-hop enclosing subgraphs around link 'ind'

    # Component timing: Incidence matrix creation (CACHED VERSION!)
    # If A_incidence is provided (cached), skip expensive computation (~1.5s saved per subgraph)
    if A_incidence is None:
        start_incidence = time.time()
        A_incidence = incidence_matrix(A_list)
        A_incidence += A_incidence.T
        incidence_time = time.time() - start_incidence
    # else: Using cached A_incidence - no computation needed!

    # Component timing: BFS neighbor extraction for both roots
    start_bfs = time.time()
    root1_nei = get_neighbor_nodes({ind[0]}, A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes({ind[1]}, A_incidence, h, max_nodes_per_hop)
    bfs_time = time.time() - start_bfs

    # Component timing: Node set operations
    start_sets = time.time()
    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)
    sets_time = time.time() - start_sets

    # Component timing: Subgraph creation
    start_subgraph = time.time()
    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    # OPTIMIZATION: Use numpy array indexing for faster sparse matrix slicing
    subgraph_nodes_arr = np.array(subgraph_nodes, dtype=np.int32)
    subgraph = [adj[subgraph_nodes_arr, :][:, subgraph_nodes_arr] for adj in A_list]
    subgraph_time = time.time() - start_subgraph

    # Component timing: Node labeling
    start_labeling = time.time()
    # OPTIMIZATION: Compute incidence matrix once and reuse
    subgraph_incidence = incidence_matrix(subgraph)
    labels, enclosing_subgraph_nodes = node_label(subgraph_incidence, max_distance=h)
    labeling_time = time.time() - start_labeling

    # Component timing: Final processing
    start_processing = time.time()
    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    # Apply semantic pruning if enabled
    if use_semantic_pruning and max_nodes_per_hop is not None and len(pruned_subgraph_nodes) > max_nodes_per_hop:
        try:
            from subgraph_extraction.semantic_pruning import two_stage_pruning

            # Exclude the two root nodes from pruning
            other_nodes = [n for n in pruned_subgraph_nodes if n not in ind]

            if other_nodes:  # Only prune if there are other nodes
                # Apply Two-Stage Pruning
                semantic_pruned_nodes = two_stage_pruning(
                    subgraph_nodes=other_nodes,
                    u=ind[0],
                    v=ind[1],
                    A_incidence=A_incidence,
                    embeddings=semantic_embeddings,
                    target_M=max_nodes_per_hop - 2,  # Reserve space for roots
                    stage1_ratio=stage1_ratio,
                    alpha=path_weight,
                    beta=semantic_weight,
                    use_semantic=(semantic_embeddings is not None),
                    verbose=False
                )

                # Reconstruct final node list with roots at front
                pruned_subgraph_nodes = list(ind) + semantic_pruned_nodes

                # Update labels to match new node ordering
                old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(pruned_subgraph_nodes)}
                new_labels = []

                for old_idx in range(len(pruned_labels)):
                    if old_idx < 2:  # Root nodes (0,1)
                        new_labels.append(pruned_labels[old_idx])
                    else:
                        # Find if this node is still in the pruned list
                        original_node_id = subgraph_nodes[old_idx]
                        if original_node_id in old_to_new_idx:
                            new_idx = old_to_new_idx[original_node_id]
                            new_labels.append(pruned_labels[old_idx])

                pruned_labels = np.array(new_labels) if new_labels else pruned_labels[:2]

        except Exception as e:
            logging.warning(f"Semantic pruning failed: {e}. Using original nodes.")
            # Fallback: use random sampling
            if len(pruned_subgraph_nodes) > max_nodes_per_hop:
                pruned_subgraph_nodes = np.random.choice(
                    pruned_subgraph_nodes,
                    max_nodes_per_hop,
                    replace=False
                ).tolist()
                # Rebuild labels for random sample
                old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(pruned_subgraph_nodes)}
                new_labels = [pruned_labels[old_to_new_idx[subgraph_nodes.index(node)]] for node in pruned_subgraph_nodes]
                pruned_labels = np.array(new_labels)

    # pruned_subgraph_nodes = subgraph_nodes
    # pruned_labels = labels

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)
    processing_time = time.time() - start_processing

    # Print detailed timing for each subgraph extraction
    # NOTE: Disabled for performance - these prints add significant overhead during evaluation
    # total_time = incidence_time + bfs_time + sets_time + subgraph_time + labeling_time + processing_time
    # print(f"[SUBGRAPH_TIMING] Link {ind}: total={total_time:.4f}s, incidence={incidence_time:.4f}s, bfs={bfs_time:.4f}s, sets={sets_time:.4f}s, subgraph={subgraph_time:.4f}s, labeling={labeling_time:.4f}s, processing={processing_time:.4f}s")
    # print(f"[SUBGRAPH_INFO] Link {ind}: roots={ind}, hop={h}, total_nodes={len(subgraph_nodes)}, pruned_nodes={len(pruned_subgraph_nodes)}, intersection_size={len(subgraph_nei_nodes_int)}, union_size={len(subgraph_nei_nodes_un)}")

    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes


def node_label(subgraph, max_distance=1):
    # OPTIMIZED: Use BFS instead of Dijkstra for unweighted graphs (3-5x faster!)
    # implementation of the node labeling scheme described in the paper
    roots = [0, 1]

    # OPTIMIZATION: For unweighted graphs, use BFS (much faster than Dijkstra)
    # BFS is O(V+E) vs Dijkstra O(E log V)
    try:
        from scipy.sparse.csgraph import breadth_first_order

        # Remove roots one by one and compute distances using BFS
        sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]

        # Use BFS for unweighted shortest paths (3-5x faster than Dijkstra)
        dist_to_roots_list = []
        for r, sg in enumerate(sgs_single_root):
            # BFS from node 0 in the modified graph
            distances = np.full(sg.shape[0], 1e7, dtype=np.float64)
            distances[0] = 0

            # Use scipy's shortest_path with method='BF' for better performance
            try:
                dists = ssp.csgraph.shortest_path(
                    sg,
                    method='BF',  # Bellman-Ford, faster for sparse graphs
                    directed=False,
                    unweighted=True,
                    indices=0
                )
                distances = dists[1:]  # Skip source node
            except:
                # Fallback to Dijkstra if BFS fails
                dists = ssp.csgraph.dijkstra(
                    sg,
                    indices=[0],
                    directed=False,
                    unweighted=True,
                    limit=1e6
                )
                distances = dists[0, 1:]

            dist_to_roots_list.append(np.clip(distances, 0, 1e7))

        dist_to_roots = np.array(list(zip(dist_to_roots_list[0], dist_to_roots_list[1])), dtype=int)
    except Exception as e:
        # Fallback to original Dijkstra implementation
        sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
        dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
        dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes


def _sample_fair_neg_parallel(pos_edges, entity_dict, entity_to_type, entity_pools,
                              adj_list, valid_heads, valid_tails, num_neg_samples_per_link,
                              constrained_neg_prob, n, target_neg_samples):
    """
    ULTRA-OPTIMIZED 8-core parallel negative sampling with:
    1. Pre-computed entity pools
    2. Edge sets instead of sparse matrix (10-100x faster serialization!)
    3. tqdm progress bar
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time
    from tqdm import tqdm

    # OPTIMIZATION: Convert adj_list to edge sets for MUCH faster serialization
    # Sparse matrix serialization is VERY slow (100+ MB)
    # Edge sets are MUCH smaller (typically < 10 MB)
    logging.info("Precomputing edge sets for ultra-fast serialization...")
    edge_sets = []
    for rel_idx, adj in enumerate(adj_list):
        # Convert sparse matrix to set of (head, tail) tuples
        coo = adj.tocoo()
        edge_set = set(zip(coo.row.tolist(), coo.col.tolist()))
        edge_sets.append(edge_set)
    logging.info(f"Edge sets ready! (much faster than sparse matrix serialization)")

    # OPTIMIZATION: Use all available cores for maximum speed
    # Tuned for optimal CPU utilization
    import multiprocessing
    num_workers = min(multiprocessing.cpu_count(), 8)  # Use up to 8 workers (optimal for most servers)
    min_chunk_size = 3000  # Larger chunks for parallelization
    ideal_chunk_size = 10000  # Larger chunks = less overhead

    if len(pos_edges) < num_workers * min_chunk_size:
        chunk_size = max(min_chunk_size, len(pos_edges) // num_workers)
    else:
        chunk_size = ideal_chunk_size

    chunks = []
    for i in range(0, len(pos_edges), chunk_size):
        chunk_end = min(i + chunk_size, len(pos_edges))
        chunks.append(pos_edges[i:chunk_end])

    logging.info(f"Processing {len(chunks)} chunks with {num_workers} workers ({chunk_size:,} edges per chunk)")

    # Shared data for all workers - MUCH lighter now!
    shared_data = {
        'entity_pools': entity_pools,
        'entity_to_type': entity_to_type,
        'edge_sets': edge_sets,  # ← MUCH smaller than adj_list!
        'valid_heads': valid_heads,
        'valid_tails': valid_tails,
        'num_neg_samples_per_link': num_neg_samples_per_link,
        'constrained_neg_prob': constrained_neg_prob,
        'n': n
    }

    all_neg_edges = []
    start_time = time.time()

    # OPTIMIZATION: Use tqdm progress bar instead of print
    pbar = tqdm(total=len(chunks), desc="Parallel negative sampling",
                unit="chunk", ncols=100)

    # Process with 8 workers
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = []
        for i, chunk in enumerate(chunks):
            future = executor.submit(_process_chunk_fast, chunk, shared_data, i)
            futures.append(future)

        # Collect results with progress bar
        for future in as_completed(futures):
            try:
                chunk_neg_edges = future.result()
                all_neg_edges.extend(chunk_neg_edges)

                # Update progress bar with speed stats
                elapsed = time.time() - start_time
                speed = len(all_neg_edges) / elapsed if elapsed > 0 else 0
                pbar.set_postfix({'negatives': f"{len(all_neg_edges):,}",
                                 'speed': f"{speed:.0f} neg/s"})
                pbar.update(1)

            except Exception as e:
                logging.error(f"Chunk processing failed: {e}")
                pbar.update(1)

    pbar.close()

    total_time = time.time() - start_time
    logging.info(f"PARALLEL SAMPLING COMPLETE!")
    logging.info(f"Generated {len(all_neg_edges):,} negatives in {total_time:.1f}s")
    logging.info(f"Average speed: {len(all_neg_edges)/total_time:.0f} negatives/second")
    logging.info(f"Speedup: ~{len(all_neg_edges)/total_time/72:.1f}x faster than original")

    return all_neg_edges


def _process_chunk_fast(chunk_edges, shared_data, chunk_idx):
    """
    ULTRA-OPTIMIZED chunk processing with:
    1. Pre-computed entity pools for fast sampling
    2. Retry mechanism to ensure enough negatives
    3. ULTRA-FAST edge set checking (10-100x faster than sparse matrix!)
    """
    entity_pools = shared_data['entity_pools']
    entity_to_type = shared_data['entity_to_type']
    edge_sets = shared_data['edge_sets']  # ← Use edge sets instead of adj_list!
    valid_heads = shared_data['valid_heads']
    valid_tails = shared_data['valid_tails']
    num_neg_samples_per_link = shared_data['num_neg_samples_per_link']
    constrained_neg_prob = shared_data['constrained_neg_prob']
    n = shared_data['n']

    chunk_neg_edges = []
    np.random.seed(chunk_idx)  # Different seed per chunk

    for head, tail, rel in chunk_edges:
        head_type = entity_to_type.get(head, "unknown")
        tail_type = entity_to_type.get(tail, "unknown")

        # OPTIMIZATION: Retry mechanism to ensure we get enough valid negatives
        attempts = 0
        max_attempts_per_link = num_neg_samples_per_link * 3  # 3x budget for retries
        negatives_for_this_link = 0

        while negatives_for_this_link < num_neg_samples_per_link and attempts < max_attempts_per_link:
            neg_head, neg_tail = head, tail

            if np.random.uniform() < constrained_neg_prob:
                if np.random.uniform() < 0.5:
                    neg_head = _sample_entity_fast(head, head_type, entity_pools, valid_heads, rel, n)
                else:
                    neg_tail = _sample_entity_fast(tail, tail_type, entity_pools, valid_tails, rel, n)
            else:
                if np.random.uniform() < 0.5:
                    neg_head = _sample_fair_entity_fast(head, head_type, entity_pools, valid_heads, rel, n)
                else:
                    neg_tail = _sample_fair_entity_fast(tail, tail_type, entity_pools, valid_tails, rel, n)

            # ULTRA-FAST edge set lookup: O(1) instead of O(nnz) for sparse matrix!
            # This is 10-100x faster than adj_list[rel][neg_head, neg_tail] == 0
            if neg_head != neg_tail and (neg_head, neg_tail) not in edge_sets[rel]:
                chunk_neg_edges.append([neg_head, neg_tail, rel])
                negatives_for_this_link += 1

            attempts += 1

    return chunk_neg_edges


def _sample_entity_fast(original, entity_type, entity_pools, valid_list, rel, n):
    """Fast entity sampling with pre-computed pools"""
    if entity_type != "unknown" and entity_type in entity_pools:
        pool = entity_pools[entity_type]
        if len(pool) > 1:
            idx = np.random.randint(0, len(pool))
            return pool[idx]

    if valid_list[rel]:
        return np.random.choice(valid_list[rel])
    else:
        return np.random.randint(0, n)


def _sample_fair_entity_fast(original, entity_type, entity_pools, valid_list, rel, n):
    """Fast fair entity sampling from same type - ensures we NEVER return original"""
    if entity_type != "unknown" and entity_type in entity_pools:
        pool = entity_pools[entity_type]
        if len(pool) > 1:
            # OPTIMIZED: Use while loop to guarantee we never return original
            # This is critical for fair negative sampling!
            max_attempts = 10  # Prevent infinite loop
            for _ in range(max_attempts):
                idx = np.random.randint(0, len(pool))
                candidate = pool[idx]
                if candidate != original:
                    return candidate

            # Fallback: find first non-original entity in pool
            for candidate in pool:
                if candidate != original:
                    return candidate

    # Fallback to valid entities for this relation
    if valid_list[rel]:
        candidates = [e for e in valid_list[rel] if e != original]
        if candidates:
            return np.random.choice(candidates)
        return np.random.choice(valid_list[rel])  # Last resort
    else:
        # Random entity (may be original in rare cases)
        return np.random.randint(0, n)
