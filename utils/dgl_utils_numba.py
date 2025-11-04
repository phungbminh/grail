"""
Numba-optimized version of BFS for knowledge graph extraction.
Significantly faster than scipy sparse matrix operations.
"""
import numpy as np
import scipy.sparse as ssp
from numba import jit, prange
import random


@jit(nopython=True, cache=True)
def _bfs_csr_numba(indptr, indices, roots, max_hops, max_nodes_per_hop=-1):
    """
    Numba-optimized BFS for CSR matrix.

    Args:
        indptr: CSR format indptr array
        indices: CSR format indices array
        roots: Starting nodes (numpy array)
        max_hops: Maximum number of hops
        max_nodes_per_hop: Maximum nodes to sample per hop (-1 for no limit)

    Returns:
        List of arrays, each containing nodes at that hop level
    """
    n_nodes = len(indptr) - 1
    visited = np.zeros(n_nodes, dtype=np.bool_)

    # Mark roots as visited
    for root in roots:
        visited[root] = True

    current_lvl = roots.copy()
    levels = []

    for hop in range(max_hops):
        if len(current_lvl) == 0:
            break

        # Find all neighbors of current level
        next_lvl_set = set()
        for node in current_lvl:
            # Get neighbors from CSR structure
            start = indptr[node]
            end = indptr[node + 1]
            for i in range(start, end):
                neighbor = indices[i]
                if not visited[neighbor]:
                    next_lvl_set.add(neighbor)

        if len(next_lvl_set) == 0:
            break

        # Convert set to array
        next_lvl = np.array(list(next_lvl_set), dtype=np.int64)

        # Sample if too many nodes
        if max_nodes_per_hop > 0 and len(next_lvl) > max_nodes_per_hop:
            indices_sample = np.random.choice(len(next_lvl), max_nodes_per_hop, replace=False)
            next_lvl = next_lvl[indices_sample]

        # Mark as visited
        for node in next_lvl:
            visited[node] = True

        levels.append(next_lvl)
        current_lvl = next_lvl

    return levels


def _bfs_relational_numba(adj, roots, max_nodes_per_hop=None):
    """
    Numba-optimized BFS generator for graphs.
    Drop-in replacement for _bfs_relational with 10-100x speedup.

    Args:
        adj: Scipy sparse matrix (CSR format)
        roots: Set or list of root nodes
        max_nodes_per_hop: Maximum nodes to sample per hop

    Yields:
        Sets of nodes at each BFS level
    """
    # Convert to CSR if not already
    if not isinstance(adj, ssp.csr_matrix):
        adj = adj.tocsr()

    # Convert roots to numpy array
    roots_array = np.array(list(roots), dtype=np.int64)

    # Run numba BFS (get all levels at once)
    max_hops = adj.shape[0]  # Upper bound
    max_per_hop = max_nodes_per_hop if max_nodes_per_hop else -1

    levels = _bfs_csr_numba(adj.indptr, adj.indices, roots_array, max_hops, max_per_hop)

    # Yield each level as a set (to maintain API compatibility)
    for level in levels:
        yield set(level.tolist())


@jit(nopython=True, parallel=True, cache=True)
def _get_neighbors_numba(indptr, indices, nodes):
    """
    Fast neighbor lookup using numba.

    Args:
        indptr: CSR indptr
        indices: CSR indices
        nodes: Numpy array of node IDs

    Returns:
        Set of neighbor node IDs
    """
    neighbors = set()
    for node in nodes:
        start = indptr[node]
        end = indptr[node + 1]
        for i in range(start, end):
            neighbors.add(indices[i])
    return neighbors


def _get_neighbors_fast(adj, nodes):
    """
    Fast neighbor lookup (drop-in replacement).

    Args:
        adj: Scipy sparse CSR matrix
        nodes: Set or list of nodes

    Returns:
        Set of neighbor nodes
    """
    if not isinstance(adj, ssp.csr_matrix):
        adj = adj.tocsr()

    nodes_array = np.array(list(nodes), dtype=np.int64)
    neighbors = _get_neighbors_numba(adj.indptr, adj.indices, nodes_array)
    return neighbors
