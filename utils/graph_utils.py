import statistics
import numpy as np
import scipy.sparse as ssp
import torch
import networkx as nx
import dgl
import pickle


def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def deserialize(data):
    data_tuple = pickle.loads(data)
    keys = ('nodes', 'r_label', 'g_label', 'n_label')
    return dict(zip(keys, data_tuple))


def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def ssp_to_torch(A, device, dense=False):
    '''
    A : Sparse adjacency matrix
    '''
    idx = torch.LongTensor([A.tocoo().row, A.tocoo().col])
    dat = torch.FloatTensor(A.tocoo().data)
    A = torch.sparse.FloatTensor(idx, dat, torch.Size([A.shape[0], A.shape[1]])).to(device=device)
    return A


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    FAST VERSION: Converting ssp multigraph to DGL multigraph directly without NetworkX
    """
    import time
    print("[GRAPH] FAST: Converting sparse matrices to DGL graph...")
    start_time = time.time()

    num_nodes = graph[0].shape[0]
    print(f"[GRAPH] FAST: Graph has {num_nodes} nodes")

    # Collect all edges and relations
    all_src = []
    all_dst = []
    all_types = []

    total_edges = 0
    for rel, adj in enumerate(graph):
        coo = adj.tocoo()
        if len(coo.row) > 0:
            all_src.extend(coo.row.tolist())
            all_dst.extend(coo.col.tolist())
            all_types.extend([rel] * len(coo.row))
            total_edges += len(coo.row)
            print(f"[GRAPH] FAST: Relation {rel}: {len(coo.row)} edges")

    print(f"[GRAPH] FAST: Total edges: {total_edges}")

    # Create DGL heterograph directly - MUCH FASTER
    import torch
    src_tensor = torch.tensor(all_src, dtype=torch.int64)
    dst_tensor = torch.tensor(all_dst, dtype=torch.int64)
    type_tensor = torch.tensor(all_types, dtype=torch.int64)

    print("[GRAPH] FAST: Creating DGL graph...")
    g_dgl = dgl.graph((src_tensor, dst_tensor), num_nodes=num_nodes)
    g_dgl.edata['type'] = type_tensor

    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    conversion_time = time.time() - start_time
    print(f"[GRAPH] FAST: Graph conversion completed in {conversion_time:.2f} seconds")

    return g_dgl


def collate_dgl(samples):
    # The input `samples` is a list of pairs
    graphs_pos, g_labels_pos, r_labels_pos, graphs_negs, g_labels_negs, r_labels_negs = map(list, zip(*samples))
    batched_graph_pos = dgl.batch(graphs_pos)

    graphs_neg = [item for sublist in graphs_negs for item in sublist]
    g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
    r_labels_neg = [item for sublist in r_labels_negs for item in sublist]

    batched_graph_neg = dgl.batch(graphs_neg)
    return (batched_graph_pos, r_labels_pos), g_labels_pos, (batched_graph_neg, r_labels_neg), g_labels_neg


def move_batch_to_device_dgl(batch, device):
    ((g_dgl_pos, r_labels_pos), targets_pos, (g_dgl_neg, r_labels_neg), targets_neg) = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    r_labels_pos = torch.LongTensor(r_labels_pos).to(device=device)

    targets_neg = torch.LongTensor(targets_neg).to(device=device)
    r_labels_neg = torch.LongTensor(r_labels_neg).to(device=device)

    # Move entire graphs to device first, then send individual features
    g_dgl_pos = g_dgl_pos.to(device)
    g_dgl_neg = g_dgl_neg.to(device)

    return ((g_dgl_pos, r_labels_pos), targets_pos, (g_dgl_neg, r_labels_neg), targets_neg)


def send_graph_to_device(g, device):
    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device)

    # edges
    labels = g.edge_attr_schemes()
    for l in labels.keys():
        g.edata[l] = g.edata.pop(l).to(device)
    return g

#  The following three functions are modified from networks source codes to
#  accomodate diameter and radius for dirercted graphs


def eccentricity(G):
    e = {}
    for n in G.nbunch_iter():
        length = nx.single_source_shortest_path_length(G, n)
        e[n] = max(length.values())
    return e


def radius(G):
    e = eccentricity(G)
    e = np.where(np.array(list(e.values())) > 0, list(e.values()), np.inf)
    return min(e)


def diameter(G):
    e = eccentricity(G)
    return max(e.values())
