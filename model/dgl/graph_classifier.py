from .rgcn_model import RGCN
from .pooling import GraphPooling
import dgl
import torch.nn as nn
import torch
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)

        # Graph pooling module
        pool_type = getattr(params, 'pool_type', 'mean')
        pool_heads = getattr(params, 'pool_heads', 1)
        pool_dropout = getattr(params, 'pool_dropout', 0.0)
        self.pooling = GraphPooling(
            emb_dim=params.num_gcn_layers * params.emb_dim,
            pool_type=pool_type,
            num_heads=pool_heads,
            dropout=pool_dropout
        )

        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)

    def forward(self, data):
        g, rel_labels = data
        g.ndata['h'] = self.gnn(g)

        # Extract head and tail node indices
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)

        # Graph-level pooling
        # Flatten node features from (N, num_gcn_layers, emb_dim) to (N, num_gcn_layers * emb_dim)
        node_features = g.ndata['repr'].view(-1, self.params.num_gcn_layers * self.params.emb_dim)
        if self.params.pool_type == 'query_attention':
            g_out = self.pooling(g, node_features, head_ids, tail_ids)
        else:
            g_out = self.pooling(g, node_features)

        # Extract head and tail embeddings for final representation
        head_embs = g.ndata['repr'][head_ids]
        tail_embs = g.ndata['repr'][tail_ids]

        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               self.rel_emb(rel_labels)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim), self.rel_emb(rel_labels)], dim=1)

        output = self.fc_layer(g_rep)
        return output
