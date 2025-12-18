"""
Graph Classifier with Ablation Support

Allows toggling:
1. Relation embedding (r) in decoder
2. Different pooling strategies
3. Head/tail embedding concatenation
"""

from .rgcn_model import RGCN
from .compgcn_model import CompGCN
from .pooling import GraphPooling
import dgl
import torch.nn as nn
import torch
import logging


class GraphClassifierAblation(nn.Module):
    """
    Graph classifier with ablation options for studying component effectiveness.

    Args:
        params: Configuration parameters
            - use_rel_emb: Whether to use relation embedding in decoder (default: True)
            - pool_type: Pooling strategy ('mean', 'sum', 'max', 'query_attention')
            - pool_heads: Number of attention heads for query_attention
            - add_ht_emb: Whether to concatenate head/tail embeddings
    """

    def __init__(self, params, relation2id):
        super().__init__()

        self.params = params
        self.relation2id = relation2id

        # Ablation flags
        self.use_rel_emb = getattr(params, 'use_rel_emb', True)
        self.add_ht_emb = getattr(params, 'add_ht_emb', True)

        # GNN backbone
        gnn_type = getattr(params, 'gnn_type', 'rgcn').lower()
        if gnn_type == 'compgcn':
            self.gnn = CompGCN(params)
            logging.info("Using CompGCN backbone")
        else:
            self.gnn = RGCN(params)
            logging.info("Using R-GCN backbone")

        # Relation embedding (optional based on ablation)
        if self.use_rel_emb:
            self.rel_emb = nn.Embedding(params.num_rels, params.rel_emb_dim, sparse=False)
            logging.info(f"Using relation embedding: dim={params.rel_emb_dim}")
        else:
            self.rel_emb = None
            logging.info("NOT using relation embedding (ablation)")

        # Graph pooling
        pool_type = getattr(params, 'pool_type', 'mean')
        pool_heads = getattr(params, 'pool_heads', 1)
        pool_dropout = getattr(params, 'pool_dropout', 0.0)

        self.pool_type = pool_type
        self.pooling = GraphPooling(
            emb_dim=params.num_gcn_layers * params.emb_dim,
            pool_type=pool_type,
            num_heads=pool_heads,
            dropout=pool_dropout
        )
        logging.info(f"Pooling: type={pool_type}, heads={pool_heads}")

        # Calculate FC layer input dimension
        gnn_out_dim = params.num_gcn_layers * params.emb_dim
        fc_input_dim = gnn_out_dim  # graph representation

        if self.add_ht_emb:
            fc_input_dim += 2 * gnn_out_dim  # head + tail embeddings

        if self.use_rel_emb:
            fc_input_dim += params.rel_emb_dim  # relation embedding

        self.fc_layer = nn.Linear(fc_input_dim, 1)
        logging.info(f"FC layer input dim: {fc_input_dim}")

    def forward(self, data, return_attention=False):
        """
        Forward pass.

        Args:
            data: Tuple of (batched_graph, relation_labels)
            return_attention: If True, also return attention weights (only for query_attention)

        Returns:
            output: Link prediction scores (batch_size x 1)
            attention_weights: (optional) Attention weights from pooling
        """
        g, rel_labels = data

        # GNN forward
        g.ndata['h'] = self.gnn(g)

        # Extract head and tail node indices
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)

        # Flatten node features
        gnn_out_dim = self.params.num_gcn_layers * self.params.emb_dim
        node_features = g.ndata['repr'].view(-1, gnn_out_dim)

        # Graph pooling
        attention_weights = None
        if self.pool_type == 'query_attention':
            if return_attention:
                g_out, attention_weights = self.pooling(g, node_features, head_ids, tail_ids, return_attention=True)
            else:
                g_out = self.pooling(g, node_features, head_ids, tail_ids)
        else:
            g_out = self.pooling(g, node_features)

        # Build final representation
        components = [g_out.view(-1, gnn_out_dim)]

        if self.add_ht_emb:
            head_embs = g.ndata['repr'][head_ids].view(-1, gnn_out_dim)
            tail_embs = g.ndata['repr'][tail_ids].view(-1, gnn_out_dim)
            components.extend([head_embs, tail_embs])

        if self.use_rel_emb:
            components.append(self.rel_emb(rel_labels))

        g_rep = torch.cat(components, dim=1)

        # Final prediction
        output = self.fc_layer(g_rep)

        if return_attention:
            return output, attention_weights
        return output

    def get_attention_analysis(self, data):
        """
        Get detailed attention analysis for a batch.

        Returns:
            attention_weights: Attention weights for each node
            node_info: Dict with node information for analysis
        """
        g, rel_labels = data

        # GNN forward
        g.ndata['h'] = self.gnn(g)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)

        gnn_out_dim = self.params.num_gcn_layers * self.params.emb_dim
        node_features = g.ndata['repr'].view(-1, gnn_out_dim)

        if self.pool_type != 'query_attention':
            raise ValueError("Attention analysis only available for query_attention pooling")

        # Get attention weights
        _, attention_weights = self.pooling(g, node_features, head_ids, tail_ids, return_attention=True)

        # Build node info
        batch_num_nodes = g.batch_num_nodes()
        node_to_graph = torch.cat([
            torch.full((n,), i, dtype=torch.long, device=node_features.device)
            for i, n in enumerate(batch_num_nodes)
        ])

        node_info = {
            'node_ids': g.ndata.get('id', None),
            'node_to_graph': node_to_graph,
            'batch_num_nodes': batch_num_nodes,
            'head_ids': head_ids,
            'tail_ids': tail_ids,
            'rel_labels': rel_labels,
        }

        return attention_weights, node_info
