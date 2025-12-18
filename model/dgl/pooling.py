"""
Graph Pooling Strategies for GraIL

Implements various graph-level pooling methods:
- Baseline: mean, sum, max pooling
- Query-based attention: conditioned on head/tail entities (RECOMMENDED for link prediction)

For link prediction tasks, QueryAttentionPooling is recommended because:
1. It conditions attention on the specific (head, tail) pair being predicted
2. Nodes relevant to the link get higher attention weights
3. Multi-head version learns diverse importance perspectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class GraphPooling(nn.Module):
    """
    Unified graph pooling module for link prediction.

    Args:
        emb_dim: Node embedding dimension
        pool_type: Pooling strategy
            - 'mean', 'sum', 'max': Simple aggregation (fast, no learnable params)
            - 'query_attention': Query-conditioned attention (RECOMMENDED)
        num_heads: Number of attention heads (only for query_attention)
            - 1: Single-head (faster, fewer params)
            - 2-8: Multi-head (learns diverse perspectives)
        dropout: Dropout rate for attention weights
        hidden_dim: Hidden dimension for attention MLP
    """
    def __init__(self, emb_dim, pool_type='mean', num_heads=1, dropout=0.0, hidden_dim=None):
        super().__init__()
        self.pool_type = pool_type
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        if hidden_dim is None:
            hidden_dim = emb_dim // 2

        # Initialize pooling module based on type
        if pool_type == 'query_attention':
            self.pooling = QueryAttentionPooling(emb_dim, num_heads, dropout, hidden_dim)
        elif pool_type in ['mean', 'sum', 'max']:
            self.pooling = None  # Use DGL built-in
        else:
            raise ValueError(f"Unknown pooling type: {pool_type}. Use 'mean', 'sum', 'max', or 'query_attention'")

    def forward(self, g, node_features, head_ids=None, tail_ids=None, return_attention=False):
        """
        Pool node features to graph-level representation.

        Args:
            g: DGL graph (batched)
            node_features: Node features (N x emb_dim)
            head_ids: Head entity IDs for query attention
            tail_ids: Tail entity IDs for query attention
            return_attention: If True, return attention weights (only for query_attention)

        Returns:
            graph_repr: Graph-level representation (batch_size x emb_dim)
            attention_weights: (optional) Attention weights for analysis
        """
        if self.pool_type == 'mean':
            return dgl.mean_nodes(g, 'repr')
        elif self.pool_type == 'sum':
            return dgl.sum_nodes(g, 'repr')
        elif self.pool_type == 'max':
            return dgl.max_nodes(g, 'repr')
        elif self.pool_type == 'query_attention':
            if head_ids is None or tail_ids is None:
                raise ValueError("query_attention requires head_ids and tail_ids")
            return self.pooling(g, node_features, head_ids, tail_ids, return_attention)
        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type}")


class QueryAttentionPooling(nn.Module):
    """
    Query-based attention pooling for link prediction.

    Attention is conditioned on head/tail entities:
        α_i = softmax(score(h_i, [h_head; h_tail]))

    This is the RECOMMENDED pooling for link prediction because:
    1. It knows which link (head, tail) is being predicted
    2. Nodes relevant to the specific link get higher weights
    3. Multi-head learns diverse importance aspects

    Args:
        emb_dim: Node embedding dimension
        num_heads: Number of attention heads
            - 1: Single-head attention (simple, fast)
            - 2-8: Multi-head attention (learns diverse perspectives)
        dropout: Dropout rate for attention
        hidden_dim: Hidden dimension for attention MLP

    Example:
        # Single-head (simple)
        pooling = QueryAttentionPooling(emb_dim=64, num_heads=1)

        # Multi-head (recommended for complex graphs)
        pooling = QueryAttentionPooling(emb_dim=64, num_heads=4)

        # Forward pass
        graph_repr = pooling(g, node_features, head_ids, tail_ids)

        # Get attention weights for analysis
        graph_repr, attn_weights = pooling(g, node_features, head_ids, tail_ids, return_attention=True)
    """
    def __init__(self, emb_dim, num_heads=1, dropout=0.0, hidden_dim=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        if hidden_dim is None:
            hidden_dim = emb_dim // 2

        if num_heads == 1:
            # Single-head: simpler architecture
            self.head_dim = emb_dim
            self.value_projection = nn.Linear(emb_dim, emb_dim)

            # Attention scoring: [node; head; tail] -> score
            self.attention_mlp = nn.Sequential(
                nn.Linear(3 * emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

            self.output_projection = None  # No need for single head
            self.layer_norm = nn.LayerNorm(emb_dim)
        else:
            # Multi-head: each head learns different aspects
            assert emb_dim % num_heads == 0, f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})"
            self.head_dim = emb_dim // num_heads

            # Value projection for each head
            self.value_projections = nn.ModuleList([
                nn.Linear(emb_dim, self.head_dim)
                for _ in range(num_heads)
            ])

            # Attention scoring for each head
            self.attention_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(3 * emb_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                )
                for _ in range(num_heads)
            ])

            # Output projection after concatenation
            self.output_projection = nn.Linear(emb_dim, emb_dim)
            self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, g, node_features, head_ids, tail_ids, return_attention=False):
        """
        Args:
            g: Batched DGL graph
            node_features: Node features (N x emb_dim)
            head_ids: Head entity indices (batch_size,)
            tail_ids: Tail entity indices (batch_size,)
            return_attention: If True, return attention weights for analysis

        Returns:
            graph_repr: Graph representations (batch_size x emb_dim)
            attention_weights: (optional)
                - Single-head: (N x 1) tensor
                - Multi-head: list of (N x 1) tensors, one per head
        """
        batch_num_nodes = g.batch_num_nodes()

        # Build query from head/tail embeddings
        head_embeds = node_features[head_ids]  # (batch_size x emb_dim)
        tail_embeds = node_features[tail_ids]  # (batch_size x emb_dim)
        query_embeds = torch.cat([head_embeds, tail_embeds], dim=1)  # (batch_size x 2*emb_dim)

        # Map each node to its graph for broadcasting
        graph_ids = torch.cat([
            torch.full((n,), i, dtype=torch.long, device=node_features.device)
            for i, n in enumerate(batch_num_nodes)
        ])

        # Broadcast query to all nodes
        node_queries = query_embeds[graph_ids]  # (N x 2*emb_dim)
        node_query_concat = torch.cat([node_features, node_queries], dim=1)  # (N x 3*emb_dim)

        if self.num_heads == 1:
            return self._single_head_forward(g, node_features, node_query_concat, return_attention)
        else:
            return self._multi_head_forward(g, node_features, node_query_concat, return_attention)

    def _single_head_forward(self, g, node_features, node_query_concat, return_attention):
        """Single-head attention forward pass."""
        # Project values
        projected_values = self.value_projection(node_features)  # (N x emb_dim)

        # Compute attention scores
        scores = self.attention_mlp(node_query_concat)  # (N x 1)
        g.ndata['score'] = scores

        # Softmax per graph
        attention_weights = dgl.softmax_nodes(g, 'score')

        # Weighted sum
        g.ndata['weighted_h'] = attention_weights * projected_values
        graph_repr = dgl.sum_nodes(g, 'weighted_h')

        # Layer norm
        graph_repr = self.layer_norm(graph_repr)

        if return_attention:
            return graph_repr, attention_weights.detach()
        return graph_repr

    def _multi_head_forward(self, g, node_features, node_query_concat, return_attention):
        """Multi-head attention forward pass."""
        head_outputs = []
        attention_weights_list = []

        for head_idx, (attention_head, value_proj) in enumerate(
            zip(self.attention_heads, self.value_projections)
        ):
            # Project to head-specific value space
            projected_values = value_proj(node_features)  # (N x head_dim)

            # Compute attention scores for this head
            scores = attention_head(node_query_concat)  # (N x 1)
            g.ndata[f'score_{head_idx}'] = scores

            # Softmax per graph
            attention_weights = dgl.softmax_nodes(g, f'score_{head_idx}')

            if return_attention:
                attention_weights_list.append(attention_weights.detach())

            # Weighted sum with projected values
            g.ndata[f'weighted_h_{head_idx}'] = attention_weights * projected_values
            head_repr = dgl.sum_nodes(g, f'weighted_h_{head_idx}')  # (batch x head_dim)
            head_outputs.append(head_repr)

        # Concatenate all heads
        concat_repr = torch.cat(head_outputs, dim=-1)  # (batch x emb_dim)

        # Output projection + layer norm
        graph_repr = self.output_projection(concat_repr)
        graph_repr = self.layer_norm(graph_repr)

        if return_attention:
            return graph_repr, attention_weights_list
        return graph_repr

    def get_attention_weights(self, g, node_features, head_ids, tail_ids):
        """
        Convenience method to get attention weights for visualization/analysis.

        Returns:
            attention_weights: Attention weights for each node
            node_to_graph: Mapping from node index to graph index in batch
        """
        _, attention_weights = self.forward(g, node_features, head_ids, tail_ids, return_attention=True)

        # Create node-to-graph mapping for easier analysis
        batch_num_nodes = g.batch_num_nodes()
        node_to_graph = torch.cat([
            torch.full((n,), i, dtype=torch.long, device=node_features.device)
            for i, n in enumerate(batch_num_nodes)
        ])

        return attention_weights, node_to_graph
