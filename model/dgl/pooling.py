"""
Graph Pooling Strategies for GraIL

Implements various graph-level pooling methods:
- Baseline: mean, sum, max pooling
- Attention-based: global attention, multi-head attention
- Query-based: attention conditioned on head/tail entities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class GraphPooling(nn.Module):
    """
    Unified graph pooling module with multiple strategies.

    Args:
        emb_dim: Node embedding dimension
        pool_type: Pooling strategy ('mean', 'sum', 'max', 'attention', 'query_attention')
        num_heads: Number of attention heads (for multi-head attention)
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
        if pool_type == 'attention':
            if num_heads == 1:
                self.pooling = GlobalAttentionPooling(emb_dim, hidden_dim, dropout)
            else:
                self.pooling = MultiHeadAttentionPooling(emb_dim, num_heads, dropout, hidden_dim)
        elif pool_type == 'query_attention':
            if num_heads == 1:
                self.pooling = QueryAttentionPooling(emb_dim, hidden_dim, dropout)
            else:
                # Multi-head query attention
                self.pooling = MultiHeadQueryAttentionPooling(emb_dim, num_heads, dropout, hidden_dim)
        elif pool_type in ['mean', 'sum', 'max']:
            self.pooling = None  # Use DGL built-in
        else:
            raise ValueError(f"Unknown pooling type: {pool_type}")

    def forward(self, g, node_features, head_ids=None, tail_ids=None):
        """
        Pool node features to graph-level representation.

        Args:
            g: DGL graph (batched)
            node_features: Node features (N x emb_dim)
            head_ids: Head entity IDs for query attention (optional)
            tail_ids: Tail entity IDs for query attention (optional)

        Returns:
            Graph-level representation (batch_size x emb_dim)
        """
        if self.pool_type == 'mean':
            return dgl.mean_nodes(g, 'repr')
        elif self.pool_type == 'sum':
            return dgl.sum_nodes(g, 'repr')
        elif self.pool_type == 'max':
            return dgl.max_nodes(g, 'repr')
        elif self.pool_type == 'attention':
            return self.pooling(g, node_features)
        elif self.pool_type == 'query_attention':
            if head_ids is None or tail_ids is None:
                raise ValueError("query_attention requires head_ids and tail_ids")
            return self.pooling(g, node_features, head_ids, tail_ids)
        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type}")


class GlobalAttentionPooling(nn.Module):
    """
    Global attention pooling: learns importance weights for each node.

    g = Σ α_i * h_i, where α_i = softmax(MLP(h_i))
    """
    def __init__(self, emb_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim

        # Attention scoring MLP
        self.attention_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, g, node_features):
        """
        Args:
            g: Batched DGL graph
            node_features: Node features (N x emb_dim)

        Returns:
            Graph representations (batch_size x emb_dim)
        """
        # Compute attention scores for each node
        # scores: (N x 1)
        scores = self.attention_mlp(node_features)

        # Apply softmax per graph in batch
        # attention_weights: (N x 1)
        g.ndata['score'] = scores
        g.ndata['h'] = node_features

        # Softmax normalization within each graph
        attention_weights = dgl.softmax_nodes(g, 'score')
        g.ndata['alpha'] = attention_weights

        # Weighted sum: g = Σ α_i * h_i
        g.ndata['weighted_h'] = g.ndata['alpha'] * g.ndata['h']
        graph_repr = dgl.sum_nodes(g, 'weighted_h')

        return graph_repr


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling: multiple attention mechanisms in parallel.

    Each head learns different aspects of node importance.
    Final representation: concat(head_1, ..., head_K) or mean(head_1, ..., head_K)
    """
    def __init__(self, emb_dim, num_heads=4, dropout=0.0, hidden_dim=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        if hidden_dim is None:
            hidden_dim = emb_dim // 2

        # Multiple attention heads
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_heads)
        ])

        # Projection to combine heads (if using concat)
        # We'll use mean instead to keep output dimension = emb_dim
        self.combine_type = 'mean'  # or 'concat'

    def forward(self, g, node_features):
        """
        Args:
            g: Batched DGL graph
            node_features: Node features (N x emb_dim)

        Returns:
            Graph representations (batch_size x emb_dim)
        """
        head_outputs = []

        for head_idx, attention_head in enumerate(self.attention_heads):
            # Compute attention scores for this head
            scores = attention_head(node_features)

            # Store in unique key for this head
            g.ndata[f'score_{head_idx}'] = scores

            # Softmax normalization
            attention_weights = dgl.softmax_nodes(g, f'score_{head_idx}')

            # Weighted sum
            g.ndata[f'weighted_h_{head_idx}'] = attention_weights * node_features
            head_repr = dgl.sum_nodes(g, f'weighted_h_{head_idx}')
            head_outputs.append(head_repr)

        # Combine heads
        if self.combine_type == 'mean':
            # Average across heads: (batch_size x emb_dim)
            graph_repr = torch.mean(torch.stack(head_outputs, dim=0), dim=0)
        elif self.combine_type == 'concat':
            # Concatenate heads: (batch_size x (num_heads * emb_dim))
            graph_repr = torch.cat(head_outputs, dim=1)

        return graph_repr


class QueryAttentionPooling(nn.Module):
    """
    Query-based attention pooling: attention conditioned on head/tail entities.

    Uses head and tail embeddings as queries to determine node importance.
    α_i = softmax(score(h_i, [h_head; h_tail]))
    """
    def __init__(self, emb_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim

        # Attention scoring MLP conditioned on query
        # Input: [node_emb; query_emb] = [emb_dim; 2*emb_dim] = 3*emb_dim
        self.attention_mlp = nn.Sequential(
            nn.Linear(3 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, g, node_features, head_ids, tail_ids):
        """
        Args:
            g: Batched DGL graph
            node_features: Node features (N x emb_dim)
            head_ids: Head entity indices in each graph (batch_size,)
            tail_ids: Tail entity indices in each graph (batch_size,)

        Returns:
            Graph representations (batch_size x emb_dim)
        """
        batch_num_nodes = g.batch_num_nodes()
        batch_size = len(batch_num_nodes)

        # Extract head and tail embeddings
        head_embeds = node_features[head_ids]  # (batch_size x emb_dim)
        tail_embeds = node_features[tail_ids]  # (batch_size x emb_dim)
        query_embeds = torch.cat([head_embeds, tail_embeds], dim=1)  # (batch_size x 2*emb_dim)

        # Expand query to all nodes in each graph
        # Create mapping from node to its graph
        graph_ids = torch.cat([
            torch.full((n,), i, dtype=torch.long, device=node_features.device)
            for i, n in enumerate(batch_num_nodes)
        ])

        # Broadcast query to all nodes
        node_queries = query_embeds[graph_ids]  # (N x 2*emb_dim)

        # Concatenate node features with query
        node_query_concat = torch.cat([node_features, node_queries], dim=1)  # (N x 3*emb_dim)

        # Compute attention scores
        scores = self.attention_mlp(node_query_concat)  # (N x 1)

        # Softmax and weighted sum (same as GlobalAttentionPooling)
        g.ndata['score'] = scores
        g.ndata['h'] = node_features
        attention_weights = dgl.softmax_nodes(g, 'score')
        g.ndata['alpha'] = attention_weights
        g.ndata['weighted_h'] = g.ndata['alpha'] * g.ndata['h']
        graph_repr = dgl.sum_nodes(g, 'weighted_h')

        return graph_repr


class MultiHeadQueryAttentionPooling(nn.Module):
    """
    Multi-head version of QueryAttentionPooling.
    """
    def __init__(self, emb_dim, num_heads=4, dropout=0.0, hidden_dim=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        if hidden_dim is None:
            hidden_dim = emb_dim // 2

        # Multiple query-conditioned attention heads
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_heads)
        ])

        self.combine_type = 'mean'

    def forward(self, g, node_features, head_ids, tail_ids):
        """
        Args:
            g: Batched DGL graph
            node_features: Node features (N x emb_dim)
            head_ids: Head entity indices
            tail_ids: Tail entity indices

        Returns:
            Graph representations (batch_size x emb_dim)
        """
        batch_num_nodes = g.batch_num_nodes()
        batch_size = len(batch_num_nodes)

        # Extract query embeddings
        head_embeds = node_features[head_ids]
        tail_embeds = node_features[tail_ids]
        query_embeds = torch.cat([head_embeds, tail_embeds], dim=1)

        # Map nodes to graphs
        graph_ids = torch.cat([
            torch.full((n,), i, dtype=torch.long, device=node_features.device)
            for i, n in enumerate(batch_num_nodes)
        ])
        node_queries = query_embeds[graph_ids]
        node_query_concat = torch.cat([node_features, node_queries], dim=1)

        # Multi-head attention
        head_outputs = []
        for head_idx, attention_head in enumerate(self.attention_heads):
            scores = attention_head(node_query_concat)
            g.ndata[f'score_{head_idx}'] = scores
            attention_weights = dgl.softmax_nodes(g, f'score_{head_idx}')
            g.ndata[f'weighted_h_{head_idx}'] = attention_weights * node_features
            head_repr = dgl.sum_nodes(g, f'weighted_h_{head_idx}')
            head_outputs.append(head_repr)

        # Combine heads
        graph_repr = torch.mean(torch.stack(head_outputs, dim=0), dim=0)
        return graph_repr