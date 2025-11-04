"""Integration tests for GRAIL models with updated dependencies."""

import pytest


class TestModelComponents:
    """Test model components work with updated dependencies."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample DGL graph for testing."""
        pytest.importorskip("dgl")
        pytest.importorskip("torch")

        import dgl
        import torch

        # Create a simple heterogeneous graph for link prediction
        src = torch.tensor([0, 1, 2, 3, 4])
        dst = torch.tensor([1, 2, 3, 4, 0])
        g = dgl.graph((src, dst))
        g.ndata["feat"] = torch.randn(5, 10)
        g.ndata["id"] = torch.tensor([0, 1, 2, 0, 0])
        g.edata["type"] = torch.randint(0, 3, (5,))
        return g

    @pytest.mark.requires_dgl
    def test_graph_creation(self, sample_graph):
        """Test that sample graph is created correctly."""
        assert sample_graph.num_nodes() == 5
        assert sample_graph.num_edges() == 5
        assert "feat" in sample_graph.ndata
        assert "type" in sample_graph.edata

    @pytest.mark.requires_dgl
    def test_rgcn_layer_imports(self):
        """Test RGCN layer can be imported."""
        try:
            from model.dgl.layers import RGCNLayer, RGCNBasisLayer
        except ImportError as e:
            pytest.fail(f"Failed to import RGCN layers: {e}")

    @pytest.mark.requires_dgl
    def test_identity_not_imported(self):
        """Verify custom Identity class is removed."""
        import model.dgl.layers as layers

        # The old custom Identity class should not exist
        assert not hasattr(layers, "Identity"), (
            "Custom Identity class should be removed in favor of torch.nn.Identity"
        )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_dgl
class TestTrainingPipeline:
    """Integration tests for training pipeline."""

    def test_pytorch_basic_training(self):
        """Test basic PyTorch training loop works."""
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # Simple model
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

        # Simple data
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)

        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Single training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0, "Loss should be positive"

    def test_dgl_basic_operations(self):
        """Test basic DGL operations work."""
        pytest.importorskip("dgl")
        pytest.importorskip("torch")

        import dgl
        import torch

        # Create graph
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        g = dgl.graph((src, dst))

        # Add node features
        g.ndata["h"] = torch.randn(3, 10)

        # Test message passing
        import dgl.function as fn

        g.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_new"))

        assert "h_new" in g.ndata
        assert g.ndata["h_new"].shape == (3, 10)
