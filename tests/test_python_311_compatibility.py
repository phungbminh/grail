"""Test suite to verify Python 3.11 compatibility and dependency versions."""

import sys

import pytest


class TestPythonVersion:
    """Verify Python version requirements."""

    def test_python_version_is_311(self):
        """Ensure Python 3.11.x is being used."""
        assert sys.version_info.major == 3, "Python 3.x required"
        assert sys.version_info.minor == 11, (
            f"Python 3.11 required for DGL compatibility, got {sys.version_info.minor}"
        )

    def test_python_not_313(self):
        """Verify we're not accidentally using Python 3.13 (incompatible with DGL)."""
        assert sys.version_info.minor < 13, (
            "Python 3.13 is not compatible with DGL as of 2025"
        )


class TestDependencyVersions:
    """Verify all dependencies meet minimum version requirements."""

    def test_dgl_version(self):
        """Ensure DGL is at least version 2.5.0."""
        try:
            import dgl
        except ImportError:
            pytest.skip("DGL not installed - expected in clean environment")
            return

        assert hasattr(dgl, "__version__"), "DGL version not available"
        version_parts = dgl.__version__.split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1])
        assert (major, minor) >= (2, 5), (
            f"DGL 2.5.0+ required, got {dgl.__version__}"
        )

    def test_pytorch_version(self):
        """Ensure PyTorch is at least version 2.7.0."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed - expected in clean environment")
            return

        assert hasattr(torch, "__version__"), "PyTorch version not available"
        version = torch.__version__.split("+")[0]  # Remove +cpu/+cu121 suffix
        version_parts = version.split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1])
        assert (major, minor) >= (2, 7), (
            f"PyTorch 2.7.0+ required, got {torch.__version__}"
        )

    def test_networkx_version(self):
        """Ensure NetworkX is at least version 3.5."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("NetworkX not installed - expected in clean environment")
            return

        version_parts = nx.__version__.split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1])
        assert (major, minor) >= (3, 5), (
            f"NetworkX 3.5+ required, got {nx.__version__}"
        )

    def test_sklearn_version(self):
        """Ensure scikit-learn is at least version 1.6.0."""
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not installed - expected in clean environment")
            return

        version_parts = sklearn.__version__.split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1])
        assert (major, minor) >= (1, 6), (
            f"scikit-learn 1.6.0+ required, got {sklearn.__version__}"
        )


@pytest.mark.requires_dgl
class TestDGLPyTorchCompatibility:
    """Test DGL and PyTorch integration."""

    def test_dgl_backend_pytorch(self):
        """Verify DGL is using PyTorch backend."""
        import dgl

        assert dgl.backend.backend_name == "pytorch", (
            f"DGL should use PyTorch backend, got {dgl.backend.backend_name}"
        )

    def test_create_simple_graph(self):
        """Test basic DGL graph creation with PyTorch tensors."""
        import dgl
        import torch

        # Create a simple graph
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        g = dgl.graph((src, dst))

        assert g.num_nodes() == 3
        assert g.num_edges() == 3

    @pytest.mark.cuda
    def test_gpu_availability(self):
        """Test CUDA availability if marker is set."""
        import torch

        assert torch.cuda.is_available(), "CUDA not available but test requires it"


class TestModernPythonFeatures:
    """Verify modern Python 3.11 features work correctly."""

    def test_structural_pattern_matching(self):
        """Test Python 3.10+ pattern matching works."""

        def classify_optimizer(name: str) -> str:
            match name:
                case "SGD" | "sgd":
                    return "gradient_descent"
                case "Adam" | "adam":
                    return "adaptive"
                case _:
                    return "unknown"

        assert classify_optimizer("SGD") == "gradient_descent"
        assert classify_optimizer("Adam") == "adaptive"
        assert classify_optimizer("other") == "unknown"

    def test_enhanced_error_messages(self):
        """Verify Python 3.11 enhanced error messages work (implicit test)."""
        # Python 3.11 automatically provides better error messages
        # This test just verifies the interpreter supports them
        assert sys.version_info >= (3, 11), "Python 3.11+ required for enhanced errors"

    def test_tomllib_available(self):
        """Test tomllib (new in Python 3.11) is available."""
        import tomllib

        # Test basic TOML parsing
        toml_str = b"""
        [project]
        name = "grail"
        version = "1.0.0"
        """
        config = tomllib.loads(toml_str.decode())
        assert config["project"]["name"] == "grail"


class TestIdentityReplacement:
    """Test that torch.nn.Identity works as expected."""

    def test_identity_layer(self):
        """Verify torch.nn.Identity works as drop-in replacement."""
        import torch
        import torch.nn as nn

        identity = nn.Identity()
        x = torch.randn(10, 5)
        output = identity(x)

        assert torch.equal(x, output), "Identity should return input unchanged"

    def test_identity_in_sequential(self):
        """Test Identity in Sequential module."""
        import torch
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Identity(),  # Should have no effect
            nn.Linear(20, 10),
        )

        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 10)
