# GRAIL Dependencies Guide

**Last Updated**: 2025-11-04  
**Python Version**: 3.11  
**Tested Environments**: Ubuntu 20.04+, macOS 10.15+, Windows 10+

## Overview

This document provides a comprehensive guide to GRAIL's dependencies, explaining the rationale behind version choices, compatibility requirements, and upgrade paths. Understanding these dependencies is crucial for successful development and deployment of the GRAIL (Graph Representation And Inductive Learning) framework.

## Dependency Strategy

GRAIL's dependency management follows a **foundation-first approach**:

1. **DGL (Deep Graph Library)** is the foundation package - all other dependencies must be compatible with DGL's requirements
2. **PyTorch** is the deep learning backend - must align with DGL's supported versions
3. **Scientific computing packages** (NumPy, NetworkX, scikit-learn) must be compatible with both DGL and PyTorch
4. **Python version** is constrained by DGL support (currently Python 3.11 maximum)

This approach ensures a stable, compatible ecosystem while maximizing the use of modern features and security updates.

### Version Pinning Philosophy

- **Exact versions (`==`)** for core dependencies (DGL, PyTorch, LMDB, Ruff, Mypy) to ensure reproducibility
- **Minimum versions (`>=`)** for stable ecosystem packages (NetworkX, scikit-learn, tqdm) to allow security patches
- **Version ranges** for NumPy to avoid breaking changes while allowing minor updates
- **No git dependencies** in production - only stable PyPI releases for supply chain security

## Core Dependencies

### DGL (Deep Graph Library) - `dgl==2.5.0`

**Purpose**: Core graph neural network library providing graph operations, message passing, and GNN layers.

**Why this version?**
- Latest stable version as of 2025
- Includes performance optimizations for sparse graphs
- Better integration with PyTorch 2.x features
- Security patches and bug fixes from 2024-2025

**Why pinned exactly?**
- DGL versions can have subtle API changes that affect model behavior
- Ensures reproducible graph computations across environments
- Prevents unexpected breaking changes in graph algorithms

**Python compatibility**: Python 3.8 - 3.11 (no Python 3.13 support yet)

**Key features used in GRAIL**:
- Relational Graph Convolutional Networks (RGCN)
- Graph batching and data loading
- Message passing framework
- Heterogeneous graph support

**Installation notes**:
```bash
# CPU-only version
pip install dgl==2.5.0 -f https://data.dgl.ai/wheels/torch-2.7/repo.html

# CUDA 12.1 version (check your CUDA version with nvidia-smi)
pip install dgl==2.5.0 -f https://data.dgl.ai/wheels/torch-2.7/cu121/repo.html
```

**References**:
- [DGL Documentation](https://docs.dgl.ai/)
- [DGL Installation Guide](https://www.dgl.ai/dgl_docs/install/index.html)
- [DGL Releases](https://github.com/dmlc/dgl/releases)

### PyTorch - `torch==2.7.0`

**Purpose**: Deep learning framework providing tensor operations, automatic differentiation, and neural network modules.

**Why this version?**
- Latest stable release as of 2025
- Performance improvements over PyTorch 2.2.0 (previous version)
- Enhanced `torch.compile` support for faster training
- Security patches addressing 2024-2025 CVEs
- Better memory management and CUDA 12.x support

**Why pinned exactly?**
- Model serialization compatibility (ensure saved models load correctly)
- Reproducible training runs
- Consistent numerical behavior across environments

**Python compatibility**: Python 3.8 - 3.13

**Key features used in GRAIL**:
- `torch.nn.Module` for model architecture
- `torch.optim` for optimization algorithms
- Automatic differentiation with autograd
- GPU acceleration with CUDA
- Model checkpointing with `torch.save/load`

**Performance optimizations available**:
```python
# Enable TF32 for faster training on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Compile model for faster execution (PyTorch 2.x feature)
model = torch.compile(model, mode="default")
```

**Security note**: Always use `weights_only=True` when loading untrusted models:
```python
model = torch.load("checkpoint.pth", weights_only=True)
```

**References**:
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch 2.7 Release Notes](https://pytorch.org/blog/pytorch2-6/)

### TorchVision - `torchvision==0.19.0`

**Purpose**: Computer vision utilities and pre-trained models (dependency for some DGL functionality).

**Why this version?**
- Compatible with PyTorch 2.7.0
- May be used for future visualization features

**Note**: Not heavily used in current GRAIL implementation but maintained for ecosystem compatibility.

### NumPy - `numpy>=1.26.0,<2.0.0`

**Purpose**: Fundamental package for numerical computing and array operations.

**Why this version range?**
- `>=1.26.0`: Ensures modern features and security patches
- `<2.0.0`: Avoids NumPy 2.0 breaking changes (major API overhaul)
- Compatible with DGL 2.5.0 and PyTorch 2.7.0

**Why not NumPy 2.0?**
- NumPy 2.0 introduced significant breaking changes
- Many scientific packages are still transitioning to NumPy 2.0 support
- DGL and PyTorch have specific NumPy 1.x requirements for stability

**Migration to NumPy 2.0**: Monitor DGL release notes - will migrate when officially supported.

**References**:
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)

### NetworkX - `networkx>=3.5`

**Purpose**: Graph theory and network analysis library for graph manipulation and algorithms.

**Why this version?**
- Latest stable version with Python 3.11 support
- Performance improvements in graph algorithms
- Used for graph preprocessing and analysis

**Key features used in GRAIL**:
- Graph construction and manipulation
- Graph analysis algorithms
- Conversion between graph formats
- Subgraph extraction

**References**:
- [NetworkX Documentation](https://networkx.org/documentation/stable/)

### scikit-learn - `scikit-learn>=1.6.0`

**Purpose**: Machine learning utilities for metrics, preprocessing, and evaluation.

**Why this version?**
- Latest stable version with comprehensive ML algorithms
- Performance optimizations and bug fixes
- Python 3.11 compatible

**Key features used in GRAIL**:
- Evaluation metrics (accuracy, precision, recall, F1)
- Data splitting and cross-validation
- Preprocessing utilities

**References**:
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

### LMDB - `lmdb==1.5.1`

**Purpose**: Lightning Memory-Mapped Database for efficient on-disk data storage and retrieval.

**Why this version?**
- Latest stable version
- High-performance key-value store for large graph datasets
- Memory-mapped I/O for fast data access

**Key features used in GRAIL**:
- Caching preprocessed graphs
- Efficient dataset storage for large knowledge graphs
- Random access to graph samples during training

**References**:
- [LMDB Documentation](https://lmdb.readthedocs.io/)

### tqdm - `tqdm>=4.67.0`

**Purpose**: Progress bar library for training monitoring and data processing.

**Why this version?**
- Modern version with rich output formatting
- Significant upgrade from tqdm 4.43.0 (previous version from 2020)
- Better Jupyter notebook support
- Performance improvements

**Key features used in GRAIL**:
- Training progress bars
- Data loading progress indicators
- Epoch and batch tracking

**References**:
- [tqdm Documentation](https://tqdm.github.io/)

### OGB (Open Graph Benchmark) - `ogb>=1.3.6`

**Purpose**: Standardized benchmark datasets for graph machine learning.

**Why this version?**
- Latest official PyPI release
- Migrated from git source for better supply chain security
- Includes latest datasets and evaluation protocols

**Key features used in GRAIL**:
- Knowledge graph datasets (WN18RR, FB15k-237, etc.)
- Standardized evaluation metrics
- Dataset loaders and preprocessors

**Migration note**: Previously installed from git repository, now using official PyPI package for stability.

**References**:
- [OGB Documentation](https://ogb.stanford.edu/)
- [OGB GitHub](https://github.com/snap-stanford/ogb)

## DGL + PyTorch Compatibility Matrix

Understanding the compatibility between DGL and PyTorch is critical for successful installation and operation.

| DGL Version | PyTorch Version | Python Support | CUDA Support | Notes |
|-------------|----------------|----------------|--------------|-------|
| 2.5.0 | 2.7.0 | 3.8 - 3.11 | 11.8, 12.1 | **Current** - Latest stable |
| 2.4.0 | 2.5.0 - 2.6.0 | 3.8 - 3.11 | 11.8, 12.1 | Previous stable |
| 2.3.0 | 2.4.0 - 2.5.0 | 3.8 - 3.11 | 11.8, 12.1 | Older stable |
| 2.2.0 | 2.3.0 - 2.4.0 | 3.8 - 3.11 | 11.7, 11.8 | Legacy |
| 2.1.0 | 2.2.0 | 3.8 - 3.11 | 11.7, 11.8 | **Previous GRAIL** |

### Installation Order Matters

Always install in this order to avoid conflicts:

```bash
# 1. Uninstall old versions (if upgrading)
pip uninstall -y dgl torch torchvision

# 2. Install PyTorch first (DGL depends on it)
pip install torch==2.7.0 torchvision==0.19.0

# 3. Install DGL with appropriate backend
# For CPU:
pip install dgl==2.5.0 -f https://data.dgl.ai/wheels/torch-2.7/repo.html

# For CUDA 12.1:
pip install dgl==2.5.0 -f https://data.dgl.ai/wheels/torch-2.7/cu121/repo.html

# 4. Install remaining dependencies
pip install -r requirements.txt
```

### CUDA Version Detection

Check your CUDA version before installing:

```bash
# Check CUDA version
nvidia-smi

# Look for "CUDA Version: X.Y" in the output
# Common versions:
# - CUDA 11.8: Use cu118 wheels
# - CUDA 12.1: Use cu121 wheels
# - No NVIDIA GPU: Use CPU-only version
```

### Troubleshooting Compatibility Issues

**Problem**: `ImportError: cannot import name 'XXX' from 'dgl'`
- **Cause**: DGL version mismatch or incomplete installation
- **Solution**: Reinstall DGL with correct PyTorch backend

**Problem**: `RuntimeError: CUDA error: no kernel image is available`
- **Cause**: CUDA version mismatch between PyTorch and system
- **Solution**: Reinstall PyTorch with correct CUDA version

**Problem**: `AttributeError: module 'torch' has no attribute 'XXX'`
- **Cause**: PyTorch version too old for DGL features
- **Solution**: Upgrade PyTorch to 2.7.0

## Development Dependencies

Development dependencies are specified in `requirements-dev.txt` and are only needed for contributors.

### Code Quality Tools

#### Ruff - `ruff==0.14.3`

**Purpose**: All-in-one linter and formatter replacing Black, isort, Flake8, and more.

**Why this tool?**
- 10-30x faster than Black for formatting
- Single tool replaces multiple linters
- Auto-fixes most issues
- Python 3.11 syntax support

**Usage**:
```bash
# Check code quality
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

**Configuration**: See `pyproject.toml` for detailed Ruff settings.

**References**:
- [Ruff Documentation](https://docs.astral.sh/ruff/)

#### Mypy - `mypy==1.14.0`

**Purpose**: Static type checker for Python.

**Why this tool?**
- Catches type errors before runtime
- Improves code documentation with type hints
- IDE integration for better autocomplete

**Usage**:
```bash
mypy . --ignore-missing-imports
```

**Note**: Currently configured permissively (`disallow_untyped_defs = false`) to ease adoption. Will become stricter over time.

**References**:
- [Mypy Documentation](https://mypy.readthedocs.io/)

#### Pre-commit - `pre-commit==4.0.0`

**Purpose**: Git hook framework for automated code quality checks.

**Why this tool?**
- Runs checks automatically before commits
- Catches issues early in development
- Ensures consistent code style across contributors

**Setup**:
```bash
# Install hooks (one-time setup)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

**References**:
- [Pre-commit Documentation](https://pre-commit.com/)

### Testing Framework

#### Pytest - `pytest>=8.3.0`

**Purpose**: Modern testing framework for Python.

**Why this version?**
- Latest stable version with Python 3.11 support
- Improved error messages
- Better fixture management

**Usage**:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_python_311_compatibility.py

# Run with verbose output
pytest -v
```

**Configuration**: See `tests/pytest.ini` for detailed pytest settings.

**References**:
- [Pytest Documentation](https://docs.pytest.org/)

#### Pytest Extensions

- **pytest-cov** (`>=6.0.0`): Coverage reporting
- **pytest-xdist** (`>=3.6.0`): Parallel test execution for faster runs
- **pytest-timeout** (`>=2.3.0`): Timeout protection for hanging tests
- **pytest-mock** (`>=3.14.0`): Enhanced mocking utilities

**Parallel testing example**:
```bash
# Run tests on 4 CPU cores
pytest -n 4
```

### Documentation Tools

#### Sphinx - `sphinx>=8.0.0`

**Purpose**: Documentation generation from docstrings.

**Why this tool?**
- Industry-standard for Python documentation
- Generates HTML, PDF, and other formats
- Integrates with Read the Docs

**Status**: Included for future documentation but not yet configured.

**References**:
- [Sphinx Documentation](https://www.sphinx-doc.org/)

## Testing Dependencies

Testing dependencies are specified in `tests/requirements-test.txt` for CI/CD environments.

### Core Testing Tools

- **pytest** (`>=8.3.0`): Test framework
- **pytest-cov** (`>=6.0.0`): Coverage reporting
- **pytest-xdist** (`>=3.6.0`): Parallel execution
- **pytest-timeout** (`>=2.3.0`): Timeout protection

### Enhanced Test Output

- **pytest-sugar** (`>=1.0.0`): Improved progress bar and output formatting
- **pytest-html** (`>=4.1.0`): HTML test reports for CI/CD

### Test Categories

Tests are organized by markers in `tests/pytest.ini`:

```python
@pytest.mark.cuda  # Tests requiring GPU
@pytest.mark.slow  # Tests taking >1 second
@pytest.mark.integration  # End-to-end tests
@pytest.mark.requires_dgl  # Tests needing DGL installation
```

**Run specific test categories**:
```bash
# Run only fast tests (skip slow tests)
pytest -m "not slow"

# Run only CUDA tests (if GPU available)
pytest -m cuda

# Run integration tests
pytest -m integration
```

## Optional Dependencies

These dependencies are not required but may enhance functionality.

### GPU Acceleration

**CUDA Toolkit**: Required for GPU training
- **Version**: 11.8 or 12.1 (must match DGL/PyTorch installation)
- **Installation**: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

**Verification**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

### Development Enhancements

**IPython** (`ipython>=8.0.0`): Enhanced REPL for experimentation
```bash
pip install ipython
```

**Jupyter** (`jupyter>=1.0.0`): Notebook environment for interactive development
```bash
pip install jupyter
```

## Future Upgrade Paths

### Python 3.13 Migration

**Current status**: DGL 2.5.0 does not support Python 3.13 (as of 2025-11-04)

**Migration timeline**: When DGL adds Python 3.13 support (estimated 2025-2026)

**Benefits of Python 3.13**:
- Further performance improvements
- Enhanced error messages
- Improved type system features
- Better debugging capabilities

**Migration steps** (when available):
```bash
# 1. Verify DGL Python 3.13 support
# Check: https://github.com/dmlc/dgl/releases

# 2. Update .python-version
echo "3.13.x" > .python-version

# 3. Update pyproject.toml
# Change: requires-python = ">=3.13,<3.14"

# 4. Create new virtual environment
python3.13 -m venv .venv-313
source .venv-313/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run compatibility tests
pytest tests/test_python_313_compatibility.py

# 7. Verify all tests pass
pytest
```

**Monitoring**: Check DGL releases quarterly for Python 3.13 support announcements.

**References**:
- [Python 3.13 Release Schedule](https://peps.python.org/pep-0719/)
- [DGL GitHub Issues](https://github.com/dmlc/dgl/issues)

### NumPy 2.0 Migration

**Current status**: Blocked by DGL support (using NumPy 1.x)

**Migration timeline**: When DGL adds NumPy 2.0 support

**Breaking changes in NumPy 2.0**:
- Removed deprecated APIs
- Changed default data types
- Modified string handling
- Updated C API

**Migration strategy**:
1. Wait for DGL official NumPy 2.0 support announcement
2. Update version constraint: `numpy>=2.0.0,<3.0.0`
3. Run full test suite to identify breaking changes
4. Update code using deprecated NumPy APIs
5. Validate numerical consistency with previous results

**References**:
- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)

### Continuous Monitoring

**Best practices for staying current**:

1. **Quarterly dependency review**: Check for new releases every 3 months
2. **Security audits**: Run `pip-audit` monthly to detect vulnerabilities
3. **DGL release tracking**: Subscribe to [DGL releases](https://github.com/dmlc/dgl/releases)
4. **PyTorch release notes**: Monitor [PyTorch blog](https://pytorch.org/blog/)

**Automated dependency checking**:
```bash
# Install pip-audit
pip install pip-audit

# Check for known vulnerabilities
pip-audit

# Check for outdated packages
pip list --outdated
```

## Dependency Installation Guide

### Fresh Installation

For new developers setting up the project:

```bash
# 1. Verify Python version
python --version  # Should show Python 3.11.x

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install PyTorch first
pip install torch==2.7.0 torchvision==0.19.0

# 5. Install DGL (choose CPU or GPU version)
# CPU version:
pip install dgl==2.5.0 -f https://data.dgl.ai/wheels/torch-2.7/repo.html

# CUDA 12.1 version:
# pip install dgl==2.5.0 -f https://data.dgl.ai/wheels/torch-2.7/cu121/repo.html

# 6. Install remaining dependencies
pip install -r requirements.txt

# 7. Install development tools (optional)
pip install -r requirements-dev.txt

# 8. Set up pre-commit hooks (optional, for contributors)
pre-commit install

# 9. Verify installation
python -c "import torch, dgl; print(f'PyTorch: {torch.__version__}, DGL: {dgl.__version__}')"

# 10. Run tests to verify setup
pytest tests/test_python_311_compatibility.py
```

### Upgrading from Previous Version

For upgrading existing installations:

```bash
# 1. Backup current environment
pip freeze > pre_upgrade_environment.txt

# 2. Uninstall old DGL and PyTorch
pip uninstall -y dgl torch torchvision

# 3. Follow fresh installation steps 4-10 above

# 4. Compare environments
pip freeze > post_upgrade_environment.txt
diff pre_upgrade_environment.txt post_upgrade_environment.txt
```

### Docker Installation

For containerized deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install PyTorch first
RUN pip install --no-cache-dir torch==2.7.0 torchvision==0.19.0

# Install DGL (CPU version)
RUN pip install --no-cache-dir dgl==2.5.0 -f https://data.dgl.ai/wheels/torch-2.7/repo.html

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

CMD ["python", "train.py"]
```

### Conda Installation (Alternative)

While pip is recommended, Conda installation is also possible:

```bash
# 1. Create conda environment
conda create -n grail python=3.11

# 2. Activate environment
conda activate grail

# 3. Install PyTorch via conda
conda install pytorch==2.7.0 torchvision==0.19.0 -c pytorch

# 4. Install DGL via pip (conda channel may be outdated)
pip install dgl==2.5.0 -f https://data.dgl.ai/wheels/torch-2.7/repo.html

# 5. Install remaining dependencies
pip install -r requirements.txt
```

**Note**: Mixing conda and pip can cause dependency conflicts. Use pip-only installation when possible.

## Troubleshooting

### Common Issues

#### Issue: `ImportError: cannot import name 'DGLGraph' from 'dgl'`

**Cause**: DGL version mismatch or API changes

**Solution**:
```bash
# Verify DGL version
python -c "import dgl; print(dgl.__version__)"

# Reinstall DGL if version is incorrect
pip uninstall -y dgl
pip install dgl==2.5.0 -f https://data.dgl.ai/wheels/torch-2.7/repo.html
```

#### Issue: `RuntimeError: Expected all tensors to be on the same device`

**Cause**: Mixing CPU and GPU tensors

**Solution**:
```python
# Ensure model and data are on the same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

#### Issue: `pip install` fails with dependency conflicts

**Cause**: Conflicting version requirements

**Solution**:
```bash
# Use fresh virtual environment
deactivate  # Exit current environment
rm -rf .venv  # Remove old environment
python -m venv .venv
source .venv/bin/activate

# Follow installation order strictly (PyTorch → DGL → others)
```

#### Issue: Tests fail with `ModuleNotFoundError`

**Cause**: Missing test dependencies

**Solution**:
```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Or install development dependencies
pip install -r requirements-dev.txt
```

#### Issue: CUDA out of memory errors

**Cause**: Batch size too large for GPU memory

**Solution**:
```python
# Reduce batch size in config
# Or use gradient accumulation:
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Getting Help

If you encounter issues not covered here:

1. **Check DGL documentation**: [https://docs.dgl.ai/](https://docs.dgl.ai/)
2. **Search GitHub issues**: [DGL Issues](https://github.com/dmlc/dgl/issues)
3. **PyTorch Forums**: [https://discuss.pytorch.org/](https://discuss.pytorch.org/)
4. **Create an issue**: Include Python version, dependency versions, and full error traceback

### Diagnostic Commands

Collect system information for debugging:

```bash
# Python version
python --version

# Installed packages
pip list

# DGL and PyTorch versions
python -c "import torch, dgl; print(f'PyTorch: {torch.__version__}\nDGL: {dgl.__version__}\nCUDA: {torch.version.cuda}')"

# CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}\nGPU count: {torch.cuda.device_count()}')"

# NumPy version
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# System info
python -m sysconfig
```

## Security Considerations

### Dependency Vulnerabilities

**Regular audits**: Run security audits monthly

```bash
# Install pip-audit
pip install pip-audit

# Check for known vulnerabilities
pip-audit

# Fix vulnerabilities by upgrading
pip install --upgrade <vulnerable-package>
```

### Supply Chain Security

**Best practices**:
1. **Use PyPI packages only**: Avoid git dependencies in production
2. **Verify package signatures**: Use `--require-hashes` for critical deployments
3. **Pin exact versions**: Use `==` for reproducibility
4. **Regular updates**: Review and update dependencies quarterly

**Generate hashes for requirements**:
```bash
# Install pip-tools
pip install pip-tools

# Generate requirements with hashes
pip-compile --generate-hashes requirements.in -o requirements.txt
```

### Secure Model Loading

Always use `weights_only=True` when loading models from untrusted sources:

```python
# Secure model loading (PyTorch 2.x)
checkpoint = torch.load("model.pth", weights_only=True)

# Apply to model
model.load_state_dict(checkpoint)
```

### Dependency Scanning in CI/CD

Add to your CI/CD pipeline:

```yaml
# .github/workflows/security.yml
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install pip-audit
      - name: Run security audit
        run: pip-audit
```

## References

### Official Documentation

- [DGL Documentation](https://docs.dgl.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Python 3.11 Documentation](https://docs.python.org/3.11/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

### Installation Guides

- [DGL Installation Guide](https://www.dgl.ai/dgl_docs/install/index.html)
- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit Installation](https://developer.nvidia.com/cuda-downloads)

### Release Notes & Changelogs

- [DGL Releases](https://github.com/dmlc/dgl/releases)
- [PyTorch Release Notes](https://github.com/pytorch/pytorch/releases)
- [Python 3.11 What's New](https://docs.python.org/3/whatsnew/3.11.html)

### Community Resources

- [DGL GitHub](https://github.com/dmlc/dgl)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Python Package Index](https://pypi.org/)

---

**Document Maintenance**: This document should be reviewed and updated quarterly or when major dependency changes occur. Last review: 2025-11-04.
