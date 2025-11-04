# GRAIL Python 3.11 Upgrade Guide

**Last Updated**: 2025-11-04  
**Python Version**: 3.11.10  
**Tested Environments**: Ubuntu 20.04+, macOS 10.15+, Windows 10+

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Pre-Upgrade Checklist](#pre-upgrade-checklist)
- [Installation Instructions](#installation-instructions)
  - [Step 1: Install Python 3.11](#step-1-install-python-311)
  - [Step 2: Create Virtual Environment](#step-2-create-virtual-environment)
  - [Step 3: Install Dependencies](#step-3-install-dependencies)
- [Platform-Specific Instructions](#platform-specific-instructions)
  - [macOS](#macos)
  - [Ubuntu/Debian Linux](#ubuntudebian-linux)
  - [Windows](#windows)
  - [GPU/CUDA Support](#gpucuda-support)
- [Verification Steps](#verification-steps)
- [Troubleshooting](#troubleshooting)
- [Rollback Procedures](#rollback-procedures)
- [What's New](#whats-new)
- [Migration Notes](#migration-notes)
- [Getting Help](#getting-help)

## Overview

This guide helps you upgrade the GRAIL (Graph Representation And Inductive Learning) project to Python 3.11 with modernized dependencies:

| Component | Old Version | New Version | Notes |
|-----------|-------------|-------------|-------|
| **Python** | 3.10 or system | **3.11.10** | Required for DGL compatibility |
| **DGL** | 2.1.0 | **2.2.1** | Stable version, Python 3.11 max |
| **PyTorch** | 2.2.0 | **2.2.1** | Stable release |
| **torchvision** | 0.17.0 | **0.17.1** | Compatible with PyTorch 2.2.1 |
| **NetworkX** | 3.0+ | **3.5+** | Latest compatible |
| **scikit-learn** | unversioned | **1.6.0+** | Latest stable |
| **tqdm** | 4.43.0 | **4.67.0+** | 4+ years of updates |
| **lmdb** | 0.98 | **1.5.1** | Latest version |
| **ogb** | git source | **1.3.6+** | Official PyPI package |

### Why Upgrade?

- **Performance**: Python 3.11 is 10-60% faster than Python 3.10
- **Security**: Latest dependencies include critical security patches from 2024-2025
- **Stability**: Updated libraries with bug fixes and improved memory management
- **Better Debugging**: Python 3.11's enhanced error messages significantly improve development experience
- **Future-Ready**: Prepared for eventual migration to Python 3.13 when DGL adds support

### Why Python 3.11 (Not 3.13)?

**DGL 2.2.1 does not support Python 3.13 as of 2025.** Python 3.11 is the latest version fully compatible with DGL. This guide will be updated when DGL adds Python 3.13 support.

## Prerequisites

### Required Software

- **Python 3.11.x**: Must be installed on your system
- **pip**: Latest version (will be upgraded during setup)
- **Git**: For cloning the repository
- **Virtual environment tool**: `venv` (built-in with Python)

### System Requirements

- **CPU**: Any modern multi-core processor
- **RAM**: Minimum 8GB (16GB+ recommended for large graphs)
- **Disk**: 10GB+ free space for dependencies and datasets
- **OS**: 
  - macOS 10.15 (Catalina) or later
  - Ubuntu 20.04 LTS or later
  - Windows 10 or later
- **GPU** (Optional): NVIDIA GPU with CUDA 12.1+ for GPU acceleration

### Knowledge Requirements

- Basic command line usage
- Familiarity with Python virtual environments
- Understanding of pip package management

## Pre-Upgrade Checklist

Before starting the upgrade, complete these steps:

### 1. Backup Current Environment

```bash
# Navigate to project directory
cd /Users/minhbui/Personal/Project/Master/PAPERS/RASG/source/grail

# Save current dependencies
pip freeze > pre_upgrade_environment.txt

# Backup requirements file
cp requirements.txt requirements.txt.backup

# Backup any important model checkpoints
mkdir -p backups/models
cp -r saved_models/* backups/models/ 2>/dev/null || echo "No saved models to backup"
```

### 2. Check Current Python Version

```bash
# Check system Python version
python --version
python3 --version

# If you have Python 3.11 already installed
python3.11 --version
```

### 3. Review Installed Packages

```bash
# List currently installed ML packages
pip list | grep -E "torch|dgl|networkx|sklearn"
```

### 4. Commit or Stash Changes

```bash
# Check for uncommitted changes
git status

# Commit changes
git add .
git commit -m "Pre-upgrade checkpoint"

# Or stash if not ready to commit
git stash save "Pre-upgrade changes"
```

## Installation Instructions

### Step 1: Install Python 3.11

Choose the method appropriate for your operating system:

#### Using pyenv (Recommended - All Platforms)

```bash
# Install pyenv if not already installed
# macOS:
brew install pyenv

# Linux:
curl https://pyenv.run | bash

# Install Python 3.11.10
pyenv install 3.11.10

# Set as local version for this project
cd /Users/minhbui/Personal/Project/Master/PAPERS/RASG/source/grail
pyenv local 3.11.10

# Verify
python --version  # Should show: Python 3.11.10
```

#### Using System Package Manager

**macOS (Homebrew):**
```bash
brew install python@3.11
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**Windows:**
Download the official installer from [python.org](https://www.python.org/downloads/release/python-31110/)

### Step 2: Create Virtual Environment

```bash
# Navigate to project directory
cd /Users/minhbui/Personal/Project/Master/PAPERS/RASG/source/grail

# Remove old virtual environment if it exists
rm -rf .venv

# Create new virtual environment with Python 3.11
python3.11 -m venv .venv

# Activate virtual environment
# macOS/Linux:
source .venv/bin/activate

# Windows (Command Prompt):
# .venv\Scripts\activate.bat

# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Verify correct Python version
python --version  # Should show: Python 3.11.x

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel
```

**Important**: Always activate the virtual environment before working on the project!

### Step 3: Install Dependencies

Dependencies must be installed in a specific order to ensure compatibility:

#### 3a. Install PyTorch First

PyTorch is the foundation that DGL depends on. Choose based on your system:

**macOS:**
```bash
pip install torch==2.2.1 torchvision==0.17.1
```

**Linux (CPU or GPU):**
```bash
pip install torch==2.2.1 torchvision==0.17.1
# Note: GPU support depends on your CUDA installation
```

**Verify PyTorch installation:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 3b. Install DGL

DGL installation differs by platform:

**macOS:**
```bash
pip install dgl==2.2.1
```

**Linux:**
```bash
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
```

**Note**: It's recommended to use the platform-specific requirements files (see step 3c).

**Verify DGL installation:**
```bash
python -c "import dgl; print(f'DGL {dgl.__version__}')"
python -c "import dgl; print(f'DGL backend: {dgl.backend.backend_name}')"
```

#### 3c. Install Platform-Specific Dependencies (Recommended)

Instead of installing packages individually, use the platform-specific requirements file:

**macOS:**
```bash
pip install -r requirements-mac.txt
```

**Linux:**
```bash
pip install -r requirements-linux.txt
```

These files include all necessary dependencies:
- torch==2.2.1, torchvision==0.17.1
- dgl==2.2.1 (with correct installation method for the platform)
- networkx, scikit-learn, tqdm, lmdb, ogb, numpy

#### 3d. Install Development Tools (Optional but Recommended)

```bash
pip install -r requirements-dev.txt
```

This installs:
- ruff (linting and formatting)
- mypy (type checking)
- pytest (testing framework)
- pytest-cov (test coverage)
- pre-commit (git hooks)

#### 3e. Setup Pre-commit Hooks (Optional)

```bash
pre-commit install
```

This enables automatic code quality checks before each commit.

#### 3f. Save Final Environment

```bash
pip freeze > post_upgrade_environment.txt
```

## Platform-Specific Instructions

### macOS

#### Prerequisites
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Python Installation
```bash
# Install Python 3.11
brew install python@3.11

# Verify installation
python3.11 --version
```

#### Common Issues
- **"ssl module not available"**: Reinstall Python with `brew reinstall python@3.11`
- **"command not found: python3.11"**: Add to PATH: `export PATH="/usr/local/opt/python@3.11/bin:$PATH"`

### Ubuntu/Debian Linux

#### Prerequisites
```bash
# Update package list
sudo apt update

# Install build dependencies
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
```

#### Python Installation
```bash
# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev

# Verify installation
python3.11 --version
```

#### GPU Support (NVIDIA)
```bash
# Check CUDA version
nvidia-smi

# Install CUDA 12.1 if needed (example for Ubuntu 22.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-12-1
```

### Windows

#### Prerequisites
- Install Visual Studio Build Tools (for compiling some packages)
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

#### Python Installation
1. Download Python 3.11.10 installer from [python.org](https://www.python.org/downloads/release/python-31110/)
2. Run installer with these options:
   - ✅ Add Python to PATH
   - ✅ Install pip
   - ✅ Install for all users (optional)
3. Verify installation:
   ```cmd
   python --version
   ```

#### Virtual Environment Setup
```cmd
# Navigate to project directory
cd C:\path\to\grail

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate.bat

# Or in PowerShell:
.venv\Scripts\Activate.ps1
```

#### Common Issues
- **"execution of scripts is disabled"**: Run PowerShell as Administrator and execute:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- **Long path issues**: Enable long paths in Windows:
  ```cmd
  reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
  ```

### GPU/CUDA Support

#### Checking Your CUDA Version

```bash
# Check NVIDIA driver version and CUDA capability
nvidia-smi

# Check CUDA runtime version
nvcc --version

# In Python, check PyTorch's CUDA version
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
```

#### CUDA Compatibility Matrix

| CUDA Version | PyTorch 2.2.1 | DGL 2.2.1 | Notes |
|--------------|---------------|-----------|-------|
| 12.1 | ✅ Supported | ✅ Supported | Recommended for newer GPUs |
| 11.8 | ✅ Supported | ✅ Supported | Supported for older GPUs |

#### Installing CUDA-Enabled Packages

**GPU Installation (Linux):**
```bash
# Uninstall any existing versions
pip uninstall -y torch torchvision dgl

# Install using Linux requirements file
pip install -r requirements-linux.txt

# Verify GPU is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Devices: {torch.cuda.device_count()}')"
python -c "import torch; print(f'Current Device: {torch.cuda.get_device_name(0)}')"
```

**Note**: PyTorch 2.2.1 and DGL 2.2.1 work with CUDA 11.8 or 12.1. The Linux requirements file handles the correct installation.

#### Troubleshooting GPU Issues

**Issue**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. **Check NVIDIA Driver**: Ensure NVIDIA drivers are installed
   ```bash
   nvidia-smi
   ```

2. **CUDA Version Mismatch**: Ensure PyTorch CUDA version matches system CUDA
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   nvcc --version  # Compare versions
   ```

3. **Reinstall with Correct Version**:
   ```bash
   pip uninstall torch torchvision dgl
   pip install -r requirements-linux.txt  # For Linux
   # or
   pip install -r requirements-mac.txt    # For macOS
   ```

## Verification Steps

After completing the installation, verify everything works correctly:

### 1. Environment Verification

```bash
# Ensure virtual environment is activated
which python  # Should point to .venv/bin/python

# Check Python version
python --version  # Should be 3.11.x

# Verify pip is from virtual environment
which pip  # Should point to .venv/bin/pip
```

### 2. Package Version Verification

```bash
# Run compatibility tests
pytest tests/test_python_311_compatibility.py -v

# Or manually check versions:
python -c "import sys; assert sys.version_info[:2] == (3, 11), 'Python 3.11 required'"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import dgl; print(f'DGL: {dgl.__version__}')"
python -c "import networkx as nx; print(f'NetworkX: {nx.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
```

### 3. DGL-PyTorch Integration Test

```bash
# Test basic DGL functionality
python -c "
import torch
import dgl

# Create a simple graph
src = torch.tensor([0, 1, 2])
dst = torch.tensor([1, 2, 0])
g = dgl.graph((src, dst))

print(f'Graph created: {g.num_nodes()} nodes, {g.num_edges()} edges')
print(f'DGL backend: {dgl.backend.backend_name}')
print('✓ DGL-PyTorch integration working')
"
```

### 4. GPU Verification (If Applicable)

```bash
# Check CUDA availability
python -c "
import torch
import dgl

print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    
    # Test DGL on GPU
    import dgl
    g = dgl.graph(([0, 1, 2], [1, 2, 0])).to('cuda:0')
    print(f'✓ DGL GPU support working')
"
```

### 5. Full Test Suite (Optional)

```bash
# Run all tests
pytest -v

# Run with coverage report
pytest --cov=. --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### 6. Code Quality Checks (Optional)

```bash
# Run linter
ruff check .

# Run formatter (check only, no changes)
ruff format --check .

# Run type checker
mypy . --ignore-missing-imports
```

## Troubleshooting

### Common Installation Issues

#### Issue 1: "No matching distribution found for dgl==2.2.1"

**Cause**: DGL installation requires platform-specific method.

**Solution**:
```bash
# Verify Python version is exactly 3.11.x
python --version

# Use platform-specific requirements file
# macOS:
pip install -r requirements-mac.txt

# Linux:
pip install -r requirements-linux.txt

# Or install DGL directly:
# macOS:
pip install dgl==2.2.1

# Linux:
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
```

#### Issue 2: "ImportError: cannot import name 'mean_nodes' from 'dgl'"

**Cause**: DGL API has changed between versions.

**Solution**:
```python
# Old API (DGL < 2.0):
from dgl import mean_nodes
result = mean_nodes(g, 'feat')

# New API (DGL 2.2+):
import dgl
result = dgl.readout_nodes(g, 'feat', op='mean')
# Or if mean_nodes still exists:
result = dgl.mean_nodes(g, 'feat')
```

#### Issue 3: "RuntimeError: PyTorch and DGL CUDA versions don't match"

**Cause**: PyTorch and DGL built with different CUDA versions.

**Solution**:
```bash
# Completely uninstall
pip uninstall -y torch torchvision dgl

# Reinstall using platform requirements
# For Linux (handles CUDA properly):
pip install -r requirements-linux.txt

# For macOS:
pip install -r requirements-mac.txt

# Verify CUDA versions match
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
```

#### Issue 4: "OSError: [Errno 28] No space left on device"

**Cause**: Insufficient disk space for installing dependencies.

**Solution**:
```bash
# Check available disk space
df -h

# Clean pip cache
pip cache purge

# Remove unnecessary files
rm -rf ~/.cache/pip
rm -rf ~/.cache/torch

# Free up space and retry installation
```

#### Issue 5: "AttributeError: module 'torch.nn' has no attribute 'Identity'"

**Cause**: Old code using custom Identity class not yet updated.

**Solution**:
```bash
# The project should already be updated, but if you see this error:
# Update model/dgl/layers.py to use torch.nn.Identity instead of custom Identity
# See migration notes below
```

#### Issue 6: Virtual Environment Activation Fails on Windows

**Cause**: PowerShell execution policy restrictions.

**Solution**:
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or use Command Prompt instead:
.venv\Scripts\activate.bat
```

### Dependency Conflicts

#### Issue: "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed"

**Solution**:
```bash
# Start fresh
pip uninstall -r requirements.txt -y
pip uninstall -r requirements-dev.txt -y

# Install in order using platform-specific requirements
# For macOS:
pip install -r requirements-mac.txt
# For Linux:
pip install -r requirements-linux.txt
# Then dev dependencies:
pip install -r requirements-dev.txt
```

### Performance Issues

#### Issue: Training is slower after upgrade

**Potential Causes & Solutions**:

1. **PyTorch compilation not enabled**:
```python
# Add to your training script
import torch
model = torch.compile(model, mode="default")  # Compile model for speed
```

2. **TF32 not enabled** (NVIDIA Ampere+ GPUs):
```python
# Add at the start of training script
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

3. **Suboptimal CUDA version**:
```bash
# Ensure using CUDA 12.1 or 12.4, not older versions
python -c "import torch; print(torch.version.cuda)"
```

### Test Failures

#### Issue: Tests fail with "ModuleNotFoundError"

**Solution**:
```bash
# Install the project in editable mode
pip install -e .

# Or ensure you're in the project root when running tests
cd /Users/minhbui/Personal/Project/Master/PAPERS/RASG/source/grail
pytest
```

#### Issue: Tests fail with deprecation warnings

**Solution**:
```bash
# Update pytest.ini to be more permissive during migration
# Tests should pass even with warnings during initial upgrade
pytest --disable-warnings -v
```

## Rollback Procedures

If you encounter critical issues and need to rollback:

### Option 1: Restore Previous Virtual Environment

```bash
# Deactivate current environment
deactivate

# Remove new environment
rm -rf .venv

# Restore old Python version (if using pyenv)
pyenv local 3.10.0  # or your previous version

# Create new environment with old Python
python -m venv .venv
source .venv/bin/activate

# Restore old dependencies
pip install -r requirements.txt.backup
# Or from frozen environment:
pip install -r pre_upgrade_environment.txt
```

### Option 2: Use Git to Restore Files

```bash
# Restore requirements files
git restore requirements.txt requirements-dev.txt

# Restore configuration files
git restore pyproject.toml .pre-commit-config.yaml .python-version .editorconfig

# Remove any new files
git clean -fd
```

### Option 3: Full Repository Reset

```bash
# Stash any work in progress
git stash save "Rollback stash"

# Reset to pre-upgrade commit
git log --oneline  # Find the commit before upgrade
git reset --hard <commit-hash>

# Reinstall old dependencies
pip install -r requirements.txt
```

### Verify Rollback

```bash
# Check Python version
python --version

# Check package versions
pip list | grep -E "torch|dgl"

# Run tests to ensure everything works
pytest -v
```

## What's New

### Python 3.11 Features

#### 1. Enhanced Error Messages

Python 3.11 provides **significantly better error messages** with precise locations:

**Before (Python 3.10)**:
```
TypeError: 'str' object is not callable
```

**After (Python 3.11)**:
```
TypeError: 'str' object is not callable
  model = "ResNet"()
          ^^^^^^^^^^^
```

The arrow points exactly to the problem!

#### 2. Performance Improvements

- **10-60% faster** than Python 3.10
- Faster function calls
- Better memory efficiency
- Optimized interpreter startup

#### 3. New Standard Library Features

**tomllib** - Built-in TOML parsing:
```python
import tomllib  # No external dependency needed!

with open("config.toml", "rb") as f:
    config = tomllib.load(f)
```

#### 4. Exception Groups

Handle multiple exceptions elegantly:
```python
try:
    results = [process_graph(g) for g in graphs]
except* ValueError as eg:
    for exc in eg.exceptions:
        logging.error(f"Processing error: {exc}")
```

### Library Updates

#### PyTorch 2.2.1

- Stable release with proven reliability
- Good CUDA 11.8 and 12.1 support
- Enhanced memory management
- Support for modern GPU architectures
- Efficient CPU operations

#### DGL 2.2.1

- Optimized message passing for sparse graphs
- Good PyTorch 2.x integration
- Efficient sampling algorithms
- Memory efficiency improvements
- Solid heterogeneous graph support

#### Other Libraries

- **NetworkX 3.5**: Better graph algorithms, performance improvements
- **scikit-learn 1.6.0**: New algorithms, improved performance
- **tqdm 4.67.0**: Rich output, better progress estimation

### Code Quality Tools

#### Ruff

**Lightning-fast** linter and formatter:
- 10-100x faster than traditional tools
- Replaces Black, isort, Flake8, and more
- Automatic code fixing
- Pre-commit integration

**Usage**:
```bash
ruff check .      # Lint code
ruff format .     # Format code
ruff check --fix . # Auto-fix issues
```

#### Pre-commit Hooks

Automatic checks before every commit:
- Trailing whitespace removal
- Import sorting
- Code formatting
- Type checking
- Security linting

## Migration Notes

### Code Changes Required

#### 1. Identity Class Replacement

The custom `Identity` class has been removed in favor of PyTorch's built-in version.

**Old Code**:
```python
from model.dgl.layers import Identity

if edge_dropout:
    self.edge_dropout = nn.Dropout(edge_dropout)
else:
    self.edge_dropout = Identity()
```

**New Code**:
```python
import torch.nn as nn

if edge_dropout:
    self.edge_dropout = nn.Dropout(edge_dropout)
else:
    self.edge_dropout = nn.Identity()  # Built-in since PyTorch 1.2
```

#### 2. DGL API Updates

Some DGL functions have new APIs in version 2.2.1 and later.

**Check for updated imports**:
```python
# If you see errors with mean_nodes, try:
import dgl

# New API (recommended):
result = dgl.readout_nodes(graph, 'feat', op='mean')

# Or check if old API still works:
result = dgl.mean_nodes(graph, 'feat')
```

#### 3. Model Loading Security

Use `weights_only=True` for secure model loading:

**Old Code**:
```python
model = torch.load("model.pth")
```

**New Code** (Secure):
```python
model = torch.load("model.pth", weights_only=True)
```

### Backward Compatibility

#### Model Checkpoints

PyTorch maintains backward compatibility:
- ✅ Models saved with PyTorch 2.2.0 can be loaded with PyTorch 2.2.1
- ✅ No conversion required for most models
- ⚠️  Use `weights_only=False` if you encounter issues (but verify trust)

#### Data Formats

- ✅ No changes to data formats
- ✅ Existing datasets work without modification
- ✅ LMDB databases remain compatible

#### Configuration Files

- ✅ Training configurations remain compatible
- ✅ No changes to command-line arguments
- ✅ Existing experiment scripts work as-is

## Getting Help

### Documentation Resources

- **DGL Documentation**: https://docs.dgl.ai/
- **PyTorch Documentation**: https://pytorch.org/docs/stable/
- **Python 3.11 What's New**: https://docs.python.org/3/whatsnew/3.11.html
- **Ruff Documentation**: https://docs.astral.sh/ruff/

### Project-Specific Help

- **Specification**: See `specs/feat-python-3-11-upgrade.md` for detailed technical information
- **Dependencies**: See `DEPENDENCIES.md` (if available) for dependency explanations
- **Development**: See `DEVELOPMENT.md` (if available) for development workflows

### Troubleshooting Steps

1. **Check this guide's troubleshooting section** for common issues
2. **Review the specification** at `specs/feat-python-3-11-upgrade.md`
3. **Run diagnostic commands**:
   ```bash
   python --version
   pip list | grep -E "torch|dgl"
   pytest tests/test_python_311_compatibility.py -v
   ```
4. **Check GitHub issues** for DGL and PyTorch for known issues
5. **Create an issue** in the project repository with:
   - Your platform (OS, Python version)
   - Complete error message
   - Steps to reproduce
   - Output of `pip list`

### Community Resources

- **DGL Forum**: https://discuss.dgl.ai/
- **PyTorch Forums**: https://discuss.pytorch.org/
- **Stack Overflow**: Tag questions with `dgl`, `pytorch`, `python-3.11`

---

## Quick Reference Commands

```bash
# Verify installation
python --version                    # Should be 3.11.x
pip list | grep -E "torch|dgl"      # Check versions

# Run compatibility tests
pytest tests/test_python_311_compatibility.py -v

# Test GPU support (if applicable)
python -c "import torch; print(torch.cuda.is_available())"

# Run full test suite
pytest -v

# Code quality checks
ruff check .
ruff format .

# Activate virtual environment
source .venv/bin/activate           # macOS/Linux
.venv\Scripts\activate.bat          # Windows

# Update dependencies
pip install --upgrade -r requirements.txt
```

---

**Last Updated**: 2025-11-04  
**Guide Version**: 1.0.0  
**Questions or Issues?** Check the [Troubleshooting](#troubleshooting) section or create an issue in the repository.
