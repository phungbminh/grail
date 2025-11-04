# GraIL - Graph Inductive Learning

This is the code necessary to run experiments on GraIL algorithm described in the ICML'20 paper [Inductive relation prediction by subgraph reasoning](https://arxiv.org/abs/1911.06962).

## Requirements

### Python Version
This project requires **Python 3.11** for compatibility with DGL 2.5.0. Python 3.13 is not yet supported by DGL.

### Installation

1. **Install Python 3.11**
   ```bash
   # macOS (Homebrew)
   brew install python@3.11

   # Or use pyenv
   pyenv install 3.11.10
   pyenv local 3.11.10
   ```

2. **Create virtual environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   For **macOS**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements-mac.txt
   # Or use the generic requirements.txt (same as mac)
   # pip install -r requirements.txt
   ```

   For **Linux**:
   ```bash
   pip install --upgrade pip
   # Install DGL first with special wheel repo
   pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
   # Then install remaining dependencies
   pip install -r requirements-linux.txt
   ```

4. **For development** (optional)
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```

### Dependencies

Key dependencies:
- **DGL**: 2.2.1 (Deep Graph Library)
- **PyTorch**: 2.2.1 (with torchvision 0.17.1)
- **NetworkX**: >=3.5
- **scikit-learn**: >=1.6.0
- **NumPy**: >=1.26.0, <2.0.0
- **OGB**: >=1.3.6 (Open Graph Benchmark)

For detailed dependency information, see [DEPENDENCIES.md](DEPENDENCIES.md).

For upgrade instructions from older versions, see [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md).

## Inductive relation prediction experiments

All train-graph and ind-test-graph pairs of graphs can be found in the `data` folder. We use WN18RR_v1 as a running example for illustrating the steps.

### GraIL
To start training a GraIL model, run the following command:
```bash
python train.py -d WN18RR_v1 -e grail_wn_v1
```

To test GraIL run the following commands:
```bash
python test_auc.py -d WN18RR_v1_ind -e grail_wn_v1
python test_ranking.py -d WN18RR_v1_ind -e grail_wn_v1
```

The trained model and the logs are stored in `experiments` folder. Note that to ensure a fair comparison, we test all models on the same negative triplets. In order to do that in the current setup, we store the sampled negative triplets while evaluating GraIL and use these later to evaluate other baseline models.

### RuleN
RuleN operates in two steps. Rules are first learned from a training graph and then applied on the test graph. Detailed instructions can be found [here](http://web.informatik.uni-mannheim.de/RuleN/).
- Learn rules: `source learn_rules.sh WN18RR_v1`
- Apply rules:
	- To get AUC: `source auc_apply_rules.sh WN18RR_v1 WN18RR_v1_ind num_of_samples_to_score(=1000)`
	- To get ranking score: `source ranking_apply_rules.sh WN18RR_v1 WN18RR_v1_ind num_of_samples_to_score(=1000)`

### NeuralLP and Drum
We use the implementations provided by the authors of the respective papers to evaluate these models.

## Transductive experiments

The full transductive datasets used in these experiments are present in the `data` folder.

### GraIL
The training and testing protocols of GraIL remains the same.

### KGE models
We use the comprehensive implementation provided by authors of RotatE. This gives state-of-the-art results on all datasets. The best configurations can be found [here](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/best_config.sh). To train these KGE models, navigate to the `kge` folder and run the commands as shown in the above reference. For example, to train TransE on FB237-15k, run the following command:

```bash
bash run.sh train TransE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
```

This will store the trained model and the logs in a folder named `experiments/kge_baselines/TransE_FB15k-237`.

### Ensembling instructions
Once the KGE models are trained, to get ensembling results with GraIL, navigate to the `ensembling` folder and run the following command:
```bash
source get_ensemble_predictions.sh WN18RR TransE
```

To get ensembling among different KGE models, from the `ensembling` folder run the following command:
```bash
source get_kge_predictions.sh WN18RR TransE ComplEx
```

## Testing

Run the test suite to verify your installation:
```bash
# Basic compatibility tests
pytest tests/test_python_311_compatibility.py -v

# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=. --cov-report=html
```

See [tests/README.md](tests/README.md) for more information about the test suite.

## Development

For development setup and guidelines, install the development dependencies:
```bash
pip install -r requirements-dev.txt
pre-commit install
```

This provides:
- **Ruff** - Fast Python linter and formatter
- **mypy** - Static type checking
- **pytest** - Testing framework with coverage
- **pre-commit** - Automated code quality checks

Run code quality checks:
```bash
ruff check .        # Linting
ruff format .       # Formatting
mypy .             # Type checking (if types are added)
pytest tests/      # Run tests
```

## Citation

If you make use of this code or the GraIL algorithm in your work, please cite the following paper:

```bibtex
@article{Teru2020InductiveRP,
  title={Inductive Relation Prediction by Subgraph Reasoning.},
  author={Komal K. Teru and Etienne Denis and William L. Hamilton},
  journal={arXiv: Learning},
  year={2020}
}
```

## License

See LICENSE file for details.

## Acknowledgments

This implementation is based on the DGL tutorial on RGCN. See individual files for specific attributions.
