import argparse
import os
import sys
import logging
import numpy as np
from sklearn import metrics

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_scores(file_path, score_type='positive'):
    """Load prediction scores from file with robust error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Prediction file not found: {file_path}\n"
            f"Make sure you have run test_auc.py with save=True to generate predictions."
        )

    logging.info(f"Loading {score_type} scores from: {file_path}")

    scores = []
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # Filter empty lines

        if not lines:
            raise ValueError(f"File is empty: {file_path}")

        for line_num, line in enumerate(lines, 1):
            parts = line.split()

            # Validate format: expect at least 4 fields (entity1, relation, entity2, score)
            if len(parts) < 4:
                raise ValueError(
                    f"Invalid format at line {line_num} in {file_path}\n"
                    f"Expected: entity1 relation entity2 score\n"
                    f"Got: {line}"
                )

            try:
                score = float(parts[-1])
                scores.append(score)
            except ValueError as e:
                raise ValueError(
                    f"Cannot parse score at line {line_num} in {file_path}\n"
                    f"Value: '{parts[-1]}'\n"
                    f"Line: {line}"
                ) from e

    logging.info(f"Loaded {len(scores)} {score_type} samples")
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute AUC from scored positive and negative triplets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute AUC for test set
  python compute_auc.py -d ogbl-biokg -m grail -t test

  # Compute AUC for validation set
  python compute_auc.py -d ogbl-biokg -m grail -t valid

  # With custom constrained_neg_prob
  python compute_auc.py -d ogbl-biokg -m grail -t test -cn 0.8
        """
    )

    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='Dataset name (e.g., ogbl-biokg, WN18RR_v1_ind)')
    parser.add_argument('--model', '-m', type=str, default='grail',
                        help='Model name (default: grail)')
    parser.add_argument('--test_file', '-t', type=str, default='test',
                        help='Test file name (default: test, can be: test, valid)')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0,
                        help='Constrained negative sampling probability (default: 0)')

    params = parser.parse_args()

    # Get absolute path to data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', params.dataset)

    if not os.path.exists(data_dir):
        logging.error(f"Dataset directory not found: {data_dir}")
        logging.error(f"Available datasets: {os.listdir(os.path.join(script_dir, '..', 'data'))}")
        sys.exit(1)

    # Construct file paths
    pos_file = os.path.join(data_dir, f"{params.model}_{params.test_file}_predictions.txt")
    neg_file = os.path.join(data_dir, f"{params.model}_neg_{params.test_file}_{params.constrained_neg_prob}_predictions.txt")
    output_file = os.path.join(data_dir, f"{params.model}_{params.test_file}_auc.txt")

    logging.info("="*60)
    logging.info("Computing AUC from prediction files")
    logging.info("="*60)
    logging.info(f"Dataset: {params.dataset}")
    logging.info(f"Model: {params.model}")
    logging.info(f"Test file: {params.test_file}")
    logging.info(f"Constrained neg prob: {params.constrained_neg_prob}")
    logging.info("="*60)

    # Load positive and negative prediction scores
    try:
        pos_scores = load_scores(pos_file, 'positive')
        neg_scores = load_scores(neg_file, 'negative')
    except (FileNotFoundError, ValueError) as e:
        logging.error(str(e))
        sys.exit(1)

    # Validate data
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        logging.error(f"Empty predictions: positive={len(pos_scores)}, negative={len(neg_scores)}")
        sys.exit(1)

    # Combine scores and labels
    scores = pos_scores + neg_scores
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)

    logging.info(f"Total samples: {len(scores)} (pos={len(pos_scores)}, neg={len(neg_scores)})")
    logging.info(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]")

    # Compute AUC metrics with error handling
    try:
        auc = metrics.roc_auc_score(labels, scores)
        auc_pr = metrics.average_precision_score(labels, scores)
    except ValueError as e:
        logging.error(f"Error computing AUC: {e}")
        sys.exit(1)

    # Sanity checks
    if np.isnan(auc) or np.isnan(auc_pr):
        logging.warning("NaN detected in AUC metrics - check your predictions!")

    if auc < 0.55:
        logging.warning(f"AUC={auc:.4f} is very low - might indicate random predictions or issues")

    if auc > 0.99:
        logging.warning(f"AUC={auc:.4f} is suspiciously high - might indicate data leakage")

    # Print results
    logging.info("="*60)
    logging.info("RESULTS")
    logging.info("="*60)
    logging.info(f"AUC:    {auc:.6f}")
    logging.info(f"AUC-PR: {auc_pr:.6f}")
    logging.info("="*60)

    # Save results
    with open(output_file, 'w') as f:
        f.write(f"AUC: {auc:.6f}, AUC_PR: {auc_pr:.6f}\n")
        f.write(f"Positive samples: {len(pos_scores)}\n")
        f.write(f"Negative samples: {len(neg_scores)}\n")
        f.write(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]\n")

    logging.info(f"Results saved to: {output_file}")
    logging.info("Done!")
