#!/usr/bin/env python3
"""
Training Visualization Tool for GraIL
Reads training logs and generates publication-quality plots for thesis/papers.

Usage:
    python utils/visualize_training.py -e grail_biokg_1
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def parse_log_file(log_path):
    """Parse training log file and extract metrics."""

    metrics = {
        'epochs': [],
        'train_loss': [],
        'train_auc': [],
        'train_auc_pr': [],
        'valid_auc': [],
        'valid_auc_pr': [],
        'weight_norm': [],
        'epoch_time': [],
        'best_valid_auc': []
    }

    # Also track validation evaluations within epochs
    validation_history = {
        'iteration': [],
        'auc': [],
        'auc_pr': []
    }

    current_iteration = 0

    with open(log_path, 'r') as f:
        for line in f:
            # Parse epoch summary
            epoch_match = re.search(r'Epoch (\d+) with loss: ([\d.]+), training auc: ([\d.]+), training auc_pr: ([\d.]+), best validation AUC: ([\d.]+), weight_norm: ([\d.]+) in ([\d.]+)', line)
            if epoch_match:
                metrics['epochs'].append(int(epoch_match.group(1)))
                metrics['train_loss'].append(float(epoch_match.group(2)))
                metrics['train_auc'].append(float(epoch_match.group(3)))
                metrics['train_auc_pr'].append(float(epoch_match.group(4)))
                metrics['best_valid_auc'].append(float(epoch_match.group(5)))
                metrics['weight_norm'].append(float(epoch_match.group(6)))
                metrics['epoch_time'].append(float(epoch_match.group(7)))

            # Parse validation during training
            perf_match = re.search(r"Performance:\{'auc': ([\d.]+), 'auc_pr': ([\d.]+)\}", line)
            if perf_match:
                validation_history['iteration'].append(current_iteration)
                validation_history['auc'].append(float(perf_match.group(1)))
                validation_history['auc_pr'].append(float(perf_match.group(2)))
                current_iteration += 1

    # Extract final validation AUC for each epoch
    for epoch in metrics['epochs']:
        # Find the last validation AUC before each epoch
        epoch_validations = [v for i, v in enumerate(validation_history['auc'])
                           if i < len(validation_history['auc'])]
        if epoch_validations:
            metrics['valid_auc'].append(epoch_validations[-1] if len(epoch_validations) >= epoch else 0)
            metrics['valid_auc_pr'].append(validation_history['auc_pr'][-1] if len(validation_history['auc_pr']) >= epoch else 0)
        else:
            metrics['valid_auc'].append(0)
            metrics['valid_auc_pr'].append(0)

    return metrics, validation_history


def plot_training_overview(metrics, save_path):
    """Create comprehensive training overview with 6 subplots."""

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    epochs = np.array(metrics['epochs'])

    # 1. Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, metrics['train_loss'], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 2. AUC Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, metrics['train_auc'], 'b-', linewidth=2, marker='o', markersize=4, label='Train AUC')
    ax2.plot(epochs, metrics['best_valid_auc'], 'r-', linewidth=2, marker='s', markersize=4, label='Valid AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('AUC: Training vs Validation')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.7, 1.0])

    # 3. AUC-PR Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, metrics['train_auc_pr'], 'b-', linewidth=2, marker='o', markersize=4, label='Train AUC-PR')
    # Note: Using best_valid_auc as proxy since we don't track valid_auc_pr separately
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC-PR')
    ax3.set_title('Precision-Recall AUC Over Time')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.7, 1.0])

    # 4. Generalization Gap
    ax4 = fig.add_subplot(gs[1, 1])
    gap = np.array(metrics['train_auc']) - np.array(metrics['best_valid_auc'])
    ax4.plot(epochs, gap * 100, 'g-', linewidth=2, marker='D', markersize=4)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Generalization Gap (%)')
    ax4.set_title('Train-Validation AUC Gap')
    ax4.grid(True, alpha=0.3)
    ax4.fill_between(epochs, 0, gap * 100, alpha=0.3, color='green')

    # 5. Training Time per Epoch
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.bar(epochs, metrics['epoch_time'], color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Time (seconds)')
    ax5.set_title('Training Time per Epoch')
    ax5.grid(True, alpha=0.3, axis='y')

    # Add average line
    avg_time = np.mean(metrics['epoch_time'])
    ax5.axhline(y=avg_time, color='red', linestyle='--', linewidth=2,
                label=f'Average: {avg_time:.1f}s')
    ax5.legend()

    # 6. Weight Norm Evolution
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(epochs, metrics['weight_norm'], 'purple', linewidth=2, marker='v', markersize=4)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('L2 Weight Norm')
    ax6.set_title('Model Weight Norm Evolution')
    ax6.grid(True, alpha=0.3)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f'✓ Saved training overview: {save_path}')
    plt.close()


def plot_convergence_analysis(metrics, save_path):
    """Create convergence analysis plot."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = np.array(metrics['epochs'])

    # Loss convergence
    ax1.semilogy(epochs, metrics['train_loss'], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss (log scale)')
    ax1.set_title('Loss Convergence')
    ax1.grid(True, alpha=0.3, which='both')

    # Find best epoch
    best_epoch = np.argmax(metrics['best_valid_auc']) + 1
    best_auc = max(metrics['best_valid_auc'])

    # AUC convergence with best marker
    ax2.plot(epochs, metrics['train_auc'], 'b-', linewidth=2, marker='o', markersize=4,
             label='Training AUC', alpha=0.7)
    ax2.plot(epochs, metrics['best_valid_auc'], 'r-', linewidth=2, marker='s', markersize=4,
             label='Validation AUC', alpha=0.7)
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best epoch: {best_epoch}')
    ax2.scatter([best_epoch], [best_auc], s=200, c='gold', marker='*',
                edgecolors='red', linewidths=2, zorder=5, label=f'Best AUC: {best_auc:.4f}')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('AUC Convergence')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.7, 1.0])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f'✓ Saved convergence analysis: {save_path}')
    plt.close()


def plot_learning_curve(metrics, save_path):
    """Create learning curve (training vs validation performance)."""

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = np.array(metrics['epochs'])

    # Plot with error bands
    ax.plot(epochs, metrics['train_auc'], 'b-', linewidth=2.5, marker='o',
            markersize=5, label='Training AUC', alpha=0.8)
    ax.plot(epochs, metrics['best_valid_auc'], 'r-', linewidth=2.5, marker='s',
            markersize=5, label='Validation AUC', alpha=0.8)

    # Fill between to show gap
    ax.fill_between(epochs, metrics['train_auc'], metrics['best_valid_auc'],
                     alpha=0.2, color='orange', label='Generalization Gap')

    # Mark best epoch
    best_epoch = np.argmax(metrics['best_valid_auc']) + 1
    best_auc = max(metrics['best_valid_auc'])
    ax.scatter([best_epoch], [best_auc], s=300, c='gold', marker='*',
               edgecolors='darkred', linewidths=2, zorder=5)
    ax.annotate(f'Best: {best_auc:.4f}\nEpoch {best_epoch}',
                xy=(best_epoch, best_auc), xytext=(best_epoch + 3, best_auc - 0.05),
                fontsize=10, ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                              color='black', lw=1.5))

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title('Learning Curve: Training vs Validation Performance', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.7, 1.0])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f'✓ Saved learning curve: {save_path}')
    plt.close()


def generate_training_report(metrics, save_path):
    """Generate text summary report."""

    best_epoch = np.argmax(metrics['best_valid_auc']) + 1
    best_valid_auc = max(metrics['best_valid_auc'])
    final_train_auc = metrics['train_auc'][-1]
    final_gap = (final_train_auc - metrics['best_valid_auc'][-1]) * 100

    total_time = sum(metrics['epoch_time'])
    avg_time = np.mean(metrics['epoch_time'])

    report = f"""
{'='*80}
GRAIL TRAINING REPORT
{'='*80}

EXPERIMENT SUMMARY:
  Total Epochs:           {len(metrics['epochs'])}
  Best Epoch:             {best_epoch}
  Best Validation AUC:    {best_valid_auc:.6f}
  Final Training AUC:     {final_train_auc:.6f}
  Final Generalization:   {final_gap:.2f}%

PERFORMANCE METRICS:
  Initial Train AUC:      {metrics['train_auc'][0]:.6f}
  Final Train AUC:        {metrics['train_auc'][-1]:.6f}
  AUC Improvement:        {(metrics['train_auc'][-1] - metrics['train_auc'][0]):.6f}

  Initial Valid AUC:      {metrics['best_valid_auc'][0]:.6f}
  Best Valid AUC:         {best_valid_auc:.6f}
  Valid Improvement:      {(best_valid_auc - metrics['best_valid_auc'][0]):.6f}

TRAINING EFFICIENCY:
  Total Training Time:    {total_time/60:.2f} minutes
  Average Time/Epoch:     {avg_time:.2f} seconds
  Min Time/Epoch:         {min(metrics['epoch_time']):.2f} seconds
  Max Time/Epoch:         {max(metrics['epoch_time']):.2f} seconds

CONVERGENCE ANALYSIS:
  Initial Loss:           {metrics['train_loss'][0]:.4f}
  Final Loss:             {metrics['train_loss'][-1]:.4f}
  Loss Reduction:         {((metrics['train_loss'][0] - metrics['train_loss'][-1]) / metrics['train_loss'][0] * 100):.2f}%

REGULARIZATION:
  Initial Weight Norm:    {metrics['weight_norm'][0]:.4f}
  Final Weight Norm:      {metrics['weight_norm'][-1]:.4f}
  Weight Norm Change:     {((metrics['weight_norm'][-1] - metrics['weight_norm'][0]) / metrics['weight_norm'][0] * 100):.2f}%

{'='*80}
"""

    with open(save_path, 'w') as f:
        f.write(report)

    print(f'✓ Saved training report: {save_path}')
    print(report)


def main(args):
    # Paths
    exp_dir = os.path.join('experiments', args.experiment_name)
    log_path = os.path.join(exp_dir, 'log_train.txt')

    if not os.path.exists(log_path):
        print(f'❌ Error: Log file not found at {log_path}')
        return

    print(f'📊 Analyzing training logs from: {args.experiment_name}')
    print(f'📁 Log file: {log_path}')
    print('='*80)

    # Parse logs
    metrics, validation_history = parse_log_file(log_path)

    if not metrics['epochs']:
        print('❌ Error: No training data found in log file')
        return

    print(f'✓ Parsed {len(metrics["epochs"])} epochs of training data')

    # Create output directory for plots
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Generate visualizations
    print('\n📈 Generating visualizations...')

    plot_training_overview(metrics, os.path.join(plots_dir, 'training_overview.png'))
    plot_convergence_analysis(metrics, os.path.join(plots_dir, 'convergence_analysis.png'))
    plot_learning_curve(metrics, os.path.join(plots_dir, 'learning_curve.png'))

    # Generate report
    print('\n📝 Generating training report...')
    generate_training_report(metrics, os.path.join(plots_dir, 'training_report.txt'))

    print('\n' + '='*80)
    print(f'✅ All visualizations saved to: {plots_dir}/')
    print('='*80)
    print('\nGenerated files:')
    print(f'  - training_overview.png     : Comprehensive 6-panel overview')
    print(f'  - convergence_analysis.png  : Loss and AUC convergence')
    print(f'  - learning_curve.png        : Publication-ready learning curve')
    print(f'  - training_report.txt       : Detailed text summary')
    print('\n💡 These plots are publication-quality (300 DPI) and ready for thesis!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize GraIL training results')
    parser.add_argument('-e', '--experiment_name', type=str, required=True,
                       help='Name of experiment folder in experiments/')

    args = parser.parse_args()
    main(args)