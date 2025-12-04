"""
Stable Mixed Precision Trainer with NaN protection
"""
import os
import logging
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from sklearn import metrics


class StableAMPTrainer():
    """
    Trainer with Stable Mixed Precision Training

    Features:
    - Conservative gradient scaling
    - Gradient clipping before scaling
    - NaN detection and recovery
    - FP32 loss computation
    - Fallback to FP32 on persistent NaN
    """

    def __init__(self, params, graph_classifier, train, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr,
                                      momentum=params.momentum,
                                      weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr,
                                       weight_decay=self.params.l2)

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='mean')

        # Stable AMP setup
        self.use_amp = getattr(params, 'use_amp', True) and torch.cuda.is_available()

        if self.use_amp:
            # Conservative settings to avoid NaN
            self.scaler = GradScaler(
                init_scale=2.**10,      # Start small (default 2^16)
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000    # Grow slowly
            )

            # NaN tracking
            self.nan_count = 0
            self.max_nan_tolerance = 10

            logging.info('🚀 Stable Mixed Precision (AMP) ENABLED')
            logging.info('   - Conservative gradient scaling (init_scale=2^10)')
            logging.info('   - Gradient clipping (max_norm=1.0)')
            logging.info('   - NaN detection and recovery')
            logging.info('   - FP32 loss computation')
        else:
            self.scaler = None
            logging.info('Mixed Precision disabled')

        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []

        # Use the number of workers specified in the parameters
        num_workers = self.params.num_workers
        dataloader = DataLoader(self.train_data,
                              batch_size=self.params.batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=self.params.collate_fn,
                              pin_memory=torch.cuda.is_available(),
                              persistent_workers=True if num_workers > 0 else False,
                              prefetch_factor=4 if num_workers > 0 else None)
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())

        # Add progress bar for training
        pbar = tqdm(dataloader, desc=f"Training Epoch", leave=False)

        for b_idx, batch in enumerate(pbar):
            data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(
                batch, self.params.device
            )

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                score_pos = self.graph_classifier(data_pos)
                score_neg = self.graph_classifier(data_neg)

            # Loss computation in FP32 for stability
            # Cast scores to FP32 before loss
            loss = self.criterion(
                score_pos.squeeze(1).float(),
                score_neg.view(len(score_pos), -1).mean(dim=1).float(),
                torch.ones(len(score_pos)).to(device=self.params.device)
            )

            # Backward pass with gradient scaling
            if self.use_amp:
                # Scale loss and backward
                self.scaler.scale(loss).backward()

                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)

                # Check for NaN/Inf in gradients
                has_nan = self._check_nan_gradients()

                if has_nan:
                    self.nan_count += 1
                    logging.warning(
                        f"⚠️  NaN/Inf detected in gradients at iteration {b_idx} "
                        f"(count: {self.nan_count}/{self.max_nan_tolerance})"
                    )

                    # Skip this step
                    self.optimizer.zero_grad()
                    self.scaler.update()  # This will reduce scale

                    # If too many NaNs, disable AMP
                    if self.nan_count >= self.max_nan_tolerance:
                        logging.error(
                            f"❌ Too many NaN occurrences ({self.nan_count}). "
                            f"DISABLING AMP for stability."
                        )
                        self.use_amp = False
                        self.scaler._enabled = False

                    continue  # Skip this batch

                # Gradient clipping (after unscale, before step)
                torch.nn.utils.clip_grad_norm_(
                    self.graph_classifier.parameters(),
                    max_norm=1.0
                )

                # Optimizer step with scaled gradients
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Standard FP32 training
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.graph_classifier.parameters(),
                    max_norm=1.0
                )
                self.optimizer.step()

            self.updates_counter += 1

            # Track metrics
            with torch.no_grad():
                all_scores += score_pos.squeeze().detach().cpu().tolist() + \
                             score_neg.squeeze().detach().cpu().tolist()
                all_labels += targets_pos.tolist() + targets_neg.tolist()
                total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'batch': b_idx,
                'amp': 'ON' if self.use_amp else 'OFF',
                'scale': f'{self.scaler.get_scale():.0f}' if self.use_amp else 'N/A'
            })

            # Validation during training
            if self.valid_evaluator and self.params.eval_every_iter and \
               self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()

                result = self.valid_evaluator.eval()
                logging.info('\nPerformance:' + str(result) + ' in ' + str(time.time() - tic))

                if result['auc'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result['auc']
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(
                            f"Validation performance didn't improve for "
                            f"{self.params.early_stop} epochs. Training stops."
                        )
                        break
                self.last_metric = result['auc']

        # Epoch metrics
        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)
        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        # Report AMP status
        if self.use_amp:
            logging.info(
                f'AMP Status: scale={self.scaler.get_scale():.0f}, '
                f'nan_count={self.nan_count}'
            )

        return total_loss, auc, auc_pr, weight_norm

    def _check_nan_gradients(self):
        """Check if any gradient contains NaN or Inf"""
        for param in self.graph_classifier.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return True
        return False

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            loss, auc, auc_pr, weight_norm = self.train_epoch()
            time_elapsed = time.time() - time_start

            logging.info(
                f'Epoch {epoch} with loss: {loss}, training auc: {auc}, '
                f'training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, '
                f'weight_norm: {weight_norm} in {time_elapsed:.2f}s'
            )

            if epoch % self.params.save_every == 0:
                torch.save(
                    self.graph_classifier,
                    os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth')
                )

    def save_classifier(self):
        torch.save(
            self.graph_classifier,
            os.path.join(self.params.exp_dir, 'best_graph_classifier.pth')
        )
        logging.info('Better models found w.r.t accuracy. Saved it!')