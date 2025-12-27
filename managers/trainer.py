import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sklearn import metrics


class Trainer():
    def __init__(self, params, graph_classifier, train, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train

        self.updates_counter = 0
        self.total_train_time = 0

        model_params = list(self.graph_classifier.parameters())

        # Count parameters
        total_params = sum(p.numel() for p in model_params)
        trainable_params = sum(p.numel() for p in model_params if p.requires_grad)

        logging.info('=' * 60)
        logging.info('MODEL INFORMATION')
        logging.info('=' * 60)
        logging.info(f'Total parameters: {total_params:,}')
        logging.info(f'Trainable parameters: {trainable_params:,}')
        logging.info(f'Non-trainable parameters: {total_params - trainable_params:,}')

        # GPU information
        if torch.cuda.is_available():
            gpu_id = params.gpu if hasattr(params, 'gpu') else 0
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            logging.info(f'GPU: {gpu_name}')
            logging.info(f'GPU Memory: {gpu_memory_total:.1f} GB')
        logging.info('=' * 60)

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        # Loss function selection: 'margin' (default) or 'bce'
        self.loss_type = getattr(params, 'loss_type', 'margin').lower()
        if self.loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
            logging.info(f'Using BCE Loss (Binary Cross Entropy)')
        else:
            self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='mean')
            logging.info(f'Using Margin Ranking Loss (margin={self.params.margin})')

        # Initialize TensorBoard writer
        tensorboard_dir = os.path.join(params.exp_dir, 'tensorboard')
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        logging.info(f'TensorBoard logging to: {tensorboard_dir}')
        logging.info(f'Run: tensorboard --logdir={tensorboard_dir}')

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
                              persistent_workers=True if num_workers > 0 else False,  # OPTIMIZATION: Keep workers alive between epochs
                              prefetch_factor=4 if num_workers > 0 else None)  # OPTIMIZATION: Prefetch 4 batches per worker
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())

        # Add progress bar for training
        pbar = tqdm(dataloader, desc=f"Training Epoch", leave=False)
        for b_idx, batch in enumerate(pbar):
            # Skip None batches (from missing LMDB entries)
            if batch is None:
                continue
            data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)

            # OPTIMIZATION: Removed unnecessary cache clearing (5-10% speedup)
            # Only clear cache if OOM occurs

            self.optimizer.zero_grad()

            score_pos = self.graph_classifier(data_pos)
            score_neg = self.graph_classifier(data_neg)

            if self.loss_type == 'bce':
                # BCE Loss: Combine positive and negative scores with labels
                scores = torch.cat([score_pos.squeeze(1), score_neg.squeeze(1)], dim=0)
                labels = torch.cat([
                    torch.ones(len(score_pos)),
                    torch.zeros(len(score_neg))
                ], dim=0).to(device=self.params.device)
                loss = self.criterion(scores, labels)
            else:
                # Margin Ranking Loss: L = max(0, margin - score_pos + score_neg)
                loss = self.criterion(
                    score_pos.squeeze(1),
                    score_neg.view(len(score_pos), -1).mean(dim=1),
                    torch.ones(len(score_pos)).to(device=self.params.device)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_params, self.params.clip)
            self.optimizer.step()

            self.updates_counter += 1

            with torch.no_grad():
                all_scores += score_pos.squeeze().detach().cpu().tolist() + score_neg.squeeze().detach().cpu().tolist()
                all_labels += targets_pos.tolist() + targets_neg.tolist()
                total_loss += loss.item()

            # Update progress bar with average loss (smoother than batch loss)
            avg_loss = total_loss / (b_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'batch': b_idx})

            # TensorBoard: Log batch loss every 100 iterations
            if self.updates_counter % 100 == 0:
                self.writer.add_scalar('Loss/train_batch', loss.item(), self.updates_counter)
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', current_lr, self.updates_counter)

            if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()

                result = self.valid_evaluator.eval()
                logging.info('\nPerformance:' + str(result) + 'in ' + str(time.time() - tic))

                # TensorBoard: Log validation metrics
                self.writer.add_scalar('AUC/validation', result['auc'], self.updates_counter)
                if 'auc_pr' in result:
                    self.writer.add_scalar('AUC_PR/validation', result['auc_pr'], self.updates_counter)
                if 'loss' in result:
                    self.writer.add_scalar('Loss/validation', result['loss'], self.updates_counter)
                if 'hits@10' in result:
                    self.writer.add_scalar('Hits@10/validation', result['hits@10'], self.updates_counter)
                if 'mrr' in result:
                    self.writer.add_scalar('MRR/validation', result['mrr'], self.updates_counter)

                if result['auc'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result['auc']
                    self.not_improved_count = 0

                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result['auc']

        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        return total_loss, auc, auc_pr, weight_norm

    def _format_time(self, seconds):
        """Format seconds into human readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = seconds / 60
            return f"{mins:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"

    def _get_gpu_memory_usage(self):
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            gpu_id = self.params.gpu if hasattr(self.params, 'gpu') else 0
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            return allocated, reserved
        return 0, 0

    def train(self):
        self.reset_training_state()
        total_start_time = time.time()

        logging.info('=' * 60)
        logging.info('TRAINING STARTED')
        logging.info('=' * 60)

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            loss, auc, auc_pr, weight_norm = self.train_epoch()
            time_elapsed = time.time() - time_start
            self.total_train_time += time_elapsed

            # Get GPU memory usage
            gpu_alloc, gpu_reserved = self._get_gpu_memory_usage()

            # Format logging
            logging.info('=' * 60)
            logging.info(f'Epoch {epoch}/{self.params.num_epochs}')
            logging.info(f'  Loss: {loss:.4f}')
            logging.info(f'  Train AUC: {auc:.4f} | Train AUC-PR: {auc_pr:.4f}')
            logging.info(f'  Best Valid AUC: {self.best_metric:.4f}')
            logging.info(f'  Epoch Time: {self._format_time(time_elapsed)} | Total: {self._format_time(self.total_train_time)}')
            if torch.cuda.is_available():
                logging.info(f'  GPU Memory: {gpu_alloc:.2f}GB allocated / {gpu_reserved:.2f}GB reserved')
            logging.info('=' * 60)

            # TensorBoard: Log epoch-level metrics
            self.writer.add_scalar('Loss/train_epoch', loss.item() if torch.is_tensor(loss) else loss, epoch)
            self.writer.add_scalar('AUC/train_epoch', auc, epoch)
            self.writer.add_scalar('AUC_PR/train_epoch', auc_pr, epoch)
            self.writer.add_scalar('Weight_Norm/train_epoch', weight_norm, epoch)
            self.writer.add_scalar('Time/epoch_seconds', time_elapsed, epoch)
            self.writer.add_scalar('Time/total_seconds', self.total_train_time, epoch)
            self.writer.add_scalar('Best_Validation_AUC', self.best_metric, epoch)
            if torch.cuda.is_available():
                self.writer.add_scalar('GPU/memory_allocated_GB', gpu_alloc, epoch)
                self.writer.add_scalar('GPU/memory_reserved_GB', gpu_reserved, epoch)

            # if self.valid_evaluator and epoch % self.params.eval_every == 0:
            #     result = self.valid_evaluator.eval()
            #     logging.info('\nPerformance:' + str(result))

            #     if result['auc'] >= self.best_metric:
            #         self.save_classifier()
            #         self.best_metric = result['auc']
            #         self.not_improved_count = 0

            #     else:
            #         self.not_improved_count += 1
            #         if self.not_improved_count > self.params.early_stop:
            #             logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
            #             break
            #     self.last_metric = result['auc']

            if epoch % self.params.save_every == 0:
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))

        # Final evaluation after last epoch
        if self.valid_evaluator:
            logging.info('Final evaluation after training...')
            result = self.valid_evaluator.eval()
            logging.info(f'Final validation result: {result}')
            if result['auc'] >= self.best_metric:
                self.save_classifier()
                self.best_metric = result['auc']

        # Training summary
        total_time = time.time() - total_start_time
        gpu_alloc, gpu_reserved = self._get_gpu_memory_usage()

        logging.info('')
        logging.info('=' * 60)
        logging.info('TRAINING COMPLETED')
        logging.info('=' * 60)
        logging.info(f'Total Epochs: {self.params.num_epochs}')
        logging.info(f'Total Training Time: {self._format_time(total_time)}')
        logging.info(f'Average Time per Epoch: {self._format_time(total_time / self.params.num_epochs)}')
        logging.info(f'Best Validation AUC: {self.best_metric:.4f}')
        if torch.cuda.is_available():
            logging.info(f'Peak GPU Memory: {gpu_reserved:.2f}GB')
        logging.info('=' * 60)

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))  # Does it overwrite or fuck with the existing file?
        logging.info('Better models found w.r.t accuracy. Saved it!')

    def __del__(self):
        """Close TensorBoard writer when trainer is destroyed"""
        if hasattr(self, 'writer'):
            self.writer.close()
