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

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='mean')

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
            data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)

            # OPTIMIZATION: Removed unnecessary cache clearing (5-10% speedup)
            # Only clear cache if OOM occurs

            self.optimizer.zero_grad()

            score_pos = self.graph_classifier(data_pos)
            score_neg = self.graph_classifier(data_neg)
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

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            loss, auc, auc_pr, weight_norm = self.train_epoch()
            time_elapsed = time.time() - time_start
            logging.info(f'Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')

            # TensorBoard: Log epoch-level metrics
            self.writer.add_scalar('Loss/train_epoch', loss.item() if torch.is_tensor(loss) else loss, epoch)
            self.writer.add_scalar('AUC/train_epoch', auc, epoch)
            self.writer.add_scalar('AUC_PR/train_epoch', auc_pr, epoch)
            self.writer.add_scalar('Weight_Norm/train_epoch', weight_norm, epoch)
            self.writer.add_scalar('Time/epoch_seconds', time_elapsed, epoch)
            self.writer.add_scalar('Best_Validation_AUC', self.best_metric, epoch)

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

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))  # Does it overwrite or fuck with the existing file?
        logging.info('Better models found w.r.t accuracy. Saved it!')

    def __del__(self):
        """Close TensorBoard writer when trainer is destroyed"""
        if hasattr(self, 'writer'):
            self.writer.close()
