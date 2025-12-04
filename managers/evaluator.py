import os
import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        # Use num_workers from params for faster data loading (default: 0 for compatibility)
        # IMPORTANT: Evaluation often fails with num_workers > 0 due to multiprocessing + LMDB issues
        # SOLUTION: Force single-threaded evaluation to avoid "received 0 items of ancdata" error
        num_workers = getattr(self.params, 'num_workers', 0)
        # Force eval_workers = 0 to avoid multiprocessing errors
        # Training can still use num_workers > 0, but eval must be single-threaded with LMDB
        eval_workers = 0  # FORCE 0 to prevent RuntimeError with multiprocessing + LMDB

        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False,
                               num_workers=eval_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            # Add progress bar for evaluation
            pbar = tqdm(dataloader, desc="Evaluating", total=len(dataloader))
            for b_idx, batch in enumerate(pbar):

                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)

                score_pos = self.graph_classifier(data_pos)
                score_neg = self.graph_classifier(data_neg)

                # preds += torch.argmax(logits.detach().cpu(), dim=1).tolist()
                pos_scores += score_pos.squeeze(1).detach().cpu().tolist()
                neg_scores += score_neg.squeeze(1).detach().cpu().tolist()
                pos_labels += targets_pos.tolist()
                neg_labels += targets_neg.tolist()

                # Update progress bar
                pbar.set_postfix({'batch': f'{b_idx+1}/{len(dataloader)}'})

            pbar.close()

        # acc = metrics.accuracy_score(labels, preds)
        auc = metrics.roc_auc_score(pos_labels + neg_labels, pos_scores + neg_scores)
        auc_pr = metrics.average_precision_score(pos_labels + neg_labels, pos_scores + neg_scores)

        if save:
            pos_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_{}_predictions.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/neg_{}_0.txt'.format(self.params.dataset, self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset, self.data.file_name, self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        return {'auc': auc, 'auc_pr': auc_pr}
