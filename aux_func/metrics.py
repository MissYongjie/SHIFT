# evaluation/metrics.py

from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
import numpy as np

def compute_metrics(ground_truth, prediction):
    gt_flat = ground_truth.flatten()
    pred_flat = prediction.flatten()

    conf_mat = confusion_matrix(gt_flat > 0, pred_flat > 0)
    oa = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
    f1 = f1_score(gt_flat > 0, pred_flat > 0)
    kappa = cohen_kappa_score(gt_flat > 0, pred_flat > 0)

    return oa, f1, kappa, conf_mat
