"""
Metrics and threshold optimization suite for Lunar Surface Classification.
"""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'recall_0': float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'recall_1': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    }

def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> tuple:
    """
    Sweeps decision thresholds from 0.10 to 0.90 to find the operating point
    that strictly maximizes Balanced Accuracy.
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    best_bal_acc = -1.0
    best_th = 0.50
    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        bal = balanced_accuracy_score(y_true, preds)
        if bal > best_bal_acc:
            best_bal_acc = bal
            best_th = float(th)
    return best_th, best_bal_acc
