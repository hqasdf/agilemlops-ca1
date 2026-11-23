"""Shared chronological threshold-calibration helpers for team comparisons."""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def recent_training_window(train_dates, share=0.20):
    """Select the latest whole dates in training for threshold calibration."""
    if not 0 < share < 1 or len(train_dates) < 2 or train_dates.isna().any():
        raise ValueError("Threshold calibration needs valid dates and a share between 0 and 1")
    dates = pd.to_datetime(train_dates, errors="raise")
    start = dates.sort_values().iloc[int(len(dates) * (1 - share))]
    recent = dates >= start
    if recent.all() or not recent.any():
        raise ValueError("Training needs at least two different order dates")
    return recent


def historical_calibration_mask(train_dates, current_batch_indices, share=0.20):
    """Reserve historical calibration rows while retaining current rows for fit.

    The calibration rows predate some fitted rows; disclose that difference.
    """
    if not train_dates.index.is_unique:
        raise ValueError('Training indices must be unique')
    current=train_dates.index.isin(current_batch_indices)
    if not current.any() or current.all():
        raise ValueError('Both historical and current-batch training rows are required')
    historical=train_dates.loc[~current]
    recent=recent_training_window(historical,share)
    mask=pd.Series(False,index=train_dates.index)
    mask.loc[recent.index]=recent
    return mask


def best_f1_threshold(y_true, probabilities):
    """Choose the threshold with maximum fraud F1; break ties toward higher cutoffs."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or probabilities.ndim != 1 or len(y_true) != len(probabilities) or not len(y_true):
        raise ValueError("Threshold labels and probabilities must have matching nonempty arrays")
    if not np.isin(y_true, [0, 1]).all() or set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Threshold calibration requires both binary classes")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Threshold probabilities must be finite values between 0 and 1")
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        raise ValueError("Threshold calibration produced no candidate cutoffs")
    total = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], total,
                   out=np.zeros_like(total), where=total != 0)
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best])


