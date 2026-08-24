"""Cost metric and operating-point selection."""
import numpy as np
from sklearn.metrics import confusion_matrix

from .config import COST


def total_cost(y_true, y_pred) -> int:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    return int((cm * COST).sum())


def expected_cost_decision(proba: np.ndarray) -> np.ndarray:
    """Choose the class minimising expected cost, not the most likely class."""
    return (proba @ COST).argmin(axis=1)


def threshold_decision(score: np.ndarray, threshold: float) -> np.ndarray:
    """Binary inspect / don't-inspect rule at a fixed risk threshold."""
    return np.where(score >= threshold, 4, 0)


def budget_curve(score, y_true, budgets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    """Recall and cost as a function of inspection capacity."""
    import pandas as pd
    n4 = max((np.asarray(y_true) == 4).sum(), 1)
    rows = []
    for b in budgets:
        k = int(b * len(score))
        thr = np.sort(score)[-k]
        pred = threshold_decision(score, thr)
        cm = confusion_matrix(y_true, pred, labels=[0, 1, 2, 3, 4])
        rows.append({"alert_rate": b, "inspected": k, "threshold": float(thr),
                     "recall_class4": cm[4, 4] / n4, "missed_class4": int(cm[4, 0]),
                     "false_alarms": int(cm[0, 4]), "total_cost": total_cost(y_true, pred)})
    return pd.DataFrame(rows)