"""Evaluation metrics: ranking quality (PR-AUC, ROC-AUC, Gini, KS) and naira profit.

Interview-critical (rewrite from memory once): expected_profit_per_application.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from credit_risk.config import CostsConfig


def expected_profit_per_application(
    y_true: np.ndarray, p_default: np.ndarray, threshold: float, costs: CostsConfig
) -> float:
    """Average profit per application (naira) if we decline when P(default) > threshold.

    Per application:
      approved, repays   -> +c_fp  (the fee earned; it is what declining would forgo)
      approved, defaults -> -c_fn  (principal lost after recovery)
      declined           ->  0
    Computed in float64 so large naira sums do not lose precision.
    """
    y = np.asarray(y_true, dtype=np.float64)
    approve = np.asarray(p_default, dtype=np.float64) <= threshold
    profit = np.where(approve, np.where(y == 1, -costs.c_fn, costs.c_fp), 0.0)
    return float(profit.mean())


def approve_all_profit(y_true: np.ndarray, costs: CostsConfig) -> float:
    """The floor every model must beat: approve everyone (threshold = 1)."""
    y = np.asarray(y_true, dtype=np.float64)
    return float(np.where(y == 1, -costs.c_fn, costs.c_fp).mean())


def ks_statistic(y_true: np.ndarray, p_default: np.ndarray) -> float:
    """Kolmogorov-Smirnov: largest gap between the score distributions of defaulters and
    non-defaulters. A standard credit-scoring metric (equals max(TPR - FPR))."""
    fpr, tpr, _ = roc_curve(y_true, p_default)
    return float(np.max(tpr - fpr))


def evaluate_predictions(
    y_true: np.ndarray, p_default: np.ndarray, threshold: float, costs: CostsConfig
) -> dict[str, float]:
    """All Stage 4 metrics for one set of predictions."""
    y = np.asarray(y_true)
    p = np.asarray(p_default, dtype=np.float64)
    declined = p > threshold
    roc_auc = roc_auc_score(y, p)
    profit = expected_profit_per_application(y, p, threshold, costs)
    floor = approve_all_profit(y, costs)
    return {
        "pr_auc": float(average_precision_score(y, p)),  # random model ~ default rate (0.081)
        "roc_auc": float(roc_auc),
        "gini": float(2 * roc_auc - 1),
        "ks": ks_statistic(y, p),
        "expected_profit": profit,
        "approve_all_profit": floor,
        "profit_uplift": profit - floor,
        "approval_rate": float(1 - declined.mean()),
        # Of the real defaulters, the share we would decline (recall at the threshold).
        "default_capture_rate": float(declined[y == 1].mean()),
    }
