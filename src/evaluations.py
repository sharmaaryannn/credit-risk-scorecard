# type: ignore
from typing import Dict, Tuple
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def compute_metrics(y_true, y_probs) -> Dict[str, float]:
    # --- 1. ROC-AUC Score ---
    auc_roc = float(roc_auc_score(y_true, y_probs))

    # --- 2. PR-AUC (Precision-Recall Area Under Curve) ---
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    auc_pr = float(auc(recall, precision))

    # --- 3. Kolmogorov-Smirnov (KS) Statistic ---
    goods = y_probs[y_true == 0]
    bads = y_probs[y_true == 1]
    ks_res = ks_2samp(goods, bads)
    ks_stat = float(getattr(ks_res, "statistic", ks_res[0]))

    return {
        "ROC-AUC": round(auc_roc, 4),
        "PR-AUC": round(auc_pr, 4),
        "KS-Statistic": round(ks_stat, 4),
    }


def find_cost_optimal_threshold(
    y_true, y_probs, margin: float, lgd: float
) -> Tuple[float, float]:
    # --- 1. Generate Candidate Risk Thresholds ---
    thresholds = np.linspace(0.01, 0.50, 500)
    net_profits = []

    # --- 2. Evaluate Financial Profit Across Thresholds ---
    for t in thresholds:
        approved = y_probs < t
        is_bad_approved = np.logical_and(y_true == 1, approved)
        is_good_approved = np.logical_and(y_true == 0, approved)

        fp_loss = float(np.sum(is_bad_approved)) * lgd
        tp_gain = float(np.sum(is_good_approved)) * margin
        net_profits.append(tp_gain - fp_loss)

    # --- 3. Identify Maximum Profit Threshold ---
    best_idx = int(np.argmax(net_profits))
    optimal_t = float(thresholds[best_idx])

    # --- 4. Compare with Naive 0.5 Cutoff ---
    app_05 = y_probs < 0.5
    bad_05 = np.logical_and(y_true == 1, app_05)
    good_05 = np.logical_and(y_true == 0, app_05)

    profit_05 = float(np.sum(good_05) * margin - np.sum(bad_05) * lgd)
    opt_profit = net_profits[best_idx]

    denom = abs(profit_05) if profit_05 != 0 else 1.0
    pct_improvement = float(((opt_profit - profit_05) / denom) * 100)

    return optimal_t, pct_improvement