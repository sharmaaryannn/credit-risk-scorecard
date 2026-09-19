# type: ignore
from typing import Any, Dict
import numpy as np
import pandas as pd
import shap


def compute_shap_importance(
    model: Any, X: pd.DataFrame, max_samples: int = 5000
) -> pd.DataFrame:
    # --- 1. Downsample for Computational Efficiency ---
    sample_df = X.sample(n=min(len(X), max_samples), random_state=42)

    # --- 2. Calculate TreeExplainer SHAP Values ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_df)

    # --- 3. Extract Mean Absolute SHAP Importance ---
    if isinstance(shap_values, list):
        shap_vals = np.abs(shap_values[1]).mean(axis=0)
    else:
        shap_vals = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame({"feature": sample_df.columns, "importance": shap_vals})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df


def audit_demographic_fairness(
    df: pd.DataFrame,
    sensitive_col: str,
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    # --- 1. Construct Audit DataFrame ---
    audit_df = df[[sensitive_col]].copy()
    audit_df["y_true"] = y_true
    audit_df["approved"] = (y_probs < threshold).astype(int)  # 1 = Approved

    results = []

    # --- 2. Evaluate Error Disparities Across Subgroups ---
    for group_name, group_data in audit_df.groupby(sensitive_col):
        total = len(group_data)
        approval_rate = group_data["approved"].mean()

        # False Positive Rate (FPR): Defaulted borrowers incorrectly approved
        defaults = group_data[group_data["y_true"] == 1]
        fpr = (defaults["approved"] == 1).mean() if len(defaults) > 0 else 0.0

        # False Negative Rate (FNR): Good borrowers incorrectly denied
        repayers = group_data[group_data["y_true"] == 0]
        fnr = (repayers["approved"] == 0).mean() if len(repayers) > 0 else 0.0

        results.append(
            {
                "Subgroup": group_name,
                "Sample Size": total,
                "Approval Rate": round(float(approval_rate), 4),
                "False Positive Rate (Bad Approved)": round(float(fpr), 4),
                "False Negative Rate (Good Denied)": round(float(fnr), 4),
            }
        )

    return pd.DataFrame(results)