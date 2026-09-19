# type: ignore
import os
import re
import warnings
import yaml

# Suppress non-critical library deprecation warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# Local module imports
from src.evaluations import compute_metrics, find_cost_optimal_threshold
from src.explainability import audit_demographic_fairness, compute_shap_importance
from src.feature_engineering import load_and_aggregate_data
from src.scorecard import WoEScorecard


def main():
    print("=" * 60)
    print("      HOME CREDIT DEFAULT RISK: END-TO-END PIPELINE      ")
    print("=" * 60)

    # --- 1. Load Configuration ---
    print("\n[1/6] Loading Configuration...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config["paths"]["raw_data_dir"]
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    margin = config["financials"]["net_interest_margin"]
    lgd = config["financials"]["loss_given_default"]
    lgb_params = config["model_params"]["lgbm"]

    # --- 2. Feature Engineering & Aggregations ---
    print("\n[2/6] Running Relational Feature Engineering...")
    df = load_and_aggregate_data(raw_dir)

    target_col = "TARGET"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in aggregated dataset.")

    # Drop non-feature identifiers
    X = df.drop(columns=[target_col, "SK_ID_CURR"], errors="ignore")
    y = df[target_col].values

    # Clean feature names strictly for LightGBM JSON compatibility
    X.columns = [re.sub(r"[^\w]", "_", str(col)) for col in X.columns]

    # --- 3. Stratified Train-Test Split ---
    print("\n[3/6] Performing Stratified Train/Test Split (80/20)...")
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.20, random_state=42, stratify=y
    )

    # --- 4. Model Training & Baseline Evaluation ---
    print("\n[4/6] Training Baseline WoE Scorecard...")
    scorecard = WoEScorecard(n_bins=10)
    scorecard.fit(X_train, y_train)
    scorecard_probs = scorecard.predict_proba(X_test)[:, 1]
    scorecard_metrics = compute_metrics(y_test, scorecard_probs)

    print("\nTraining LightGBM Gradient Boosting Model...")
    lgbm = LGBMClassifier(**lgb_params)
    lgbm.fit(X_train, y_train)
    lgbm_probs = lgbm.predict_proba(X_test)[:, 1]
    lgbm_metrics = compute_metrics(y_test, lgbm_probs)

    print("\n---------------- MODEL COMPARISON ----------------")
    print(f"WoE Scorecard Baseline  : {scorecard_metrics}")
    print(f"LightGBM Model          : {lgbm_metrics}")

    # --- 5. Financial Cost-Threshold Optimization ---
    print("\n[5/6] Optimizing Decision Threshold for Portfolio Profit...")
    opt_threshold, profit_improvement = find_cost_optimal_threshold(
        y_test, lgbm_probs, margin, lgd
    )
    print(f"Optimal Default Probability Threshold : {opt_threshold:.4f}")
    print(f"Financial Gain over Naive 0.5 Cutoff   : +{profit_improvement:.2f}%")

    # --- 6. Explainability & Fairness Audits ---
    print("\n[6/6] Generating SHAP Importance & Demographic Fairness Audit...")
    shap_df = compute_shap_importance(lgbm, X_test, max_samples=3000)
    shap_df.to_csv(os.path.join(output_dir, "shap_feature_importance.csv"), index=False)

    # Demographic Fairness Audit on Gender
    gender_col = [c for c in df_test.columns if "CODE_GENDER" in c]
    if gender_col:
        gender_audit = audit_demographic_fairness(
            df_test, gender_col[0], y_test, lgbm_probs, opt_threshold
        )
        gender_audit.to_csv(os.path.join(output_dir, "fairness_audit_gender.csv"), index=False)
        print("\nDemographic Fairness Audit Summary (Gender):")
        print(gender_audit.to_string(index=False))

    # Save summary report
    summary_report = {
        "Metric": list(lgbm_metrics.keys()) + ["Optimal Threshold", "Profit Improvement (%)"],
        "Value": list(lgbm_metrics.values()) + [round(opt_threshold, 4), round(profit_improvement, 2)],
    }
    pd.DataFrame(summary_report).to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("   PIPELINE EXECUTION COMPLETE! Artifacts saved to 'outputs/'   ")
    print("=" * 60)


if __name__ == "__main__":
    main()