import os
import pandas as pd


def load_and_aggregate_data(data_dir: str) -> pd.DataFrame:
    # --- STEP 1: Load Primary Application Table ---
    print("Loading primary application dataset...")
    app_train = pd.read_csv(os.path.join(data_dir, "application_train.csv"))

    # --- STEP 2: Aggregate Bureau Data (External Credit Bureau History) ---
    print("Aggregating Bureau table...")
    bureau = pd.read_csv(os.path.join(data_dir, "bureau.csv"))
    
    # Compress multiple bureau records per applicant into summary statistics
    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        {
            "DAYS_CREDIT": ["min", "max", "mean"],      # How long ago loans were opened
            "CREDIT_DAY_OVERDUE": ["max", "mean"],     # Max/average overdue days
            "AMT_CREDIT_SUM": ["sum", "mean", "max"],   # Total credit limits
            "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],     # Total current debt
            "CNT_CREDIT_PROLONG": ["sum"],             # How many times credit was extended
        }
    )
    # Flatten multi-level column names (e.g., bureau_agg['DAYS_CREDIT']['min'] -> 'BUREAU_DAYS_CREDIT_MIN')
    bureau_agg.columns = [
        f"BUREAU_{col[0]}_{col[1].upper()}" for col in bureau_agg.columns
    ]

    # --- STEP 3: Aggregate Previous Applications (Past Home Credit Loans) ---
    print("Aggregating Previous Applications table...")
    prev = pd.read_csv(os.path.join(data_dir, "previous_application.csv"))
    
    # Create binary indicator for approved loans
    prev["APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    
    prev_agg = prev.groupby("SK_ID_CURR").agg(
        {
            "SK_ID_PREV": ["count"],                   # Number of prior applications
            "APPROVED": ["mean", "sum"],                # Past approval rate and total approvals
            "AMT_APPLICATION": ["mean", "max"],        # Amounts requested
            "AMT_DOWN_PAYMENT": ["mean", "max"],       # Down payments made
        }
    )
    prev_agg.columns = [
        f"PREV_{col[0]}_{col[1].upper()}" for col in prev_agg.columns
    ]

    # --- STEP 4: Aggregate Installment Payments (Payment Timeliness & Delays) ---
    print("Aggregating Installment Payments table...")
    inst = pd.read_csv(os.path.join(data_dir, "installments_payments.csv"))
    
    # Calculate Days Past Due (DPD): Positive value means payment was late
    inst["DPD"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    
    # Calculate Payment Percentage: Ratio of what was actually paid vs billed amount
    inst["PAYMENT_PERCENTAGE"] = inst["AMT_PAYMENT"] / (inst["AMT_INSTALMENT"] + 1)
    
    inst_agg = inst.groupby("SK_ID_CURR").agg(
        {
            "DPD": ["max", "mean"],                     # Worst and average payment delays
            "PAYMENT_PERCENTAGE": ["mean", "min"],      # Underpayment metrics
            "AMT_PAYMENT": ["sum", "mean"],             # Total cash paid back
        }
    )
    inst_agg.columns = [
        f"INSTAL_{col[0]}_{col[1].upper()}" for col in inst_agg.columns
    ]

    # --- STEP 5: Merge Relational Tables Back to Main Application ---
    print("Merging feature sets...")
    df = (
        app_train.set_index("SK_ID_CURR")
        .join(bureau_agg, how="left")
        .join(prev_agg, how="left")
        .join(inst_agg, how="left")
        .reset_index()
    )

    # --- STEP 6: Categorical Encoding ---
    # Convert text features (like contract type, gender) into numeric dummy variables
    # --- STEP 6: Categorical Encoding ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df