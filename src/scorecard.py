# type: ignore
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class WoEScorecard:
    """
    Weight of Evidence (WoE) and Information Value (IV) Scorecard baseline.
    Uses quantile binning to transform features and fits an L1-regularized
    Logistic Regression model.
    """

    def __init__(self, n_bins: int = 10, min_iv: float = 0.02, max_iv: float = 0.50):
        self.n_bins = n_bins
        self.min_iv = min_iv
        self.max_iv = max_iv
        self.woe_maps = {}
        self.iv_scores = {}
        self.selected_cols = []
        self.model = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)

    def _calc_woe_iv(self, feature: pd.Series, target: pd.Series):
        try:
            binned = pd.qcut(feature, q=self.n_bins, duplicates="drop")
        except Exception:
            return None, 0.0

        df = pd.DataFrame({"bin": binned, "target": target})
        grouped = df.groupby("bin", observed=False)["target"].agg(
            goods=lambda x: (x == 0).sum(),
            bads=lambda x: (x == 1).sum()
        )

        total_goods = (target == 0).sum()
        total_bads = (target == 1).sum()

        eps = 1e-5
        grouped["dist_goods"] = (grouped["goods"] + eps) / total_goods
        grouped["dist_bads"] = (grouped["bads"] + eps) / total_bads

        grouped["woe"] = np.log(grouped["dist_goods"] / grouped["dist_bads"])
        grouped["iv"] = (grouped["dist_goods"] - grouped["dist_bads"]) * grouped["woe"]

        iv_total = float(grouped["iv"].sum())
        woe_map = grouped["woe"].to_dict()

        return woe_map, iv_total

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        y_series = pd.Series(y, index=X.index)
        self.selected_cols = []

        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.number):
                filled_feature = X[col].fillna(X[col].median())
                woe_map, iv = self._calc_woe_iv(filled_feature, y_series)
                if woe_map is not None and self.min_iv <= iv <= self.max_iv:
                    self.woe_maps[col] = woe_map
                    self.iv_scores[col] = iv
                    self.selected_cols.append(col)

        X_woe = self._transform_df(X[self.selected_cols])
        self.model.fit(X_woe, y)
        return self

    def _transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        X_woe = pd.DataFrame(index=X.index)
        for col in self.selected_cols:
            filled_feature = X[col].fillna(X[col].median())
            binned = pd.qcut(filled_feature, q=self.n_bins, duplicates="drop")
            X_woe[col] = binned.map(self.woe_maps[col]).astype(float).fillna(0)
        return X_woe

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_woe = self._transform_df(X[self.selected_cols])
        return self.model.predict_proba(X_woe)  