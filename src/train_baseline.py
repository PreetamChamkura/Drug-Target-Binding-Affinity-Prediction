"""Train baseline ML models (Random Forest, XGBoost) to predict DAVIS
binding affinity from the molecular/protein descriptors extracted in
src/features.py.

Usage:
    python -m src.train_baseline
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42


def concordance_index(y_true, y_pred):
    """Fraction of comparable pairs correctly ranked (standard DTA metric)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    order = np.argsort(y_true)
    y_true, y_pred = y_true[order], y_pred[order]

    n = len(y_true)
    concordant = 0.0
    permissible = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[j] == y_true[i]:
                continue
            permissible += 1
            if y_pred[j] > y_pred[i]:
                concordant += 1
            elif y_pred[j] == y_pred[i]:
                concordant += 0.5
    return concordant / permissible if permissible > 0 else float("nan")


def load_dataset(path="data/davis_features.csv"):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c.startswith(("drug_", "target_"))]
    X = df[feature_cols]
    # DAVIS affinity (Kd, nM) is heavily right-skewed; model on pKd as is
    # standard practice for this dataset.
    y = -np.log10(df["Y"].values * 1e-9)
    return X, y, feature_cols


def evaluate(name, y_test, y_pred, ci_sample_size=2000, rng=None):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # CI is O(n^2); subsample for speed on large test sets.
    if len(y_test) > ci_sample_size:
        rng = rng or np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(y_test), ci_sample_size, replace=False)
        ci = concordance_index(y_test[idx], y_pred[idx])
    else:
        ci = concordance_index(y_test, y_pred)

    print(f"{name:>15} | RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}  CI: {ci:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "ci": ci}


def main():
    X, y, feature_cols = load_dataset()
    print(f"Loaded {len(X)} samples, {len(feature_cols)} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append(evaluate(name, y_test, y_pred))

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/baseline_metrics.csv", index=False)
    print("\nSaved metrics to results/baseline_metrics.csv")

    best = results_df.loc[results_df["rmse"].idxmin()]
    print(f"\nBest model by RMSE: {best['model']} (RMSE={best['rmse']:.4f}, R2={best['r2']:.4f})")

    rf_model = models["RandomForest"]
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print("\nTop 10 feature importances (Random Forest):")
    print(importances.head(10))
    importances.to_csv("results/feature_importances.csv", header=["importance"])


if __name__ == "__main__":
    main()
