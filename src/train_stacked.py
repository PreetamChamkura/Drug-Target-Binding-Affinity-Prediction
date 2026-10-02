"""Stacked ensemble on top of the baseline models in src/train_baseline.py.

Ridge, RandomForest, and XGBoost are used as base learners; their
out-of-fold predictions are combined by a gradient-boosting meta-learner
(the "boosting layer") via sklearn's StackingRegressor. This typically
beats any single base model since it lets the meta-learner learn which
base model to trust in which region of feature space.

Usage:
    python -m src.train_stacked
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.train_baseline import RANDOM_STATE, evaluate, load_dataset


def build_stacked_model():
    base_learners = [
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ("random_forest", RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
        )),
        ("xgboost", XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ]
    meta_learner = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
    )
    return StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )


def main():
    X, y, _ = load_dataset()
    print(f"Loaded {len(X)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    stacked_model = build_stacked_model()
    stacked_model.fit(X_train, y_train)
    y_pred = stacked_model.predict(X_test)
    result = evaluate("Stacked", y_test, y_pred)

    try:
        baseline_df = pd.read_csv("results/baseline_metrics.csv")
        baseline_df = baseline_df[baseline_df["model"] != "Stacked"]
        results_df = pd.concat([baseline_df, pd.DataFrame([result])], ignore_index=True)
    except FileNotFoundError:
        results_df = pd.DataFrame([result])

    results_df.to_csv("results/baseline_metrics.csv", index=False)
    print("\nUpdated results/baseline_metrics.csv with stacked ensemble row")

    best = results_df.loc[results_df["rmse"].idxmin()]
    print(f"Best model overall: {best['model']} (RMSE={best['rmse']:.4f}, R2={best['r2']:.4f})")


if __name__ == "__main__":
    main()
