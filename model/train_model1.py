"""
Trains Model 1: XGBoost classifier on the full mixed-sex thyroid cohort,
predicting thyroid_dysfunction. Cohort is 69% female (2182F/908M), so
this is a generic baseline, not a "male-skewed" one.

Comparable to Model 2 (same target, PCOS cohort) but not a clean
sex-representation test alone -- dataset size/label quality/features differ.
Cannot predict PCOS: no PCOS labels/features exist for this cohort.
"""

import os
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib

from pipeline_common import XGBOOST_DETERMINISTIC_PARAMS, save_json_with_metadata

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "splits")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TARGET = "thyroid_dysfunction"

FEATURE_COLS = ["age", "sex", "pregnant", "sick", "tsh", "t3", "t4", "t4_uptake", "free_thyroxine_index"]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["pregnant"] = df["pregnant"].map({"f": 0, "t": 1})
    df["sick"] = df["sick"].map({"f": 0, "t": 1})
    return df[FEATURE_COLS]


def load_split(name: str):
    df = pd.read_csv(os.path.join(DATA_DIR, f"thyroid_{name}.csv"))
    X = encode_features(df)
    y = df[TARGET]
    return X, y, df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_test, y_test, test_df = load_split("test")

    # ~4.8% positive; scale_pos_weight balances gradients instead of resampling.
    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    scale_pos_weight = n_neg / n_pos
    print(f"Train class balance: {n_neg} negative, {n_pos} positive (scale_pos_weight={scale_pos_weight:.2f})")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        **XGBOOST_DETERMINISTIC_PARAMS,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    print("\n=== Model 1 (generic clinical baseline) -- thyroid_dysfunction ===")
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y, proba)
        print(f"\n[{name}] AUC = {auc:.3f}")
        print(f"[{name}] confusion matrix (rows=actual, cols=predicted):\n{confusion_matrix(y, pred)}")

    print("\n[test] full classification report:")
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    print(classification_report(y_test, pred_test, target_names=["no dysfunction", "dysfunction"]))

    # Check whether Model 1 performs differently for men vs. women.
    print("\n--- Sex-disaggregated test performance ---")
    sex_metrics = {}
    for sex_label in ["female", "male"]:
        mask = (test_df["sex"] == sex_label).values
        if mask.sum() < 5:
            print(f"  {sex_label}: too few test rows ({mask.sum()}) to evaluate")
            continue
        y_sub, proba_sub = y_test[mask], proba_test[mask]
        pred_sub = (proba_sub >= 0.5).astype(int)
        auc_sub = roc_auc_score(y_sub, proba_sub) if y_sub.nunique() > 1 else float("nan")
        print(f"  {sex_label} (n={mask.sum()}, positives={int(y_sub.sum())}): AUC={auc_sub:.3f}")
        print(f"    confusion matrix:\n{confusion_matrix(y_sub, pred_sub)}")
        sex_metrics[sex_label] = {"n": int(mask.sum()), "positives": int(y_sub.sum()), "auc": float(auc_sub)}

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances.to_string())

    model_path = os.path.join(MODEL_DIR, "model1_generic_baseline.joblib")
    joblib.dump(model, model_path)
    print(f"\nSaved -> {model_path}")

    metrics = {
        "model": "model1_generic_clinical_baseline",
        "target": TARGET,
        "train_auc": float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])),
        "val_auc": float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])),
        "test_auc": float(roc_auc_score(y_test, proba_test)),
        "sex_disaggregated_test_auc": sex_metrics,
        "feature_importances": importances.to_dict(),
        "naming_correction": (
            "Previously called 'male-skewed'; this cohort is actually "
            "69% female (2182F/908M), so that name was never verified "
            "against the data. Renamed to 'generic clinical baseline'."
        ),
    }
    metrics_path = os.path.join(MODEL_DIR, "model1_metrics.json")
    save_json_with_metadata(metrics, metrics_path)
    print(f"Saved -> {metrics_path}")


if __name__ == "__main__":
    main()