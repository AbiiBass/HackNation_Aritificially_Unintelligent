"""
train_model2.py
Step 4b of the Women's Hormonal Health ensemble model pipeline.

Trains Model 2, the "women-only" model: an XGBoost classifier trained on
the PCOS cohort (541 women), predicting the SAME target as Model 1
(thyroid_dysfunction), so the two are directly comparable.

The point of this comparison: Model 2 does NOT have access to T3, T4,
T4U, or FTI -- the actual clinical thyroid panel that made Model 1 so
accurate. Those labs simply don't exist in this dataset. What it DOES
have is a much richer set of women's-health-context features (cycle
regularity, hirsutism, BMI, follicle counts, etc.) that Model 1 never
sees. This is meant to demonstrate, concretely, the "narrow women-only
model missing the right diagnostic markers" problem from the hackathon
brief -- not just assert it.

LEAKAGE NOTE (carried from label_engineering.py / split.py): `tsh` is
excluded from the feature set below because thyroid_dysfunction was
constructed directly from it. Using it as a feature here would just be
decoding the label, not predicting it.
"""

import os
import json
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib

from pipeline_common import XGBOOST_DETERMINISTIC_PARAMS, save_json_with_metadata

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "splits")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TARGET = "thyroid_dysfunction"

# Deliberately excludes tsh (leakage) and pcos_diagnosis/joint_class
# (those are outcomes, not predictors, for this particular model).
FEATURE_COLS = [
    "age", "bmi", "cycle_regularity", "cycle_length_days",
    "weight_gain", "hirsutism", "skin_darkening", "hair_loss", "acne",
    "fast_food", "regular_exercise", "bp_systolic", "bp_diastolic",
    "fsh", "lh", "fsh_lh_ratio", "amh", "prl", "vit_d3", "progesterone",
    "random_blood_sugar", "follicle_count_left", "follicle_count_right",
    "avg_follicle_size_left", "avg_follicle_size_right", "endometrium_mm",
]


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix the one stray cycle_regularity=5 value -> 4 (irregular), based on
    # that patient's very short cycle_length_days=7. Documented, not hidden.
    n_stray = (df["cycle_regularity"] == 5).sum()
    if n_stray:
        print(f"  [fix] cycle_regularity: remapped {n_stray} value(s) of 5 -> 4 (irregular)")
        df.loc[df["cycle_regularity"] == 5, "cycle_regularity"] = 4

    return df[FEATURE_COLS]


def load_split(name: str):
    df = pd.read_csv(os.path.join(DATA_DIR, f"pcos_{name}.csv"))
    X = clean_features(df)
    y = df[TARGET]
    return X, y


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    scale_pos_weight = n_neg / n_pos
    print(f"Train class balance: {n_neg} negative, {n_pos} positive (scale_pos_weight={scale_pos_weight:.2f})")

    # missing_value handles the 1 NaN in fast_food / amh natively (no
    # separate imputation step needed -- XGBoost learns a split direction
    # for missing values directly).
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,          # shallower than Model 1: much smaller dataset (541 vs 3163)
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        **XGBOOST_DETERMINISTIC_PARAMS,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    print("\n=== Model 2 (women-only) -- thyroid_dysfunction ===")
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

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("Feature importances:")
    print(importances.to_string())

    model_path = os.path.join(MODEL_DIR, "model2_women_only.joblib")
    joblib.dump(model, model_path)
    print(f"\nSaved -> {model_path}")

    metrics = {
        "model": "model2_women_only",
        "target": TARGET,
        "train_auc": float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])),
        "val_auc": float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])),
        "test_auc": float(roc_auc_score(y_test, proba_test)),
        "feature_importances": importances.to_dict(),
    }
    metrics_path = os.path.join(MODEL_DIR, "model2_metrics.json")
    save_json_with_metadata(metrics, metrics_path)
    print(f"Saved -> {metrics_path}")


if __name__ == "__main__":
    main()