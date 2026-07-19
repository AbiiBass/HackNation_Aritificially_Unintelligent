"""
Trains Model 2b: women-only XGBoost classifier predicting pcos_diagnosis
directly (PCOS-axis counterpart to Model 2's thyroid-axis predictor).

No leakage here: pcos_diagnosis wasn't derived from these features, so
the full feature set including tsh is usable.
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
TARGET = "pcos_diagnosis"

FEATURE_COLS = [
    "age", "bmi", "cycle_regularity", "cycle_length_days",
    "weight_gain", "hirsutism", "skin_darkening", "hair_loss", "acne",
    "fast_food", "regular_exercise", "bp_systolic", "bp_diastolic",
    "fsh", "lh", "fsh_lh_ratio", "amh", "prl", "vit_d3", "progesterone",
    "random_blood_sugar", "follicle_count_left", "follicle_count_right",
    "avg_follicle_size_left", "avg_follicle_size_right", "endometrium_mm",
    "tsh",  # no leakage: pcos_diagnosis isn't derived from tsh
]


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["cycle_regularity"] == 5, "cycle_regularity"] = 4  # remap stray value, as in Model 2
    return df[FEATURE_COLS]


def load_split(name: str):
    df = pd.read_csv(os.path.join(DATA_DIR, f"pcos_{name}.csv"))
    return clean_features(df), df[TARGET]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    scale_pos_weight = n_neg / n_pos
    print(f"Train class balance: {n_neg} negative, {n_pos} positive (scale_pos_weight={scale_pos_weight:.2f})")

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="auc", random_state=42,
        **XGBOOST_DETERMINISTIC_PARAMS,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    print("\n=== Model 2b (women-only) -- pcos_diagnosis ===")
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        print(f"\n[{name}] AUC = {roc_auc_score(y, proba):.3f}")
        print(f"[{name}] confusion matrix:\n{confusion_matrix(y, pred)}")

    proba_test = model.predict_proba(X_test)[:, 1]
    print("\n[test] full classification report:")
    print(classification_report(y_test, (proba_test >= 0.5).astype(int), target_names=["no PCOS", "PCOS"]))

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("Feature importances:")
    print(importances.to_string())

    model_path = os.path.join(MODEL_DIR, "model2b_women_only_pcos.joblib")
    joblib.dump(model, model_path)
    print(f"\nSaved -> {model_path}")

    metrics = {
        "model": "model2b_women_only_pcos", "target": TARGET,
        "train_auc": float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])),
        "val_auc": float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])),
        "test_auc": float(roc_auc_score(y_test, proba_test)),
        "feature_importances": importances.to_dict(),
    }
    save_json_with_metadata(metrics, os.path.join(MODEL_DIR, "model2b_metrics.json"))


if __name__ == "__main__":
    main()