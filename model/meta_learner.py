"""
meta_learner.py

Combines model outputs into the final 4-way joint prediction:
  0 = neither, 1 = thyroid only, 2 = PCOS only, 3 = both

Model 1 is excluded: it needs T3/T4/T4U/FTI, which the PCOS cohort never
has (100% missing - a structural mismatch, not sporadic missing data), so
it has no learnable signal on the cohort this meta-learner is trained on.

Meta-features used (all from the PCOS cohort):
  - model2_thyroid_proba   (Model 2,  women-only, thyroid_dysfunction)
  - model2_pcos_proba      (Model 2b, women-only, pcos_diagnosis)
  - pcos_criteria_count    (Model 3,  Rotterdam criteria, 0-3)
  - thyroid_criteria_call  (Model 3,  TSH threshold, 0/1)
"""

import os
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Reuse the base models' feature-prep so inputs match what they were trained on.
from train_model2 import clean_features as clean_features_model2
from train_model2b_pcos import clean_features as clean_features_model2b
from model3 import pcos_criteria_score, thyroid_criteria_score
from pipeline_common import CLASS_NAMES, save_json_with_metadata

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "splits")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")



def build_meta_features(df: pd.DataFrame, model2, model2b) -> pd.DataFrame:
    model2_thyroid_proba = model2.predict_proba(clean_features_model2(df))[:, 1]
    model2_pcos_proba = model2b.predict_proba(clean_features_model2b(df))[:, 1]

    pcos_scores = pcos_criteria_score(df)
    thyroid_call = thyroid_criteria_score(df["tsh"])

    return pd.DataFrame({
        "model2_thyroid_proba": model2_thyroid_proba,
        "model2_pcos_proba": model2_pcos_proba,
        "pcos_criteria_count": pcos_scores["pcos_criteria_count"].values,
        "thyroid_criteria_call": thyroid_call.values,
    })


def main():
    model2 = joblib.load(os.path.join(MODEL_DIR, "model2_women_only.joblib"))
    model2b = joblib.load(os.path.join(MODEL_DIR, "model2b_women_only_pcos.joblib"))

    splits = {}
    for name in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(DATA_DIR, f"pcos_{name}.csv"))
        X_meta = build_meta_features(df, model2, model2b)
        y = df["joint_class"]
        splits[name] = (X_meta, y)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    print("Meta-feature preview (train, first 5 rows):")
    print(X_train.head())

    # Logistic regression: simple, enough for 4 features and ~380 rows.
    meta_model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42,
    )
    meta_model.fit(X_train, y_train)

    print("\n=== Meta-learner -- 4-class joint prediction ===")
    for name, (X, y) in splits.items():
        pred = meta_model.predict(X)
        acc = accuracy_score(y, pred)
        print(f"\n[{name}] accuracy = {acc:.3f}")
        print(f"[{name}] confusion matrix (rows=actual, cols=predicted, order=0,1,2,3):\n{confusion_matrix(y, pred, labels=[0,1,2,3])}")

    print("\n[test] full classification report:")
    pred_test = meta_model.predict(X_test)
    print(classification_report(y_test, pred_test, target_names=[CLASS_NAMES[i] for i in range(4)], zero_division=0))

    print("\nLearned coefficients (per class, per meta-feature):")
    coef_df = pd.DataFrame(meta_model.coef_, columns=X_train.columns, index=[CLASS_NAMES[i] for i in meta_model.classes_])
    print(coef_df.round(3))

    model_path = os.path.join(MODEL_DIR, "meta_learner.joblib")
    joblib.dump(meta_model, model_path)
    print(f"\nSaved -> {model_path}")

    metrics = {
        "model": "meta_learner",
        "meta_features": list(X_train.columns),
        "train_accuracy": float(accuracy_score(y_train, meta_model.predict(X_train))),
        "val_accuracy": float(accuracy_score(y_val, meta_model.predict(X_val))),
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "excluded_model1_reason": (
            "Model 1 requires T3/T4/T4U/FTI, unavailable for 100% of the "
            "PCOS cohort (structural feature mismatch, not missing data)."
        ),
    }
    save_json_with_metadata(metrics, os.path.join(MODEL_DIR, "meta_learner_metrics.json"))
    print(f"Saved -> {os.path.join(MODEL_DIR, 'meta_learner_metrics.json')}")


if __name__ == "__main__":
    main()