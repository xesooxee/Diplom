"""
Чихрийн шижин таамаглах — SUPPORT VECTOR MACHINE (SVM)
======================================================
Ажиллуулах жишээ:
  python model/svm.py
"""

import argparse
import os
import sys
import pathlib

import joblib
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# BASE_DIR
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

# Тохиргоо
CONFIG = {
    "csv_path": str(BASE_DIR / "data" / "diabetes.csv"),
    "target": "diabetes",
    "zero_fix_cols": [],
    "features": ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"],
    "categorical_cols": ["gender", "smoking_history"],
    "model_dir": str(BASE_DIR / "model" / "svm"),
    "labels": ["Эрүүл", "Чихрийн шижин"],
}

# RandomizedSearchCV-д ашиглах parameter grid
# Pipeline-тай ашигладаг тул "clf__" prefix шаардлагатай
SVM_PARAM_DIST = {
    "clf__C":      [0.1, 1.0, 10.0, 100.0],
    "clf__gamma":  ["scale", "auto", 0.001, 0.01, 0.1],
    "clf__kernel": ["rbf", "sigmoid"],
}

# Суурь SVM тохиргоо (class_weight, probability бол туning-д оролцохгүй)
SVM_BASE = {
    "class_weight": "balanced",
    "probability":  True,
    "random_state": 42,
    "max_iter":     5000,
}

def load_and_prepare(cfg):
    if not os.path.exists(cfg["csv_path"]):
        sys.exit(f"\n[Алдаа] Файл олдсонгүй: {cfg['csv_path']}\n")

    df = pd.read_csv(cfg["csv_path"])
    print(f"\n Датасет ачаалагдлаа → {df.shape[0]} мөр, {df.shape[1]} багана")

    for col in cfg["zero_fix_cols"]:
        if col in df.columns:
            n_zeros = (df[col] == 0).sum()
            if n_zeros:
                df[col] = df[col].replace(0, np.nan)
                df[col] = df[col].fillna(df[col].median())

    # Fit хийсэн encoder-уудыг pkl-д хадгалж, main.py inference-д ачаална.
    EXPECTED_GENDER  = ["Female", "Male"]
    EXPECTED_SMOKING = ["No Info", "current", "ever", "former", "never", "not current"]
    encoders: dict = {}
    for col in cfg["categorical_cols"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            classes = list(le.classes_)
            if col == "gender":
                assert classes == EXPECTED_GENDER, \
                    f"gender encoding өөрчлөгдлөө! Одоо: {classes}. main.py-ийн _encode()-г шинэчилнэ үү."
            elif col == "smoking_history":
                assert classes == EXPECTED_SMOKING, \
                    f"smoking_history encoding өөрчлөгдлөө! Одоо: {classes}. main.py-ийн SMOKING_CLASSES-г шинэчилнэ үү."

    enc_path = BASE_DIR / "model" / "encoders.pkl"
    enc_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, enc_path)

    X = df[cfg["features"]].copy()
    y = df[cfg["target"]].copy()

    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median(numeric_only=True))

    return X, y


def train_svm(X, y, cfg):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n Train: {len(X_train)}, Test: {len(X_test)}")

    # Pipeline: StandardScaler + SVC
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(**SVM_BASE)),
    ])

    # ── Hyperparameter Tuning (RandomizedSearchCV) ──────────────
    # RF болон XGBoost-тай адил оновчтой параметр хайна
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=SVM_PARAM_DIST,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    print("\n🔍 SVM Hyperparameter Tuning эхэлж байна...")
    search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    print(f" Best params : {search.best_params_}")
    print(f" Best CV ROC-AUC: {search.best_score_:.4f}")

    # ── Эцсийн үнэлгээ (test set) ────────────────────────────────
    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]

    print(f"\n SVM Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f" ROC-AUC      : {roc_auc_score(y_test, y_prob):.4f}")
    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y_test, y_pred, target_names=cfg["labels"]))

    cm = confusion_matrix(y_test, y_pred)
    print("\n── Confusion Matrix ───────────────────────────────")
    print(f"{'':>18} | {cfg['labels'][0]:>15} | {cfg['labels'][1]:>15}")
    print(f"  {'-' * 55}")
    for i, label in enumerate(cfg["labels"]):
        print(f"  Бодит {label:>10} | {cm[i][0]:>15} | {cm[i][1]:>15}")

    # Хадгалах
    os.makedirs(cfg["model_dir"], exist_ok=True)
    joblib.dump(best_pipe, os.path.join(cfg["model_dir"], "pipeline.pkl"))
    print(f"\n SVM модель хадгалагдлаа → {cfg['model_dir']}/pipeline.pkl")

    return best_pipe


def predict_sample(pipe, cfg):
    sample = pd.DataFrame([{
        "gender": "Female", "age": 45.0, "hypertension": 0, "heart_disease": 0,
        "smoking_history": "never", "bmi": 28.5, "HbA1c_level": 6.5, "blood_glucose_level": 140
    }])

    for col in cfg["categorical_cols"]:
        if col in sample.columns:
            le = LabelEncoder()
            le.fit(["Female", "Male"] if col == "gender" else ["No Info", "current", "ever", "former", "never", "not current"])
            sample[col] = le.transform(sample[col].astype(str))

    pred = pipe.predict(sample)[0]
    proba = pipe.predict_proba(sample)[0]

    print("\n── Жишээ таамаглал ────────────────────────────────")
    print(f"  Таамаглал  : {cfg['labels'][pred]}")
    print(f"  Магадлал   : Эрүүл {proba[0]*100:.1f}%  |  Чихрийн шижин {proba[1]*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="SVM Чихрийн шижин таамаглал")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    cfg = CONFIG

    print("=" * 70)
    print("  🩺 SVM Чихрийн шижин таамаглал")
    print("=" * 70)

    X, y = load_and_prepare(cfg)
    pipe = train_svm(X, y, cfg)

    if args.predict:
        predict_sample(pipe, cfg)

    print("\n✔ Дууслаа!\n")


if __name__ == "__main__":
    main()