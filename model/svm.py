"""
Чихрийн шижин таамаглах — SUPPORT VECTOR MACHINE (SVM)
======================================================
Ажиллуулах жишээ:
  python model/train.py --dataset new --predict
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# BASE_DIR
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

# Тохиргоо
PIMA_CONFIG = {
    "csv_path": str(BASE_DIR / "data" / "diabetes.csv"),
    "target": "Outcome",
    "zero_fix_cols": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    "features": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
    "categorical_cols": [],
    "model_dir": str(BASE_DIR / "model" / "svm_pima"),
    "labels": ["Эрүүл", "Чихрийн шижин"],
}

NEW_CONFIG = {
    "csv_path": str(BASE_DIR / "data" / "diabetes.csv"),
    "target": "diabetes",
    "zero_fix_cols": [],
    "features": ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"],
    "categorical_cols": ["gender", "smoking_history"],
    "model_dir": str(BASE_DIR / "model" / "svm"),
    "labels": ["Эрүүл", "Чихрийн шижин"],
}

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 10.0,
    "gamma": "scale",
    "class_weight": "balanced",
    "probability": True,
    "random_state": 42,
    "max_iter": 1000,
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

    for col in cfg["categorical_cols"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    X = df[cfg["features"]].copy()
    y = df[cfg["target"]].copy()

    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median(numeric_only=True))

    return X, y


def train_svm(X, y, cfg):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n Train: {len(X_train)}, Test: {len(X_test)}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(**SVM_PARAMS)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f" 5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

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
    joblib.dump(pipe, os.path.join(cfg["model_dir"], "pipeline.pkl"))
    print(f"\n SVM модель хадгалагдлаа → {cfg['model_dir']}/pipeline.pkl")

    return pipe


def predict_sample(pipe, cfg, dataset):
    sample = pd.DataFrame([{
        "Pregnancies": 2, "Glucose": 138, "BloodPressure": 62, "SkinThickness": 35,
        "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50
    }]) if dataset == "pima" else pd.DataFrame([{
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
    parser.add_argument("--dataset", choices=["pima", "new"], default="new")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    cfg = PIMA_CONFIG if args.dataset == "pima" else NEW_CONFIG

    print("=" * 70)
    print(f"  🩺 SVM Чихрийн шижин таамаглал — [{args.dataset.upper()}]")
    print("=" * 70)

    X, y = load_and_prepare(cfg)
    pipe = train_svm(X, y, cfg)

    if args.predict:
        predict_sample(pipe, cfg, args.dataset)

    print("\n✔ Дууслаа!\n")


if __name__ == "__main__":
    main()