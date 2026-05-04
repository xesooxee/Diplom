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

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
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

# LinearSVC parameter grid — Pipeline-тай тул "clf__base_estimator__C" prefix
LINEAR_SVM_PARAMS = {
    "clf__estimator__C": [0.01, 0.1, 1.0, 10.0, 100.0],
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
    EXPECTED_GENDER  = ["Female", "Male", "Other"]
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
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    neg = int((y_train_fit == 0).sum())
    pos = int((y_train_fit == 1).sum())
    print(f"\n Train: {len(X_train_fit)} (pos={pos}, neg={neg}), Val: {len(X_val)}, Test: {len(X_test)}")

    # LinearSVC + CalibratedClassifierCV:
    # - LinearSVC нь RBF SVM-аас 100x хурдан, том датасетэд тохиромжтой
    # - class_weight="balanced" тэнцвэргүй датасетийг зохицуулна
    # - CalibratedClassifierCV магадлал (predict_proba) гаргах боломж олгоно
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=5000, random_state=42),
            cv=3
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=LINEAR_SVM_PARAMS,
        n_iter=5,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    print("\n LinearSVM Hyperparameter Tuning эхэлж байна...")
    search.fit(X_train_fit, y_train_fit)

    best_svc = search.best_estimator_
    print(f" Best params : {search.best_params_}")
    print(f" Best CV ROC-AUC: {search.best_score_:.4f}")

    # ── Threshold tuning (recall >= MIN_RECALL, дараа F1 maximize) ──
    # Pipeline доторх scaler автоматаар хэрэглэгдэнэ
    MIN_RECALL = 0.82
    y_val_prob = best_svc.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.1, 0.85, 0.005)
    best_f1, best_thresh = 0, 0.5

    for thresh in thresholds:
        y_pred_t = (y_val_prob >= thresh).astype(int)
        r = recall_score(y_val, y_pred_t, zero_division=0)
        if r >= MIN_RECALL:
            current_f1 = f1_score(y_val, y_pred_t, zero_division=0)
            if current_f1 > best_f1:
                best_f1, best_thresh = current_f1, thresh

    if best_f1 == 0:
        print(f"  ⚠ Recall >= {MIN_RECALL} шаардлага хангагдсангүй. F1-max ашиглав.")
        for thresh in thresholds:
            y_pred_t = (y_val_prob >= thresh).astype(int)
            current_f1 = f1_score(y_val, y_pred_t, zero_division=0)
            if current_f1 > best_f1:
                best_f1, best_thresh = current_f1, thresh

    val_recall = recall_score(y_val, (y_val_prob >= best_thresh).astype(int), zero_division=0)
    print(f" Оновчтой Threshold: {best_thresh:.3f}  (val F1={best_f1:.4f}, val Recall={val_recall:.4f})")

    # ── Эцсийн үнэлгээ ────────────────────────────────────────────
    y_prob = best_svc.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_thresh).astype(int)

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

    # Хадгалах — pipeline (scaler + LinearSVM) + threshold
    os.makedirs(cfg["model_dir"], exist_ok=True)
    joblib.dump(
        {"model": best_svc, "best_threshold": best_thresh},
        os.path.join(cfg["model_dir"], "pipeline.pkl")
    )
    print(f"\n SVM модель хадгалагдлаа → {cfg['model_dir']}/pipeline.pkl")

    return best_svc, best_thresh


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
    best_svc, best_thresh = train_svm(X, y, cfg)

    if args.predict:
        predict_sample(best_svc, cfg)

    print("\n✔ Дууслаа!\n")


if __name__ == "__main__":
    main()