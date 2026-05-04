"""
Чихрийн шижин таамаглах ML Pipeline — LOGISTIC REGRESSION
============================================================
Ажиллуулах арга:
  cd <төслийн үндсэн хавтас>
  python model/logisticRegression.py
  python model/logisticRegression.py --predict
"""

import argparse
import os
import sys
import pathlib

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────────────────────
# 0. BASE_DIR
# ─────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────
# 1. Тохиргоо (Config)
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "csv_path":        str(BASE_DIR / "data" / "diabetes.csv"),
    "target":          "diabetes",
    "zero_fix_cols":   [],
    "features": [
        "gender", "age", "hypertension", "heart_disease",
        "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level",
    ],
    "categorical_cols": ["gender", "smoking_history"],
    "model_dir":        str(BASE_DIR / "model" / "lr"),
    "labels":           ["Эрүүл", "Чихрийн шижин"],
}

# Logistic Regression-ийн тохиргоо
LR_PARAMS = {
    "max_iter": 1000,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",      # тэнцвэргүй датасетэд зориулсан
    "solver": "lbfgs"
}

# ─────────────────────────────────────────────────────────────
# 2. Өгөгдөл бэлтгэх
# ─────────────────────────────────────────────────────────────
def load_and_prepare(cfg: dict):
    if not os.path.exists(cfg["csv_path"]):
        sys.exit(f"\n[Алдаа] Файл олдсонгүй: {cfg['csv_path']}\n")

    df = pd.read_csv(cfg["csv_path"])
    print(f"\n Датасет ачаалагдлаа → {df.shape[0]} мөр, {df.shape[1]} багана")

    # Pima датасетэд 0 утгыг медианаар солих
    for col in cfg["zero_fix_cols"]:
        if col in df.columns:
            n_zeros = (df[col] == 0).sum()
            if n_zeros:
                df[col] = df[col].replace(0, np.nan)
                df[col] = df[col].fillna(df[col].median())
                print(f"  ✔ {col}: {n_zeros} тэгийг медианаар солив")

    # Категори багануудыг тоон болгох
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
            print(f"  ✔ {col}: категори → тоо болгов  ({classes})")

    enc_path = BASE_DIR / "model" / "encoders.pkl"
    enc_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, enc_path)
    print(f"  ✔ Encoder-ууд хадгалагдлаа → {enc_path.name}")

    missing = [f for f in cfg["features"] if f not in df.columns]
    if missing:
        sys.exit(f"[Алдаа] Дутуу багана: {missing}")

    X = df[cfg["features"]].copy()
    y = df[cfg["target"]].copy()

    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median(numeric_only=True))

    return X, y


# ─────────────────────────────────────────────────────────────
# 3. Logistic Regression сургах
# ─────────────────────────────────────────────────────────────
def train_logistic_regression(X: pd.DataFrame, y: pd.Series, cfg: dict):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    print(f"\n Train: {len(X_train_fit)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Pipeline: StandardScaler + Logistic Regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**LR_PARAMS)),
    ])

    # 5-Fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train_fit, y_train_fit, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f" 5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Загвар сургах
    pipe.fit(X_train_fit, y_train_fit)

    # ── Threshold tuning (recall >= MIN_RECALL, дараа F1 maximize) ──
    MIN_RECALL = 0.82
    y_val_prob = pipe.predict_proba(X_val)[:, 1]
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

    # Үнэлгээ
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_thresh).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n Logistic Regression Test Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f" ROC-AUC Score                    : {auc:.4f}")
    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y_test, y_pred, target_names=cfg["labels"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n── Confusion Matrix ───────────────────────────────")
    print(f"{'':>18} | {cfg['labels'][0]:>15} | {cfg['labels'][1]:>15}")
    print(f"  {'-' * 55}")
    for i, label in enumerate(cfg["labels"]):
        print(f"  Бодит {label:>10} | {cm[i][0]:>15} | {cm[i][1]:>15}")

    # Feature Importance (Logistic Regression-д коэффициент ашиглана)
    coefficients = pipe.named_steps["clf"].coef_[0]
    feature_names = cfg["features"]
    coef_df = pd.Series(coefficients, index=feature_names).abs().sort_values(ascending=False)

    print("\n── Feature Importance (Коэффициент | Дээд 8) ────────")
    for feat, coef in coef_df.head(8).items():
        bar = "█" * int(coef * 30)
        print(f"  {feat:<32} {bar} {coef:.4f}")

    # Загвар + threshold хадгалах
    os.makedirs(cfg["model_dir"], exist_ok=True)
    model_path = os.path.join(cfg["model_dir"], "pipeline.pkl")
    joblib.dump({"model": pipe, "best_threshold": best_thresh}, model_path)
    print(f"\n Logistic Regression Pipeline хадгалагдлаа → {model_path}")

    return pipe


# ─────────────────────────────────────────────────────────────
# 4. Жишээ таамаглал
# ─────────────────────────────────────────────────────────────
SAMPLE = pd.DataFrame([{
    "gender": "Female", "age": 45.0, "hypertension": 0, "heart_disease": 0,
    "smoking_history": "never", "bmi": 28.5, "HbA1c_level": 6.5,
    "blood_glucose_level": 140,
}])


def predict_sample(pipe, cfg):
    sample = SAMPLE.copy()

    for col in cfg["categorical_cols"]:
        if col in sample.columns:
            le = LabelEncoder()
            le.fit(["Female", "Male"] if col == "gender"
                   else ["No Info", "current", "ever", "former", "never", "not current"])
            sample[col] = le.transform(sample[col].astype(str))

    pred = pipe.predict(sample)[0]
    proba = pipe.predict_proba(sample)[0]

    print("\n── Жишээ таамаглал ────────────────────────────────")
    print(f"  Таамаглал  : {cfg['labels'][pred]}")
    print(f"  Магадлал   : Эрүүл {proba[0]*100:.1f}%  |  Чихрийн шижин {proba[1]*100:.1f}%")


# ─────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Чихрийн шижин таамаглах — Logistic Regression")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    cfg = CONFIG

    print("=" * 65)
    print("  🩺 Чихрийн шижин таамаглал — LOGISTIC REGRESSION")
    print("=" * 65)

    X, y = load_and_prepare(cfg)
    pipe = train_logistic_regression(X, y, cfg)

    if args.predict:
        predict_sample(pipe, cfg)

    print("\n✔ Амжилттай дууслаа!\n")


if __name__ == "__main__":
    main()