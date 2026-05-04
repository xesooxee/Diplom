"""
Чихрийн шижин таамаглах ML Pipeline — Random Forest
=====================================================
Ажиллуулах арга:
  cd <төслийн үндсэн хавтас>
  python model/train.py
  python model/train.py --predict
"""

import argparse
import os
import sys
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
# 0. Үндсэн зам — train.py хаанаас ажиллуулсан ч зөв олно
# ─────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # → backend/

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
    "model_dir":        str(BASE_DIR / "model" / "rf"),
    "labels":           ["Эрүүл", "Чихрийн шижин"],
}

RF_PARAMS = {
    "n_estimators":    200,
    "max_depth":       8,
    "min_samples_split": 5,
    "class_weight":    "balanced",
    "random_state":    42,
    "n_jobs":          -1,
}

# ─────────────────────────────────────────────────────────────
# 2. Өгөгдөл бэлтгэх
# ─────────────────────────────────────────────────────────────
def load_and_prepare(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """CSV уншиж, цэвэрлэж, X, y буцаана."""

    if not os.path.exists(cfg["csv_path"]):
        sys.exit(
            f"\n[Алдаа] Файл олдсонгүй: {cfg['csv_path']}\n"
            "  → CSV файлаа data/ хавтаст байрлуулна уу.\n"
        )

    df = pd.read_csv(cfg["csv_path"])
    print(f"\n Датасет ачаалагдлаа  →  {df.shape[0]} мөр, {df.shape[1]} багана")

    # 0 → NaN засал (Pima датасетэд)
    for col in cfg["zero_fix_cols"]:
        if col in df.columns:
            n_zeros = (df[col] == 0).sum()
            if n_zeros:
                df[col] = df[col].replace(0, np.nan)
                df[col] = df[col].fillna(df[col].median())
                print(f"  ✔ {col}: {n_zeros} тэгийг медианаар ({df[col].median():.1f}) орлуулав")

    # Categorical → numeric (LabelEncoder)
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

    # Байхгүй feature шалгах
    missing = [f for f in cfg["features"] if f not in df.columns]
    if missing:
        sys.exit(f"[Алдаа] Дутуу багана(ууд): {missing}")

    X = df[cfg["features"]].copy()
    y = df[cfg["target"]].copy()

    # NaN шалгах
    nan_count = X.isnull().sum().sum()
    if nan_count:
        print(f"  ⚠  Нийт {nan_count} NaN утга байна → медианаар дүүргэж байна")
        X = X.fillna(X.median(numeric_only=True))

    return X, y


# ─────────────────────────────────────────────────────────────
# 3. Загвар сургах ба үнэлэх
# ─────────────────────────────────────────────────────────────
def train_and_evaluate(X: pd.DataFrame, y: pd.Series, cfg: dict) -> Pipeline:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Threshold tuning-д зориулсан validation set (train-аас тусгаарлана)
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    print(f"\n Train: {len(X_train_fit)}, Val: {len(X_val)}, Test: {len(X_test)}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(**RF_PARAMS)),
    ])

    # ── 5-Fold Cross-Validation ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train_fit, y_train_fit, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\n 5-Fold CV ROC-AUC:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Эцсийн сургалт ──
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

    y_pred      = (pipe.predict_proba(X_test)[:, 1] >= best_thresh).astype(int)
    y_pred_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\n Test Accuracy : {acc:.4f}  ({acc * 100:.1f}%)")
    print(f" ROC-AUC Score : {auc:.4f}")
    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y_test, y_pred, target_names=cfg["labels"]))

    # ── Confusion Matrix ──
    cm     = confusion_matrix(y_test, y_pred)
    labels = cfg["labels"]
    print("── Confusion Matrix ───────────────────────────────")
    print(f"{'':>18} | {labels[0]:>15} | {labels[1]:>15}")
    print(f"  {'-' * 55}")
    for i, row_label in enumerate(labels):
        print(f"  Бодит {row_label:>10} | {cm[i][0]:>15} | {cm[i][1]:>15}")

    # ── Feature Importance ──
    rf_model    = pipe.named_steps["clf"]
    importances = pd.Series(rf_model.feature_importances_, index=cfg["features"])
    importances = importances.sort_values(ascending=False)
    print("\n── Feature Importance (дээд 5) ────────────────────")
    for feat, imp in importances.head(5).items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<30} {bar} {imp:.4f}")

    # ── Загвар + threshold хадгалах ──
    os.makedirs(cfg["model_dir"], exist_ok=True)
    model_path = os.path.join(cfg["model_dir"], "pipeline.pkl")
    joblib.dump({"model": pipe, "best_threshold": best_thresh}, model_path)
    print(f"\n Pipeline хадгалагдлаа → {model_path}")

    return pipe


# ─────────────────────────────────────────────────────────────
# 4. Жишээ таамаглал
# ─────────────────────────────────────────────────────────────
SAMPLE = pd.DataFrame([{
    "gender": "Female", "age": 45.0, "hypertension": 0, "heart_disease": 0,
    "smoking_history": "never", "bmi": 28.5, "HbA1c_level": 6.5,
    "blood_glucose_level": 140,
}])


def predict_sample(pipe: Pipeline, cfg: dict):
    sample = SAMPLE.copy()

    for col in cfg["categorical_cols"]:
        if col in sample.columns:
            le = LabelEncoder()
            le.fit(["Female", "Male"] if col == "gender"
                   else ["No Info", "current", "ever", "former", "never", "not current"])
            sample[col] = le.transform(sample[col].astype(str))

    pred  = pipe.predict(sample)[0]
    proba = pipe.predict_proba(sample)[0]

    print("\n── Жишээ таамаглал ────────────────────────────────")
    print(f"  Оролт      : {sample.to_dict(orient='records')[0]}")
    print(f"  Таамаглал  : {cfg['labels'][pred]}")
    print(f"  Магадлал   : Эрүүл {proba[0]*100:.1f}%  |  Чихрийн шижин {proba[1]*100:.1f}%")


# ─────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Чихрийн шижин таамаглах ML Pipeline")
    parser.add_argument(
        "--predict", action="store_true",
        help="Сургасны дараа жишээ таамаглал ажиллуулах"
    )
    args = parser.parse_args()

    cfg = CONFIG

    print("=" * 58)
    print(f"   Чихрийн шижин ML Pipeline")
    print(f"   BASE_DIR: {BASE_DIR}")
    print("=" * 58)

    X, y = load_and_prepare(cfg)
    pipe  = train_and_evaluate(X, y, cfg)

    if args.predict:
        predict_sample(pipe, cfg)

    print("\n✔ Амжилттай дууслаа!\n")


if __name__ == "__main__":
    main()