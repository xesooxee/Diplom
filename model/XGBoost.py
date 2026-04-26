"""
САЙЖРУУЛСАН XGBoost — Чихрийн шижин таамаглал
===============================================
Feature Engineering + SMOTE + Hyperparameter Tuning + Threshold Tuning
"""

import argparse
import os
import sys
import pathlib
import joblib
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────
# Тохиргоо (NEW датасетийг голчлон ашиглана)
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "csv_path":        str(BASE_DIR / "data" / "diabetes.csv"),
    "target":          "diabetes",
    "features": [
        "gender", "age", "hypertension", "heart_disease",
        "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level",
    ],
    "categorical_cols": ["gender", "smoking_history"],
    "model_dir":       str(BASE_DIR / "model" / "xgb"),
    "labels":          ["Эрүүл", "Чихрийн шижин"],
}

# ─────────────────────────────────────────────────────────────
# Өгөгдөл унших + Feature Engineering
# ─────────────────────────────────────────────────────────────
def load_and_engineer(cfg):
    if not os.path.exists(cfg["csv_path"]):
        sys.exit(f"Файл олдсонгүй: {cfg['csv_path']}")

    df = pd.read_csv(cfg["csv_path"])
    print(f"📂 Өгөгдөл ачаалагдлаа: {df.shape[0]:,} мөр, {df.shape[1]} багана")

    # Categorical → numeric; encoder-уудыг хадгалж main.py inference-д ачаална
    encoders: dict = {}
    for col in cfg["categorical_cols"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    enc_path = BASE_DIR / "model" / "encoders.pkl"
    enc_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, enc_path)

    # Feature Engineering (сайжруулалт)
    df['glucose_hba1c_ratio'] = df['blood_glucose_level'] / (df['HbA1c_level'] + 1e-5)
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['high_risk_comorb'] = df['hypertension'] + df['heart_disease']
    df['metabolic_score'] = df['bmi'] + df['HbA1c_level'] * 10 + df['blood_glucose_level'] / 10

    # Шинэ feature-үүдийг features жагсаалтад нэмэх (global cfg-г мутейт хийхгүй)
    all_features = cfg["features"] + ['glucose_hba1c_ratio', 'age_bmi_interaction',
                                      'high_risk_comorb', 'metabolic_score']

    X = df[all_features].copy()
    y = df[cfg["target"]].copy()

    print(f"✨ Feature Engineering хийгдлээ. Нийт feature: {X.shape[1]}")
    return X, y, all_features


# ─────────────────────────────────────────────────────────────
# Гол сургалтын функц
# ─────────────────────────────────────────────────────────────
def train_improved_xgboost(X, y, cfg, all_features):
    # Train-test split (test set нь эцсийн үнэлгээнд л ашиглагдана)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Threshold tuning-д зориулсан validation set (train-аас тусгаарлана)
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # SMOTE ашиглан imbalance засах (зөвхөн train дээр!)
    print(f"\n SMOTE хийж байна... (Original positive: {y_train_fit.sum()})")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_fit, y_train_fit)
    print(f"   SMOTE-ийн дараа positive: {y_train_res.sum()}")

    # Hyperparameter search
    param_dist = {
        'n_estimators': [400, 500, 600],
        'max_depth': [5, 6, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
    }

    model = xgb.XGBClassifier(
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=25,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    print("\n🔍 Hyperparameter Tuning эхэлж байна...")
    search.fit(X_train_res, y_train_res)

    best_model = search.best_estimator_
    print(f" Best params: {search.best_params_}")
    print(f" Best CV ROC-AUC: {search.best_score_:.4f}")

    # Validation set дээр threshold tuning (test set-ийг хөндөхгүй)
    y_val_prob = best_model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.3, 0.8, 0.01)
    best_f1 = 0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred_t = (y_val_prob >= thresh).astype(int)
        current_f1 = f1_score(y_val, y_pred_t)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = thresh

    print(f"\n Оновчтой Threshold (validation F1-max): {best_thresh:.3f}  (F1 = {best_f1:.4f})")

    # Test set дээр таамаглал (threshold validation дээр сонгогдсон)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_thresh).astype(int)

    print("\n── Эцсийн үр дүн (Best Threshold) ─────────────────────")
    print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred, target_names=cfg["labels"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n── Confusion Matrix ───────────────────────────────")
    print(f"{'':>18} | {cfg['labels'][0]:>15} | {cfg['labels'][1]:>15}")
    print("-" * 60)
    for i, label in enumerate(cfg["labels"]):
        print(f"  Бодит {label:>10} | {cm[i][0]:>15} | {cm[i][1]:>15}")

    # Feature Importance
    importances = pd.Series(best_model.feature_importances_, index=all_features).sort_values(ascending=False)
    print("\n── Топ 10 Feature Importance ───────────────────────")
    for feat, imp in importances.head(10).items():
        print(f"  {feat:<30} {'█' * int(imp*40)} {imp:.4f}")

    # Загвар + threshold хадгалах
    os.makedirs(cfg["model_dir"], exist_ok=True)
    save_dict = {
        'model': best_model,
        'best_threshold': best_thresh,
        'features': all_features,
        'label_encoder_info': "gender & smoking_history encoded"
    }
    model_path = os.path.join(cfg["model_dir"], "pipeline.pkl")
    joblib.dump(save_dict, model_path)

    print(f"\n Сайжруулсан модель амжилттай хадгалагдлаа → {model_path}")
    return best_model, best_thresh


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Сайжруулсан XGBoost Diabetes Model")
    parser.add_argument("--dataset", choices=["new"], default="new")
    args = parser.parse_args()

    cfg = CONFIG

    print("=" * 75)
    print("   САЙЖРУУЛСАН XGBoost — Feature Engineering + SMOTE + Tuning")
    print("=" * 75)

    X, y, all_features = load_and_engineer(cfg)
    model, threshold = train_improved_xgboost(X, y, cfg, all_features)

    print(f"\n Амжилттай дууслаа! Санал болгож буй threshold: {threshold:.3f}")


if __name__ == "__main__":
    main()