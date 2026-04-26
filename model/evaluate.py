"""
Бүх загваруудын бодит үнэлгээний метрик болон диаграм гаргах скрипт.

Ажиллуулах:
  cd /Users/temuulen/Documents/diplom/backend
  python model/evaluate.py
"""

import pathlib
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # дэлгэцгүй орчинд ажиллуулахад
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

# ── Замууд ────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "diabetes.csv"
OUT_DIR    = BASE_DIR / "model" / "evaluation"
OUT_DIR.mkdir(exist_ok=True)

SMOKING_CLASSES = ["No Info", "current", "ever", "former", "never", "not current"]

MODELS = {
    "Random Forest":        BASE_DIR / "model" / "rf"  / "pipeline.pkl",
    "Logistic Regression":  BASE_DIR / "model" / "lr"  / "pipeline.pkl",
    "SVM":                  BASE_DIR / "model" / "svm" / "pipeline.pkl",
    "XGBoost":              BASE_DIR / "model" / "xgb" / "pipeline.pkl",
}

COLORS = {
    "Random Forest":       "#2ec4b6",
    "Logistic Regression": "#e8a838",
    "SVM":                 "#e85555",
    "XGBoost":             "#7c5cbf",
}

# ── Өгөгдөл бэлдэх ────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    for col in ["gender", "smoking_history"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    features = ["gender", "age", "hypertension", "heart_disease",
                "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]
    X = df[features].values
    y = df["diabetes"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test

# ── XGBoost нэмэлт feature ────────────────────────────────────
def add_xgb_features(X_base: np.ndarray) -> np.ndarray:
    """XGBoost загвар 12 feature шаарддаг тул 4 engineered feature нэмнэ."""
    bmi   = X_base[:, 5]
    hba1c = X_base[:, 6]
    gluc  = X_base[:, 7]
    hyp   = X_base[:, 2]
    hd    = X_base[:, 3]
    age   = X_base[:, 1]

    ratio   = gluc / (hba1c + 1e-5)
    age_bmi = age * bmi
    comorb  = hyp + hd
    meta    = bmi + hba1c * 10 + gluc / 10

    return np.hstack([X_base, ratio[:,None], age_bmi[:,None],
                      comorb[:,None], meta[:,None]])

# ── Загвар ачаалж таамаглах ───────────────────────────────────
def predict_model(name: str, path: pathlib.Path, X_base: np.ndarray):
    obj = joblib.load(str(path))
    X   = add_xgb_features(X_base) if name == "XGBoost" else X_base

    if isinstance(obj, dict):
        model     = obj["model"]
        threshold = obj.get("best_threshold", 0.5)
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)
    else:
        prob = obj.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(int)

    return pred, prob

# ── Метрик тооцоолох ──────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-score":  f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_prob),
    }

# ─────────────────────────────────────────────────────────────
# ДИАГРАМ 1: ROC CURVE — бүх загвар нэг дотор
# ─────────────────────────────────────────────────────────────
def plot_roc_curves(results: dict, y_test: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (y_pred, y_prob) in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2.2, color=COLORS[name],
                label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Санамсаргүй (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    ax.set_title("ROC Curve — Загваруудын харьцуулалт", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "01_roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")

# ─────────────────────────────────────────────────────────────
# ДИАГРАМ 2: МЕТРИК ХАРЬЦУУЛАХ BAR CHART
# ─────────────────────────────────────────────────────────────
def plot_metrics_bar(metrics_dict: dict):
    names    = list(metrics_dict.keys())
    measures = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    x        = np.arange(len(measures))
    width    = 0.18

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(names):
        vals = [metrics_dict[name][m] for m in measures]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=COLORS[name], alpha=0.88, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="600")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(measures, fontsize=11)
    ax.set_ylabel("Оноо (0 – 1)", fontsize=11)
    ax.set_title("Гүйцэтгэлийн үзүүлэлтүүдийн харьцуулалт", fontsize=14, fontweight="bold")
    ax.set_ylim([0.5, 1.05])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "02_metrics_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")

# ─────────────────────────────────────────────────────────────
# ДИАГРАМ 3: CONFUSION MATRIX — 4 загвар дотор
# ─────────────────────────────────────────────────────────────
def plot_confusion_matrices(results: dict, y_test: np.ndarray):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Confusion Matrix — Загвар бүр", fontsize=15, fontweight="bold", y=1.01)

    for ax, (name, (y_pred, _)) in zip(axes.flat, results.items()):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Эрүүл (0)", "Чихрийн шижин (1)"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=12, fontweight="bold", color=COLORS[name])
        ax.set_xlabel("Таамагласан", fontsize=9)
        ax.set_ylabel("Бодит", fontsize=9)

    fig.tight_layout()
    path = OUT_DIR / "03_confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")

# ─────────────────────────────────────────────────────────────
# ДИАГРАМ 4: FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────────────────────
def plot_feature_importance(rf_path: pathlib.Path):
    obj  = joblib.load(str(rf_path))
    feat_names = ["gender", "age", "hypertension", "heart_disease",
                  "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]

    rf   = obj.named_steps["clf"]
    imps = pd.Series(rf.feature_importances_, index=feat_names).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(imps.index, imps.values, color=COLORS["Random Forest"],
                   alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, imps.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="600")

    ax.set_xlabel("Нөлөөллийн хэмжээ (Gini Importance)", fontsize=11)
    ax.set_title("Random Forest — Feature Importance", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "04_feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")

# ─────────────────────────────────────────────────────────────
# ДИАГРАМ 5: RADAR CHART — олон хэмжүүрт харьцуулалт
# ─────────────────────────────────────────────────────────────
def plot_radar(metrics_dict: dict):
    measures = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    N   = len(measures)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for name, m in metrics_dict.items():
        vals  = [m[k] for k in measures] + [m[measures[0]]]
        ax.plot(angles, vals, lw=2, color=COLORS[name], label=name)
        ax.fill(angles, vals, alpha=0.08, color=COLORS[name])

    ax.set_thetagrids(np.degrees(angles[:-1]), measures, fontsize=11)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=8, alpha=0.6)
    ax.set_title("Radar Chart — Загваруудын харьцуулалт", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "05_radar_chart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")

# ─────────────────────────────────────────────────────────────
# ДИАГРАМ 6: МЕТРИК ХҮСНЭГТ (хэвлэмэл зураг)
# ─────────────────────────────────────────────────────────────
def plot_metrics_table(metrics_dict: dict):
    measures = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    rows  = list(metrics_dict.keys())
    data  = [[f"{metrics_dict[r][m]:.4f}" for m in measures] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=data, rowLabels=rows,
        colLabels=measures, cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)

    # Толгой мөр өнгө
    for j in range(len(measures)):
        tbl[(0, j)].set_facecolor("#1a3340")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Мөр бүрийн загварын өнгө
    for i, name in enumerate(rows):
        tbl[(i + 1, -1)].set_text_props(color=COLORS[name], fontweight="bold")

    # Багана бүрийн хамгийн сайн утгыг тодруулах
    for j, measure in enumerate(measures):
        col_vals = [float(metrics_dict[r][measure]) for r in rows]
        best_row = int(np.argmax(col_vals))
        tbl[(best_row + 1, j)].set_facecolor("#e8f7f6")

    ax.set_title("Загваруудын харьцуулсан үнэлгээний хүснэгт",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    path = OUT_DIR / "06_metrics_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")

# ─────────────────────────────────────────────────────────────
# ГҮЙЦЭТГЭХ
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 56)
    print("   Чихрийн шижин — Загваруудын бодит үнэлгээ")
    print("=" * 56)

    print("\n📂 Өгөгдөл уншиж байна...")
    X_test, y_test = load_data()
    print(f"   Туршилтын багц: {len(y_test)} тохиолдол  "
          f"(чихрийн шижин: {y_test.sum()}, эрүүл: {(y_test==0).sum()})")

    results      = {}   # name → (y_pred, y_prob)
    metrics_dict = {}   # name → {metric: value}

    print("\n🔄 Загварууд ачаалж таамаглаж байна...\n")
    for name, path in MODELS.items():
        if not path.exists():
            print(f"  ⚠  {name}: файл олдсонгүй ({path})")
            continue
        y_pred, y_prob = predict_model(name, path, X_test)
        results[name]      = (y_pred, y_prob)
        metrics_dict[name] = compute_metrics(y_test, y_pred, y_prob)

        m = metrics_dict[name]
        print(f"  ✓ {name}")
        print(f"     Accuracy : {m['Accuracy']:.4f}  |  Precision: {m['Precision']:.4f}")
        print(f"     Recall   : {m['Recall']:.4f}  |  F1-score : {m['F1-score']:.4f}")
        print(f"     ROC-AUC  : {m['ROC-AUC']:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Эрүүл','Чихрийн шижин'], digits=4)}")

    if not results:
        print("❌ Ачааллах загвар байхгүй байна.")
        return

    print("\n🎨 Диаграмуудыг хадгалж байна...\n")
    plot_roc_curves(results, y_test)
    plot_metrics_bar(metrics_dict)
    plot_confusion_matrices(results, y_test)
    plot_feature_importance(MODELS["Random Forest"])
    plot_radar(metrics_dict)
    plot_metrics_table(metrics_dict)

    # metrics.json хадгалах — main.py best model сонгоход ашиглана
    import json
    KEY_MAP = {
        "Random Forest":       "rf",
        "Logistic Regression": "lr",
        "SVM":                 "svm",
        "XGBoost":             "xgb",
    }
    json_out = {KEY_MAP[n]: m for n, m in metrics_dict.items()}
    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  ✓ {metrics_path}  (best model сонгоход ашиглана)")

    print(f"\n✅ Бүх диаграм хадгалагдлаа → {OUT_DIR}/\n")
    print("   01_roc_curves.png         — ROC муруй")
    print("   02_metrics_bar.png        — Метрик харьцуулах баар")
    print("   03_confusion_matrices.png — Confusion Matrix (4 загвар)")
    print("   04_feature_importance.png — Feature Importance (RF)")
    print("   05_radar_chart.png        — Radar Chart")
    print("   06_metrics_table.png      — Хүснэгт зураг")


if __name__ == "__main__":
    main()
