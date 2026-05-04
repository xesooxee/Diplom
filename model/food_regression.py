"""
Өдрийн хоол хүнснээс HbA1c болон Glucose таамаглах Regression
==============================================================
Ажиллуулах арга:
  cd <төслийн үндсэн хавтас>
  python model/food_regression.py
  python model/food_regression.py --predict

Сайжруулалтууд (v2):
  - Датасет 3 000 → 10 000 өдөр
  - 3 шинэ feature: glycemic_load, net_carbs, protein_fat_ratio
  - XGBoost загвар нэмэгдсэн
  - Бодит биеийн нөлөөллийг дуурайсан нон-линеар glucose томьёо
  - Гетероскедастик шуугиан (glucose ихэх тусам variance өсдөг)
"""

import argparse
import pathlib
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "nutrition.csv"
OUT_DIR   = BASE_DIR / "model" / "food_regression"
FOODS_OUT = BASE_DIR / "data" / "mongolian_foods.csv"

# Загвар оролтын баганууд (өдрийн нийт шим тэжээл)
FEATURES = [
    "total_calories",
    "total_carbs",
    "total_sugars",
    "total_fiber",
    "total_protein",
    "total_fat",
    "carb_ratio",          # карб / нийт калори × 400
    "sugar_fiber_ratio",   # чихэр / (эслэг + 1)
    "glycemic_load",       # net_carbs × carb_ratio / 100  (GI proxy)
    "net_carbs",           # total_carbs - total_fiber  (шингэгддэг карб)
    "protein_fat_ratio",   # protein / (fat + 1)  (макро тэнцвэр)
]


# ─────────────────────────────────────────────────────────────
# 1. Монгол хоолны датабаз үүсгэх
# ─────────────────────────────────────────────────────────────
def build_mongolian_foods() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    def to_float(val):
        try:
            return float(str(val).replace(" g", "").replace(" mg", "").replace(" mcg", "").strip())
        except Exception:
            return 0.0

    for col in ["calories", "carbohydrate", "sugars", "fiber", "protein", "fat"]:
        df[col] = df[col].apply(to_float)

    keywords = [
        "beef", "lamb", "mutton", "rice", "noodle", "milk", "tea", "bread",
        "yogurt", "cream", "butter", "flour", "wheat", "pasta",
        "potato", "carrot", "onion", "cabbage", "egg", "sugar", "oil",
        "chicken", "pork", "oat", "barley", "soup", "broth",
        "cookie", "cake", "cheese",
    ]
    mask = df["name"].str.lower().str.contains("|".join(keywords), na=False)
    foods = (
        df[mask][["name", "calories", "carbohydrate", "sugars", "fiber", "protein", "fat"]]
        .drop_duplicates(subset="name")
        .reset_index(drop=True)
    )
    foods = foods[foods["calories"] > 0].reset_index(drop=True)

    foods.to_csv(FOODS_OUT, index=False)
    print(f"  Монгол хоолны датабаз: {len(foods)} хоол → {FOODS_OUT.name}")
    return foods


# ─────────────────────────────────────────────────────────────
# 2. Синтетик сургалтын датасет үүсгэх
#    Эх сурвалж: ADA 2024, Nathan et al. 2008, WHO
# ─────────────────────────────────────────────────────────────
def generate_training_data(foods: pd.DataFrame, n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Бодит монгол хоолны датабазаас санамсаргүй өдрийн хоолны
    комбинаци үүсгэж, нон-линеар физиологийн томьёогоор
    glucose болон HbA1c тооцоолно.

    Glucose томьёоны нон-линеар элементүүд:
      - Эслэг карбын хагасыг нөхцөлт бууруулдаг (saturation effect)
      - Чихэр нь нийлмэл карбаас хурдан шингэдэг (1.4x weight)
      - Тос идэлтийн хурдыг удаашруулдаг тул оргилыг бууруулна
      - Гетероскедастик шуугиан: glucose-ийн 7% хэлбэлзэл
    """
    rng = np.random.default_rng(seed)
    records = []

    for _ in range(n):
        n_items = rng.integers(3, 8)
        chosen  = foods.sample(n=n_items, random_state=int(rng.integers(0, 99999)))
        amounts = rng.uniform(50, 350, n_items) / 100

        total_cal    = float((chosen["calories"]     * amounts).sum())
        total_carbs  = float((chosen["carbohydrate"] * amounts).sum())
        total_sugars = float((chosen["sugars"]       * amounts).sum())
        total_fiber  = float((chosen["fiber"]        * amounts).sum())
        total_prot   = float((chosen["protein"]      * amounts).sum())
        total_fat    = float((chosen["fat"]          * amounts).sum())

        carb_ratio        = total_carbs  / (total_cal + 1e-5) * 4 * 100
        sugar_fiber_ratio = total_sugars / (total_fiber + 1)
        net_carbs         = max(0.0, total_carbs - total_fiber)
        protein_fat_ratio = total_prot  / (total_fat + 1)
        glycemic_load     = net_carbs   * carb_ratio / 100

        # ── Glucose тооцоолол (нон-линеар, ADA 2024 чиглэлд суурилсан) ──
        # Эслэг карбын нөлөөллийг нон-линеар байдлаар бууруулна:
        # fiber_factor → 1 байвал fiber-ийн нөлөөлөл бага, 0 → их
        fiber_factor   = total_fiber / (total_fiber + 15)   # 0–1 хоорондох
        carb_adjusted  = net_carbs * (1 - 0.55 * fiber_factor)

        # Чихэр нь нийлмэл карбаас хурдан шингэдэг тул 1.4x жинтэй
        sugar_extra    = total_sugars * 0.40

        # Тос шингэлтийн хурдыг удаашруулна, glucose оргилыг -0.07/г бууруулна
        fat_brake      = total_fat   * 0.07

        # Уураг гликонеогенезийг өчүүхэн нэмдэг ч ерөнхийдөө glucose бууруулна
        prot_effect    = total_prot  * 0.05

        glucose = (
            78.0
            + carb_adjusted * 0.52
            + sugar_extra
            - fat_brake
            - prot_effect
        )
        # Гетероскедастик шуугиан: glucose ихэх тусам variance ихэснэ
        noise   = rng.normal(0, max(5.0, glucose * 0.07))
        glucose = float(np.clip(glucose + noise, 60.0, 400.0))

        # ── HbA1c тооцоолол (Nathan et al. 2008 ADAG томьёо) ──
        # HbA1c = (eAG + 46.7) / 28.7
        hba1c = float(np.clip((glucose + 46.7) / 28.7 + rng.normal(0, 0.20), 4.0, 14.0))

        records.append({
            "total_calories":    round(total_cal,    2),
            "total_carbs":       round(total_carbs,  2),
            "total_sugars":      round(total_sugars, 2),
            "total_fiber":       round(total_fiber,  2),
            "total_protein":     round(total_prot,   2),
            "total_fat":         round(total_fat,    2),
            "carb_ratio":        round(carb_ratio,        4),
            "sugar_fiber_ratio": round(sugar_fiber_ratio, 4),
            "glycemic_load":     round(glycemic_load,     4),
            "net_carbs":         round(net_carbs,         2),
            "protein_fat_ratio": round(protein_fat_ratio, 4),
            "blood_glucose_level": round(glucose, 1),
            "hba1c_level":         round(hba1c,   2),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# 3. Нэг target-д загвар сургах
# ─────────────────────────────────────────────────────────────
def train_one(X_tr, X_te, y_tr, y_te, label: str) -> Pipeline:
    candidates = {
        "Ridge": Pipeline([
            ("sc", StandardScaler()),
            ("m",  Ridge(alpha=1.0)),
        ]),
        "RandomForest": Pipeline([
            ("sc", StandardScaler()),
            ("m",  RandomForestRegressor(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=3,
                max_features=0.75,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "GradientBoost": Pipeline([
            ("sc", StandardScaler()),
            ("m",  GradientBoostingRegressor(
                n_estimators=400,
                learning_rate=0.04,
                max_depth=5,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            )),
        ]),
        "XGBoost": Pipeline([
            ("sc", StandardScaler()),
            ("m",  XGBRegressor(
                n_estimators=400,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )),
        ]),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_pipe, best_r2 = None, None, -np.inf

    print(f"\n── {label} ──────────────────────────────────────")
    for name, pipe in candidates.items():
        cv = cross_val_score(pipe, X_tr, y_tr, cv=kf, scoring="r2", n_jobs=-1)
        print(f"  {name:<16} 5-Fold R²: {cv.mean():.4f} ± {cv.std():.4f}")
        if cv.mean() > best_r2:
            best_r2, best_name, best_pipe = cv.mean(), name, pipe

    print(f"  → Шилдэг: {best_name}")
    best_pipe.fit(X_tr, y_tr)

    y_pred = best_pipe.predict(X_te)
    mae    = mean_absolute_error(y_te, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
    r2     = r2_score(y_te, y_pred)
    mape   = float(np.mean(np.abs((y_te - y_pred) / (y_te + 1e-8))) * 100)

    print(f"  Test MAE  : {mae:.4f}")
    print(f"  Test RMSE : {rmse:.4f}")
    print(f"  Test R²   : {r2:.4f}")
    print(f"  Test MAPE : {mape:.2f}%")

    reg = best_pipe.named_steps["m"]
    if hasattr(reg, "feature_importances_"):
        imp = pd.Series(reg.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print(f"  Топ feature-ууд:")
        for feat, val in imp.head(5).items():
            print(f"    {feat:<22} {'█' * int(val * 40)} {val:.4f}")

    return best_pipe


# ─────────────────────────────────────────────────────────────
# 4. Жишээ таамаглал
# ─────────────────────────────────────────────────────────────
SAMPLE_DAY = {
    "total_calories":    1800,
    "total_carbs":       220,
    "total_sugars":       55,
    "total_fiber":        12,
    "total_protein":      65,
    "total_fat":          55,
    "carb_ratio":         48.9,
    "sugar_fiber_ratio":   4.2,
    "glycemic_load":      102.3,
    "net_carbs":          208.0,
    "protein_fat_ratio":   1.18,
}


def predict_sample(models: dict):
    sample = pd.DataFrame([SAMPLE_DAY])
    print("\n── Жишээ өдрийн хоол ──────────────────────────────")
    print(f"  Калори: {SAMPLE_DAY['total_calories']} ккал  |  Карб: {SAMPLE_DAY['total_carbs']}г"
          f"  |  Чихэр: {SAMPLE_DAY['total_sugars']}г  |  Эслэг: {SAMPLE_DAY['total_fiber']}г"
          f"  |  Net carbs: {SAMPLE_DAY['net_carbs']}г")
    print("\n── Таамагласан үр дүн ─────────────────────────────")
    for target, lbl in [("hba1c_level", "HbA1c (%)"),
                         ("blood_glucose_level", "Glucose (mg/dL)")]:
        if target in models:
            val = models[target].predict(sample)[0]
            print(f"  {lbl:<22}: {val:.2f}")


# ─────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--n", type=int, default=10000, help="Сургалтын датасетийн хэмжээ")
    args = parser.parse_args()

    print("=" * 60)
    print("  Хоол → HbA1c / Glucose Regression Pipeline  v2")
    print("=" * 60)

    print("\n1. Монгол хоолны датабаз үүсгэж байна...")
    foods = build_mongolian_foods()

    print(f"\n2. Сургалтын датасет үүсгэж байна ({args.n} өдөр)...")
    df = generate_training_data(foods, n=args.n)
    print(f"   Датасет: {df.shape[0]} мөр, {df.shape[1]} багана")
    print(f"   Glucose: {df['blood_glucose_level'].min():.0f}–{df['blood_glucose_level'].max():.0f}"
          f"  дундаж {df['blood_glucose_level'].mean():.1f} mg/dL")
    print(f"   HbA1c:  {df['hba1c_level'].min():.2f}–{df['hba1c_level'].max():.2f}"
          f"  дундаж {df['hba1c_level'].mean():.2f} %")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X = df[FEATURES]
    trained = {}

    for target, label in [("hba1c_level", "HbA1c (%)"),
                           ("blood_glucose_level", "Glucose (mg/dL)")]:
        y = df[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = train_one(X_tr, X_te, y_tr, y_te, label)
        trained[target] = pipe

        path = OUT_DIR / f"{target}_model.pkl"
        joblib.dump(pipe, path)
        print(f"  Хадгалагдлаа → {path}")

    if args.predict:
        predict_sample(trained)

    print("\nAmjilttai duuslaa!\n")


if __name__ == "__main__":
    main()
