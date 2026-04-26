import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
SECRET_KEY: str = os.getenv("SECRET_KEY", "glucocare-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 хоног

MONGO_URI: str = os.getenv("MONGO_URI", "")
ALLOWED_ORIGINS: list[str] = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:3030,http://localhost:5173",
    ).split(",")
]
ENV: str = os.getenv("ENV", "production")

SMOKING_CLASSES = ["No Info", "current", "ever", "former", "never", "not current"]

MODEL_REGISTRY: dict = {
    "rf":  {"name": "Random Forest",       "path": BASE_DIR / "model" / "rf"  / "pipeline.pkl"},
    "lr":  {"name": "Logistic Regression", "path": BASE_DIR / "model" / "lr"  / "pipeline.pkl"},
    "svm": {"name": "SVM",                 "path": BASE_DIR / "model" / "svm" / "pipeline.pkl"},
    "xgb": {"name": "XGBoost",             "path": BASE_DIR / "model" / "xgb" / "pipeline.pkl"},
}

FOOD_MODEL_PATHS: dict = {
    "hba1c_level":         BASE_DIR / "model" / "food_regression" / "hba1c_level_model.pkl",
    "blood_glucose_level": BASE_DIR / "model" / "food_regression" / "blood_glucose_level_model.pkl",
}

FOODS_CSV = BASE_DIR / "data" / "mongolian_foods.csv"
