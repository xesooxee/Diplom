import json
import logging

import joblib
import numpy as np
import pandas as pd

from app.config import BASE_DIR, FOOD_MODEL_PATHS, FOODS_CSV, MODEL_REGISTRY, SMOKING_CLASSES

logger = logging.getLogger(__name__)

loaded_models:   dict = {}
load_errors:     dict = {}
food_models:     dict = {}
_label_encoders: dict = {}
foods_df: pd.DataFrame = pd.DataFrame()
DEFAULT_MODEL = "rf"
cached_metrics:  dict = {}


def load_all_models() -> None:
    global foods_df

    for key, info in MODEL_REGISTRY.items():
        try:
            if info["path"].exists():
                loaded_models[key] = joblib.load(str(info["path"]))
                logger.info("Loaded model: %s", key)
            else:
                load_errors[key] = "Model file not found"
                logger.warning("Model file not found: %s → %s", key, info["path"])
        except Exception as e:
            load_errors[key] = "Failed to load model"
            logger.error("Error loading model '%s': %s", key, e)

    for target, path in FOOD_MODEL_PATHS.items():
        try:
            if path.exists():
                food_models[target] = joblib.load(str(path))
                logger.info("Loaded food model: %s", target)
            else:
                logger.warning("Food model not found: %s", path)
        except Exception as e:
            logger.error("Error loading food model '%s': %s", target, e)

    if FOODS_CSV.exists():
        foods_df = pd.read_csv(FOODS_CSV)
        logger.info("Loaded foods DB: %d items", len(foods_df))
    else:
        logger.warning("mongolian_foods.csv олдсонгүй")

    global _label_encoders
    enc_path = BASE_DIR / "model" / "encoders.pkl"
    if enc_path.exists():
        _label_encoders = joblib.load(str(enc_path))
        logger.info("LabelEncoder-ууд ачаалагдлаа: %s", list(_label_encoders.keys()))
    else:
        logger.warning("encoders.pkl олдсонгүй — гар кодлолт (fallback) ашиглана")


def select_best_model() -> None:
    global DEFAULT_MODEL, cached_metrics
    metrics_path = BASE_DIR / "model" / "evaluation" / "metrics.json"
    if not metrics_path.exists():
        return
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        cached_metrics = metrics
        available = {k: v for k, v in metrics.items() if k in loaded_models}
        if not available:
            return
        best = max(available, key=lambda k: available[k].get("ROC-AUC", 0))
        DEFAULT_MODEL = best
        logger.info("Best model auto-selected: %s (ROC-AUC=%.4f)", best, available[best]["ROC-AUC"])
    except Exception as e:
        logger.warning("metrics.json уншихад алдаа: %s", e)


def _encode(data) -> np.ndarray:
    assert data.bmi is not None
    if _label_encoders:
        gender_enc  = int(_label_encoders["gender"].transform([data.gender])[0])
        smoking_enc = int(_label_encoders["smoking_history"].transform([data.smoking_history])[0])
    else:
        gender_enc  = {"Female": 0, "Male": 1, "Other": 2}.get(data.gender, 1)
        smoking_enc = SMOKING_CLASSES.index(data.smoking_history)
    return np.array([[
        gender_enc, data.age, data.hypertension, data.heart_disease,
        smoking_enc, data.bmi, data.hba1c_level, data.blood_glucose_level,
    ]])


def _encode_xgb(data) -> np.ndarray:
    base = _encode(data)[0]
    glucose_hba1c_ratio = data.blood_glucose_level / (data.hba1c_level + 1e-5)
    age_bmi_interaction = data.age * data.bmi
    high_risk_comorb    = float(data.hypertension + data.heart_disease)
    metabolic_score     = data.bmi + data.hba1c_level * 10 + data.blood_glucose_level / 10
    return np.array([[*base, glucose_hba1c_ratio, age_bmi_interaction,
                      high_risk_comorb, metabolic_score]])


def get_features(key: str, data) -> np.ndarray:
    return _encode_xgb(data) if key == "xgb" else _encode(data)


def run_model(key: str, features: np.ndarray) -> tuple[int, float]:
    obj = loaded_models[key]
    if isinstance(obj, dict):
        model     = obj["model"]
        threshold = obj.get("best_threshold", obj.get("threshold", 0.5))
        prob = float(model.predict_proba(features)[0][1])
        pred = int(prob >= threshold)
    else:
        prob = float(obj.predict_proba(features)[0][1])
        pred = int(prob >= 0.5)
    return pred, prob


def get_risk(probability: float) -> tuple[str, str]:
    if probability < 0.3:
        return "Бага", "Таны чихрийн шижины эрсдэл бага байна. Эрүүл амьдралын хэв маягаа хадгалаарай."
    if probability < 0.6:
        return "Дунд", "Дунд зэргийн эрсдэлтэй байна. Эмчтэй зөвлөлдөхийг зөвлөж байна."
    return "Өндөр", "Өндөр эрсдэлтэй байна. Яаралтай эмчид хандана уу."
