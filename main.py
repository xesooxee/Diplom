import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

# ── Загваруудын бүртгэл ───────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

MODEL_REGISTRY = {
    "rf": {
        "name": "Random Forest",
        "path": BASE_DIR / "model" / "rf" / "pipeline.pkl",
    },
    "lr": {
        "name": "Logistic Regression",
        "path": BASE_DIR / "model" / "lr" / "pipeline.pkl",
    },
    "svm": {
        "name": "SVM",
        "path": BASE_DIR / "model" / "svm" / "pipeline.pkl",
    },
    "xgb": {
        "name": "XGBoost",
        "path": BASE_DIR / "model" / "xgb" / "pipeline.pkl",
    },
}

DEFAULT_MODEL = "rf"
SMOKING_CLASSES = ["No Info", "current", "ever", "former", "never", "not current"]

loaded_models: dict = {}
load_errors: dict = {}


def _load_all_models() -> None:
    """Бүртгэлтэй бүх загварыг ачаалахыг оролдоно."""
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # asyncio.to_thread ашиглан event loop-ийг блоклохгүйгээр ачаална
    await asyncio.to_thread(_load_all_models)
    yield


app = FastAPI(
    title="Чихрийн шижин илрүүлэх API",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────
_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:3030,http://localhost:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Схемүүд ──────────────────────────────────────────────────
class PatientData(BaseModel):
    gender:              str           = Field(..., example="Female")
    age:                 float         = Field(..., ge=0,  le=120, example=45.0)
    hypertension:        Literal[0, 1] = Field(..., example=0)
    heart_disease:       Literal[0, 1] = Field(..., example=0)
    smoking_history:     str           = Field(..., example="never")
    bmi:                 float         = Field(..., ge=10, le=100, example=28.5)
    hba1c_level:         float         = Field(..., ge=0,  le=20,  example=6.5)
    blood_glucose_level: float         = Field(..., ge=0,  le=500, example=140)

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in ("Female", "Male"):
            raise ValueError("gender утга 'Female' эсвэл 'Male' байх ёстой")
        return v

    @field_validator("smoking_history")
    @classmethod
    def validate_smoking(cls, v: str) -> str:
        if v not in SMOKING_CLASSES:
            raise ValueError(f"smoking_history утга буруу. Зөвшөөрөгдсөн: {SMOKING_CLASSES}")
        return v


class PredictionResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction:       int
    label:            str
    probability:      float        # 0–100 хувиар илэрхийлнэ
    risk_level:       str
    message:          str
    recommendations:  list[str]    # Өвчтөний үзүүлэлтэд тохирсон зөвлөмжүүд
    model_used:       str


class CompareResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_key:   str
    model_name:  str
    prediction:  int
    label:       str
    probability: float    # 0–100 хувиар илэрхийлнэ
    risk_level:  str
    error:       Optional[str] = None


# ── Туслах функцүүд ──────────────────────────────────────────
def _encode(data: PatientData) -> np.ndarray:
    """RF / LR / SVM загваруудад зориулсан 8 feature кодлолт."""
    gender_enc  = 0 if data.gender == "Female" else 1
    smoking_enc = SMOKING_CLASSES.index(data.smoking_history)
    return np.array([[
        gender_enc,
        data.age,
        data.hypertension,
        data.heart_disease,
        smoking_enc,
        data.bmi,
        data.hba1c_level,
        data.blood_glucose_level,
    ]])


def _encode_xgb(data: PatientData) -> np.ndarray:
    """XGBoost загварт зориулсан 12 feature кодлолт (8 суурь + 4 engineered)."""
    base = _encode(data)[0]
    glucose_hba1c_ratio = data.blood_glucose_level / (data.hba1c_level + 1e-5)
    age_bmi_interaction = data.age * data.bmi
    high_risk_comorb    = float(data.hypertension + data.heart_disease)
    metabolic_score     = data.bmi + data.hba1c_level * 10 + data.blood_glucose_level / 10
    return np.array([[*base, glucose_hba1c_ratio, age_bmi_interaction,
                      high_risk_comorb, metabolic_score]])


def _get_features(key: str, data: PatientData) -> np.ndarray:
    return _encode_xgb(data) if key == "xgb" else _encode(data)


def _risk(probability: float) -> tuple[str, str]:
    if probability < 0.3:
        return "Бага", "Таны чихрийн шижины эрсдэл бага байна. Эрүүл амьдралын хэв маягаа хадгалаарай."
    if probability < 0.6:
        return "Дунд", "Дунд зэргийн эрсдэлтэй байна. Эмчтэй зөвлөлдөхийг зөвлөж байна."
    return "Өндөр", "Өндөр эрсдэлтэй байна. Яаралтай эмчид хандана уу."


def _recommend(data: PatientData, risk_level: str) -> list[str]:
    """
    Өвчтөний бодит үзүүлэлт болон эрсдэлийн түвшинд тулгуурлан
    судалгаанд суурилсан зөвлөмж жагсаалт буцаана.

    Эх сурвалж:
      - ADA (American Diabetes Association) Standards of Care 2024
      - WHO Global Report on Diabetes
      - CDC Diabetes Prevention Program
    """
    tips: list[str] = []

    # ── HbA1c (Гликозилжсан гемоглобин) ──────────────────────
    # ADA: ≥6.5% → чихрийн шижингийн оноши, 5.7–6.4% → пред-диабет
    if data.hba1c_level >= 9.0:
        tips.append(
            "HbA1c таны {:.1f}% байна — маш өндөр. Яаралтай эмчилгээ шаардлагатай. "
            "Инсулин эмчилгээний талаар эндокринологич эмчтэй зөвлөлдөнө үү.".format(data.hba1c_level)
        )
    elif data.hba1c_level >= 6.5:
        tips.append(
            "HbA1c таны {:.1f}% байна — чихрийн шижингийн оношлогооны хэмжээнд (≥6.5%) хүрсэн. "
            "3 сар тутамд хянаж, эмчийн хяналтан дор эмчилгээ авна уу.".format(data.hba1c_level)
        )
    elif data.hba1c_level >= 5.7:
        tips.append(
            "HbA1c таны {:.1f}% байна — пред-диабет бүсэд (5.7–6.4%) орж байна. "
            "Амьдралын хэв маягаа өөрчилснөөр жинхэнэ чихрийн шижин болохоос сэргийлэх боломжтой.".format(data.hba1c_level)
        )

    # ── Цусан дахь сахар ─────────────────────────────────────
    # ADA: ≥126 mg/dL өлөн үед → диабет, 100–125 → пред-диабет, 2 цагийн дараа ≥200 → диабет
    if data.blood_glucose_level >= 200:
        tips.append(
            "Цусан дахь сахар {:.0f} mg/dL — диабетийн онош тавих хэмжээнээс (≥200) давсан. "
            "Шингэн ихэд уух, байнга шээх, ядрах шинж тэмдэг байгаа эсэхийг ажиглаарай.".format(data.blood_glucose_level)
        )
    elif data.blood_glucose_level >= 126:
        tips.append(
            "Цусан дахь сахар {:.0f} mg/dL — диабетийн оношлогооны хэмжээнд (≥126 mg/dL) байна. "
            "Цагаан будаа, талх, чихэртэй ундааны хэрэглээг хязгаарлана уу.".format(data.blood_glucose_level)
        )
    elif data.blood_glucose_level >= 100:
        tips.append(
            "Цусан дахь сахар {:.0f} mg/dL — пред-диабет бүсэд (100–125 mg/dL) байна. "
            "Цардуулын хэрэглээг бууруулж, шинжилгээгээ тогтмол хийлгэнэ үү.".format(data.blood_glucose_level)
        )

    # ── BMI (Биеийн жингийн индекс) ──────────────────────────
    # WHO: ≥40 → III зэргийн таргалалт, 30–39.9 → II зэрэг, 25–29.9 → илүүдэл жин
    if data.bmi >= 40:
        tips.append(
            "BMI таны {:.1f} — III зэргийн таргалалт (≥40). "
            "Жингийн асуудал чихрийн шижингийн хамгийн том эрсдэлт хүчин зүйл. "
            "Тэжээлийн мэргэжилтэн + физик эмчилгээний мэргэжилтэнтэй зөвлөлдөнө үү. "
            "BMI 40+ үед бариатрик мэс засал нь чихрийн шижинг бүрэн арилгах боломж өгдгийг судалгаа харуулж байна.".format(data.bmi)
        )
    elif data.bmi >= 30:
        tips.append(
            "BMI таны {:.1f} — таргалалт (30–39.9). "
            "Жингийнхаа 5–10%-ийг бууруулах нь чихрийн шижингийн эрсдэлийг 50–58%-иар бууруулдаг "
            "(CDC Diabetes Prevention Program судалгаа).".format(data.bmi)
        )
    elif data.bmi >= 25:
        tips.append(
            "BMI таны {:.1f} — илүүдэл жин (25–29.9). "
            "Долоо хоногт 150+ минут дунд эрчимтэй дасгал хийж жингээ хэвийн хэмжээнд (18.5–24.9) авчрахыг зорино уу.".format(data.bmi)
        )

    # ── Нас ──────────────────────────────────────────────────
    # ADA: 45+ насаас скрининг хийлгэхийг зөвлөдөг; 65+ насанд гипогликеми аюул нэмэгддэг
    if data.age >= 65:
        tips.append(
            "{:.0f} насны хүнд чихрийн шижин байгаа тул гипогликеми (сахар хэт буурах) эрсдэлд анхаарна уу. "
            "Эмчийн зааснаар жилд нэгээс доошгүй удаа нүд, бөөр, мэдрэлийн нарийн шинжилгээ хийлгэнэ үү. "
            "Өдөр бүр алхах дасгал (30 мин) нь насан туршдаа хамгийн аюулгүй бөгөөд үр дүнтэй арга.".format(data.age)
        )
    elif data.age >= 45:
        tips.append(
            "{:.0f} нас чихрийн шижингийн эрсдэлт бүлэгт (≥45) хамаарна. "
            "Жил бүр цусны сахар, HbA1c шинжилгээ хийлгэхийг ADA зөвлөж байна.".format(data.age)
        )

    # ── Өндөр цусны даралт ───────────────────────────────────
    if data.hypertension == 1:
        tips.append(
            "Өндөр цусны даралт + чихрийн шижин хосолсон тохиолдолд зүрх судасны өвчний эрсдэл 2 дахин нэмэгддэг. "
            "Давсны хэрэглээг өдөрт 5г-аас доош барих, цусны даралтаа гэрт өдөр бүр хэмжих, "
            "АД-ыг 130/80 mmHg-аас доош байлгахыг зорино уу."
        )

    # ── Зүрхний өвчин ────────────────────────────────────────
    if data.heart_disease == 1:
        tips.append(
            "Зүрхний өвчин байгаа тул чихрийн шижин хүндрэхэд зүрх судасны эрсдэл эрс нэмэгдэнэ. "
            "Метформин эмийг зүрхний өвчтэй өвчтөнд ихэвчлэн аюулгүй гэж үздэг — эмчтэй зөвлөлдөнө үү. "
            "Кардиологич + эндокринологич эмч хамтарч хянах нь тохиромжтой."
        )

    # ── Тамхи ────────────────────────────────────────────────
    if data.smoking_history == "current":
        tips.append(
            "Тамхи татах нь чихрийн шижингийн эрсдэлийг 30–40% нэмэгдүүлдэг (WHO). "
            "Тамхинаас гарснаар HbA1c дунджаар 0.5% буурдаг судалгаа байна. "
            "Тамхи татахаас гарах хөтөлбөрт хамрагдахыг зөвлөж байна."
        )
    elif data.smoking_history in ("ever", "former"):
        tips.append(
            "Өмнө нь тамхи татаж байсан нь одоо ч эрсдэлд нөлөөлж байна. "
            "Тамхинаас гарсаар байгаа бол цусны эргэлт аажмаар сайжирна."
        )

    # ── Хоол хүнс ────────────────────────────────────────────
    # Эрсдэлийн түвшинд тулгуурлан хоолны зөвлөмж
    if risk_level == "Өндөр":
        tips.append(
            "Хоолны дэглэм (ADA 2024): Нийт калорийн 45–60%-ийг нарийн ширхэгт ногооноос авна уу. "
            "Цагаан будаа → Хүрэн будаа, цагаан талх → Бүтэн буудайн талхаар солино уу. "
            "Чихэртэй ундаа, шоколад, боовыг бүрэн хасна уу. "
            "Загас, тахианы мах, буурцаг, тофуг уургийн эх үүсвэр болгоно уу."
        )
    elif risk_level == "Дунд":
        tips.append(
            "Хоолны дэглэм: Хүнсний ногоо, жимс, бүтэн үр тарианы хоолыг ихэсгэнэ үү. "
            "Боловсруулсан хоол, элсэн чихрийн хэрэглээг хязгаарлана уу."
        )

    # ── Дасгал ───────────────────────────────────────────────
    if risk_level == "Өндөр":
        tips.append(
            "Биеийн хөдөлгөөн (ADA зөвлөмж): Долоо хоногт 150+ минут дунд эрчимтэй дасгал хийнэ үү. "
            "Алхах, усанд сэлэх, дугуй унах нь өндөр BMI-тай хүмүүст хамгийн аюулгүй. "
            "Хүч чадлын дасгал (жин өргөх) долоо хоногт 2–3 удаа хийснээр инсулины мэдрэмжийг сайжруулна."
        )

    # ── Эмнэлгийн хяналт ─────────────────────────────────────
    if risk_level == "Өндөр":
        tips.append(
            "Эмнэлгийн хяналт: 3 сар тутамд HbA1c, жилд нэг удаа бөөрний үйл ажиллагаа (креатинин, альбумин), "
            "нүдний торлог бүрхэвчийн шинжилгээ хийлгэнэ үү. "
            "Гэрийн глюкометрээр өглөө өлөн үедээ болон хоол идсэний 2 цагийн дараа хэмжих нь тохиромжтой."
        )

    return tips


def _run_model(key: str, features: np.ndarray) -> tuple[int, float]:
    """Загвараар таамаглал хийж (prediction, probability 0-1) буцаана."""
    obj = loaded_models[key]

    # XGBoost dict хэлбэр: {"model": XGBClassifier, "best_threshold": float, ...}
    if isinstance(obj, dict):
        model     = obj["model"]
        threshold = obj.get("best_threshold", obj.get("threshold", 0.5))
        prob = float(model.predict_proba(features)[0][1])
        pred = int(prob >= threshold)
    else:
        # sklearn Pipeline (RF, LR, SVM)
        prob = float(obj.predict_proba(features)[0][1])
        pred = int(prob >= 0.5)

    return pred, prob


# ── Endpoints ────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Чихрийн шижин илрүүлэх API v2 ажиллаж байна"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_models": list(loaded_models.keys()),
        "default_model": DEFAULT_MODEL,
        "failed_models": list(load_errors.keys()),
    }


@app.get("/models")
def get_models():
    """Бүртгэлтэй загваруудын жагсаалт ба тэдгээрийн ачааллагдсан эсэх."""
    return {
        key: {
            "name":      info["name"],
            "available": key in loaded_models,
            "default":   key == DEFAULT_MODEL,
        }
        for key, info in MODEL_REGISTRY.items()
    }


@app.post("/predict", response_model=PredictionResult)
def predict(
    data: PatientData,
    model: str = Query(default=DEFAULT_MODEL, description="Загварын түлхүүр: rf | lr | svm | xgb"),
):
    """Сонгосон загвараар чихрийн шижины эрсдэл тооцоолно."""
    if model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Загварын нэр буруу. Боломжит сонголтууд: {list(MODEL_REGISTRY.keys())}",
        )
    if model not in loaded_models:
        raise HTTPException(
            status_code=503,
            detail=f"'{model}' загвар ачаалагдаагүй байна. /models endpoint-ийг шалгана уу.",
        )

    features = _get_features(model, data)
    pred, prob = _run_model(model, features)
    risk_level, message = _risk(prob)

    return PredictionResult(
        prediction      = pred,
        label           = "Чихрийн шижин" if pred == 1 else "Эрүүл",
        probability     = round(prob * 100, 1),
        risk_level      = risk_level,
        message         = message,
        recommendations = _recommend(data, risk_level),
        model_used      = MODEL_REGISTRY[model]["name"],
    )


@app.post("/compare", response_model=list[CompareResult])
def compare(data: PatientData):
    """Ачаалагдсан бүх загваруудаар нэгэн зэрэг таамаглал хийж харьцуулна."""
    if not loaded_models:
        raise HTTPException(status_code=503, detail="Ачаалагдсан загвар байхгүй байна.")

    results = []
    for key in loaded_models:
        try:
            features = _get_features(key, data)
            pred, prob = _run_model(key, features)
            risk_level, _ = _risk(prob)
            results.append(CompareResult(
                model_key   = key,
                model_name  = MODEL_REGISTRY[key]["name"],
                prediction  = pred,
                label       = "Чихрийн шижин" if pred == 1 else "Эрүүл",
                probability = round(prob * 100, 1),
                risk_level  = risk_level,
            ))
        except Exception as e:
            logger.error("Model '%s' inference failed: %s", key, e)
            results.append(CompareResult(
                model_key   = key,
                model_name  = MODEL_REGISTRY[key]["name"],
                prediction  = -1,
                label       = "Алдаа",
                probability = 0.0,
                risk_level  = "—",
                error       = str(e),
            ))

    return results


# ── Ажиллуулах ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    reload = os.getenv("ENV", "production") == "development"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=reload)
