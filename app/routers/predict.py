import logging

from fastapi import APIRouter, HTTPException, Query

from app import ml as ml_state
from app.config import MODEL_REGISTRY
from app.gemini import recommend_gemini
from app.ml import get_features, get_risk, run_model
from app.recommendations import recommend
from app.schemas import CompareResult, PatientData, PredictionResult

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predict"])


@router.get("/health")
def health():
    return {
        "status":        "ok",
        "loaded_models": list(ml_state.loaded_models.keys()),
        "default_model": ml_state.DEFAULT_MODEL,
        "failed_models": list(ml_state.load_errors.keys()),
    }


@router.get("/models")
def get_models():
    return {
        key: {
            "name":      info["name"],
            "available": key in ml_state.loaded_models,
            "default":   key == ml_state.DEFAULT_MODEL,
        }
        for key, info in MODEL_REGISTRY.items()
    }


@router.post("/predict", response_model=PredictionResult)
async def predict(
    data: PatientData,
    model: str = Query(default=None, description="Загварын түлхүүр: rf | lr | svm | xgb"),
):
    # Runtime-д шинэчлэгдсэн DEFAULT_MODEL-ийг ашиглана
    if model is None:
        model = ml_state.DEFAULT_MODEL

    if model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Загварын нэр буруу. Боломжит сонголтууд: {list(MODEL_REGISTRY.keys())}",
        )
    if model not in ml_state.loaded_models:
        raise HTTPException(
            status_code=503,
            detail=f"'{model}' загвар ачаалагдаагүй байна. /models endpoint-ийг шалгана уу.",
        )

    features = get_features(model, data)
    pred, prob = run_model(model, features)
    risk_level, message = get_risk(prob)

    gemini_recs = await recommend_gemini(data, risk_level, round(prob * 100, 1))
    recommendations = gemini_recs if gemini_recs else recommend(data, risk_level)

    return PredictionResult(
        prediction      = pred,
        label           = "Чихрийн шижин" if pred == 1 else "Эрүүл",
        probability     = round(prob * 100, 1),
        risk_level      = risk_level,
        message         = message,
        recommendations = recommendations,
        model_used      = MODEL_REGISTRY[model]["name"],
    )


@router.post("/compare", response_model=list[CompareResult])
def compare(data: PatientData):
    if not ml_state.loaded_models:
        raise HTTPException(status_code=503, detail="Ачаалагдсан загвар байхгүй байна.")

    results = []
    for key in ml_state.loaded_models:
        try:
            features = get_features(key, data)
            pred, prob = run_model(key, features)
            risk_level, _ = get_risk(prob)
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
