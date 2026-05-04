import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app import ml as ml_state
from app.config import MODEL_REGISTRY
from app.food_names import display_food_name
from app.ml import get_features, get_risk, run_model
from app.recommendations import recommend
from app.schemas import (
    FoodItem, FoodNutrition, FoodPredictRequest, FoodPredictResult, PatientData,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["food"])

FOOD_FEATURES = [
    "total_calories", "total_carbs", "total_sugars",
    "total_fiber", "total_protein", "total_fat",
    "carb_ratio", "sugar_fiber_ratio",
    "glycemic_load", "net_carbs", "protein_fat_ratio",
]


def _calc_nutrition(foods: list[FoodItem]) -> dict:
    if ml_state.foods_df.empty:
        raise HTTPException(status_code=503, detail="Хоолны датабаз ачаалагдаагүй байна.")

    totals = {"calories": 0.0, "carbohydrate": 0.0, "sugars": 0.0,
              "fiber": 0.0, "protein": 0.0, "fat": 0.0}

    for item in foods:
        row = ml_state.foods_df[
            ml_state.foods_df["name"].str.strip().str.lower() == item.name.strip().lower()
        ]
        if row.empty:
            raise HTTPException(
                status_code=404,
                detail=f"'{item.name}' хоол датабазд олдсонгүй. /foods endpoint-ээс жагсаалт харна уу.",
            )
        factor = item.amount / 100.0
        for col in totals:
            totals[col] += float(row.iloc[0][col]) * factor

    carb_ratio        = totals["carbohydrate"] / (totals["calories"] + 1e-5) * 400
    sugar_fiber_ratio = totals["sugars"] / (totals["fiber"] + 1)
    net_carbs         = max(0.0, totals["carbohydrate"] - totals["fiber"])
    glycemic_load     = net_carbs * carb_ratio / 100
    protein_fat_ratio = totals["protein"] / (totals["fat"] + 1)

    return {
        "total_calories":    round(totals["calories"],     1),
        "total_carbs":       round(totals["carbohydrate"], 1),
        "total_sugars":      round(totals["sugars"],       1),
        "total_fiber":       round(totals["fiber"],        1),
        "total_protein":     round(totals["protein"],      1),
        "total_fat":         round(totals["fat"],          1),
        "carb_ratio":        round(carb_ratio,             4),
        "sugar_fiber_ratio": round(sugar_fiber_ratio,      4),
        "glycemic_load":     round(glycemic_load,          4),
        "net_carbs":         round(net_carbs,              2),
        "protein_fat_ratio": round(protein_fat_ratio,      4),
    }


@router.get("/foods")
def get_foods(
    search: Optional[str] = Query(default=None, description="Хоолны нэрээр хайх"),
    limit: int = Query(default=5000, ge=1, le=10000, description="Буцаах мөрийн дээд тоо"),
):
    if ml_state.foods_df.empty:
        raise HTTPException(status_code=503, detail="Хоолны датабаз ачаалагдаагүй байна.")

    df = ml_state.foods_df
    df = df.copy()
    df["_display_name"] = df.apply(
        lambda row: display_food_name(row["name"], row.get("name_mn")),
        axis=1,
    )
    if search:
        q = search.lower()
        df = df[
            df["name"].str.lower().str.contains(q, na=False)
            | df["_display_name"].str.lower().str.contains(q, na=False)
        ]

    return {
        "total": len(df),
        "foods": [
            {
                **row,
                "display_name": row["_display_name"],
            }
            for row in df[["name", "_display_name", "calories", "carbohydrate", "sugars", "fiber", "protein", "fat"]]
                .head(limit)
                .to_dict(orient="records")
        ],
    }


@router.post("/predict/food", response_model=FoodPredictResult)
def predict_from_food(req: FoodPredictRequest):
    if not ml_state.food_models:
        raise HTTPException(
            status_code=503,
            detail="Хоолны регрессийн загвар ачаалагдаагүй. Эхлээд python model/food_regression.py ажиллуулна уу.",
        )
    if req.model not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Загварын нэр буруу: {list(MODEL_REGISTRY.keys())}")
    if req.model not in ml_state.loaded_models:
        raise HTTPException(status_code=503, detail=f"'{req.model}' загвар ачаалагдаагүй байна.")

    nutrition = _calc_nutrition(req.foods)
    feat_df   = pd.DataFrame([nutrition])[FOOD_FEATURES]

    predicted_hba1c   = round(float(ml_state.food_models["hba1c_level"].predict(feat_df)[0]), 2)
    predicted_glucose = round(float(ml_state.food_models["blood_glucose_level"].predict(feat_df)[0]), 1)

    predicted_hba1c   = max(4.0,  min(14.0,  predicted_hba1c))
    predicted_glucose = max(60.0, min(400.0, predicted_glucose))

    patient = PatientData(
        gender              = req.gender,
        age                 = req.age,
        hypertension        = req.hypertension,
        heart_disease       = req.heart_disease,
        smoking_history     = req.smoking_history,
        bmi                 = req.bmi,
        hba1c_level         = predicted_hba1c,
        blood_glucose_level = predicted_glucose,
    )
    features          = get_features(req.model, patient)
    pred, prob        = run_model(req.model, features)
    risk_level, message = get_risk(prob)

    return FoodPredictResult(
        nutrition          = FoodNutrition(**{k: nutrition[k] for k in
                                              ["total_calories", "total_carbs", "total_sugars",
                                               "total_fiber", "total_protein", "total_fat"]}),
        predicted_hba1c    = predicted_hba1c,
        predicted_glucose  = predicted_glucose,
        prediction         = pred,
        label              = "Чихрийн шижин" if pred == 1 else "Эрүүл",
        probability        = round(prob * 100, 1),
        risk_level         = risk_level,
        message            = message,
        recommendations    = recommend(patient, risk_level),
        model_used         = MODEL_REGISTRY[req.model]["name"],
    )
