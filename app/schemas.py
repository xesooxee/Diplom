from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

from app.config import SMOKING_CLASSES


def calculate_bmi(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    if height_cm is None or weight_kg is None:
        return None
    h_m = height_cm / 100.0
    if h_m <= 0:
        return None
    return round(weight_kg / (h_m * h_m), 1)


class _BmiMixin(BaseModel):
    height_cm: Optional[float] = Field(default=None, ge=80,  le=250, example=165.0)
    weight_kg: Optional[float] = Field(default=None, ge=20,  le=350, example=70.0)
    bmi:       Optional[float] = Field(default=None, ge=10,  le=100, example=28.5)

    @model_validator(mode="after")
    def _derive_bmi(self):
        computed = calculate_bmi(self.height_cm, self.weight_kg)
        if computed is not None:
            self.bmi = computed
            return self
        if self.bmi is None:
            raise ValueError("bmi эсвэл (height_cm + weight_kg) заавал илгээнэ үү")
        return self


class PatientData(_BmiMixin):
    gender:              str           = Field(..., example="Female")
    age:                 float         = Field(..., ge=0,  le=120, example=45.0)
    hypertension:        Literal[0, 1] = Field(..., example=0)
    heart_disease:       Literal[0, 1] = Field(..., example=0)
    smoking_history:     str           = Field(..., example="never")
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

    prediction:      int
    label:           str
    probability:     float
    risk_level:      str
    message:         str
    recommendations: list[str]
    model_used:      str


class CompareResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_key:   str
    model_name:  str
    prediction:  int
    label:       str
    probability: float
    risk_level:  str
    error:       Optional[str] = None


# ── Auth schemas ──────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email:    EmailStr
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    email:        str


# ── Food schemas ──────────────────────────────────────────────
class FoodItem(BaseModel):
    name:   str   = Field(..., example="Beef, raw")
    amount: float = Field(..., gt=0, le=2000, example=150,
                          description="Идсэн хэмжээ (грамм)")


class FoodPredictRequest(_BmiMixin):
    foods:           list[FoodItem] = Field(..., min_length=1)
    gender:          str            = Field(..., example="Male")
    age:             float          = Field(..., ge=0, le=120, example=35.0)
    hypertension:    Literal[0, 1]  = Field(..., example=0)
    heart_disease:   Literal[0, 1]  = Field(..., example=0)
    smoking_history: str            = Field(..., example="never")
    model:           str            = Field(default="rf", example="rf")

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
            raise ValueError(f"smoking_history буруу. Зөвшөөрөгдсөн: {SMOKING_CLASSES}")
        return v


class FoodNutrition(BaseModel):
    total_calories: float
    total_carbs:    float
    total_sugars:   float
    total_fiber:    float
    total_protein:  float
    total_fat:      float


class FoodPredictResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    nutrition:         FoodNutrition
    predicted_hba1c:   float
    predicted_glucose: float
    prediction:        int
    label:             str
    probability:       float
    risk_level:        str
    message:           str
    recommendations:   list[str]
    model_used:        str


# ── Admin food schemas ────────────────────────────────────────
class DishIngredient(FoodItem):
    display_name: Optional[str] = None
    calories:     float = 0
    carbohydrate: float = 0
    sugars:       float = 0
    fiber:        float = 0
    protein:      float = 0
    fat:          float = 0


class DishCreateRequest(BaseModel):
    name:        str = Field(..., min_length=2, max_length=120, example="Цуйван")
    ingredients: list[FoodItem] = Field(..., min_length=1)


class DishResponse(BaseModel):
    id:          str
    name:        str
    ingredients: list[DishIngredient]
