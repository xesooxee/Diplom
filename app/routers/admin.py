from fastapi import APIRouter, HTTPException

from app import ml as ml_state
from app.database import create_dish, delete_dish, list_dishes
from app.food_names import display_food_name
from app.schemas import DishCreateRequest, DishResponse

router = APIRouter(prefix="/admin", tags=["admin"])


def _ingredient_with_nutrition(name: str, amount: float) -> dict:
    if ml_state.foods_df.empty:
        raise HTTPException(status_code=503, detail="Хоолны датабаз ачаалагдаагүй байна.")

    row = ml_state.foods_df[
        ml_state.foods_df["name"].str.strip().str.lower() == name.strip().lower()
    ]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"'{name}' орц хоолны датабазд олдсонгүй.")

    data = row.iloc[0]
    return {
        "name": data["name"],
        "display_name": display_food_name(data["name"], data.get("name_mn")),
        "amount": amount,
        "calories": float(data["calories"]),
        "carbohydrate": float(data["carbohydrate"]),
        "sugars": float(data["sugars"]),
        "fiber": float(data["fiber"]),
        "protein": float(data["protein"]),
        "fat": float(data["fat"]),
    }


@router.get("/dishes", response_model=list[DishResponse])
def get_dishes():
    dishes = list_dishes()
    for dish in dishes:
        for ingredient in dish.get("ingredients", []):
            ingredient["display_name"] = display_food_name(
                ingredient["name"],
                ingredient.get("name_mn") or ingredient.get("display_name"),
            )
    return dishes


@router.post("/dishes", response_model=DishResponse)
def add_dish(body: DishCreateRequest):
    ingredients = [
        _ingredient_with_nutrition(item.name, item.amount)
        for item in body.ingredients
    ]
    return create_dish(body.name, ingredients)


@router.delete("/dishes/{dish_id}")
def remove_dish(dish_id: str):
    if not delete_dish(dish_id):
        raise HTTPException(status_code=404, detail="Хоол олдсонгүй.")
    return {"ok": True}
