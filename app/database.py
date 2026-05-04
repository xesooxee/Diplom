import json
import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

_USERS_FILE = Path(__file__).resolve().parent.parent / "users.json"
_DISHES_FILE = Path(__file__).resolve().parent.parent / "dishes.json"


def _load() -> dict:
    if not _USERS_FILE.exists():
        return {}
    with open(_USERS_FILE) as f:
        return json.load(f)


def _save(users: dict) -> None:
    with open(_USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def find_user(email: str) -> Optional[dict]:
    """Email-ээр хэрэглэгч хайна. Олдвол {"email": ..., "password": ...} буцаана."""
    users = _load()
    if email in users:
        return {"email": email, "password": users[email]}
    return None


def create_user(email: str, hashed_password: str) -> None:
    """Шинэ хэрэглэгч бүртгэнэ."""
    users = _load()
    users[email] = hashed_password
    _save(users)
    logger.info("Шинэ хэрэглэгч бүртгэгдлээ: %s", email)


def _load_dishes() -> list[dict]:
    if not _DISHES_FILE.exists():
        return []
    with open(_DISHES_FILE) as f:
        return json.load(f)


def _save_dishes(dishes: list[dict]) -> None:
    with open(_DISHES_FILE, "w") as f:
        json.dump(dishes, f, indent=2, ensure_ascii=False)


def list_dishes() -> list[dict]:
    return _load_dishes()


def create_dish(name: str, ingredients: list[dict]) -> dict:
    dishes = _load_dishes()
    dish = {
        "id": uuid4().hex,
        "name": name.strip(),
        "ingredients": ingredients,
    }
    dishes.append(dish)
    _save_dishes(dishes)
    logger.info("Шинэ хоол нэмэгдлээ: %s", name)
    return dish


def delete_dish(dish_id: str) -> bool:
    dishes = _load_dishes()
    next_dishes = [dish for dish in dishes if dish.get("id") != dish_id]
    if len(next_dishes) == len(dishes):
        return False
    _save_dishes(next_dishes)
    logger.info("Хоол устгагдлаа: %s", dish_id)
    return True
