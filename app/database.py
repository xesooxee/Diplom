import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_USERS_FILE = Path(__file__).resolve().parent.parent / "users.json"


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
