from fastapi import APIRouter, HTTPException

from app.database import create_user, find_user
from app.schemas import AuthResponse, LoginRequest, RegisterRequest
from app.security import create_token, hash_password, verify_password

router = APIRouter(tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def register(body: RegisterRequest):
    if find_user(body.email):
        raise HTTPException(status_code=409, detail="Энэ имэйл аль хэдийн бүртгэлтэй байна.")
    create_user(body.email, hash_password(body.password))
    return AuthResponse(access_token=create_token(body.email), email=body.email)


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest):
    user = find_user(body.email)
    if not user or not verify_password(body.password, user["password"]):
        raise HTTPException(status_code=401, detail="Имэйл эсвэл нууц үг буруу байна.")
    return AuthResponse(access_token=create_token(body.email), email=body.email)
