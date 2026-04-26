import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS
from app.ml import load_all_models, select_best_model
from app.routers import auth, food, predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(load_all_models)
    select_best_model()
    yield


app = FastAPI(
    title="Чихрийн шижин илрүүлэх API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(food.router)


@app.get("/", tags=["root"])
def root():
    return {"message": "Чихрийн шижин илрүүлэх API v2 ажиллаж байна"}
