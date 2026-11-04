"""API de rt26: arranque, middleware y rutas."""

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api import dashboard, predict, report, studies, system
from app.api.auth import router as auth_router
from app.core.rate_limit import _rate_limiter
from app.services import inference, records
from app.services.storage import StorageService
from config import settings, validate_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando aplicación...")

    validate_config()

    records.storage_service = StorageService()
    device = torch.device(settings.DEVICE)
    logger.info(f"   Device: {device}")

    inference.load_models(device)

    logger.success("Todos los modelos cargados")
    yield

    inference.ensemble_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _rate_limiter.cleanup()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
for module in (system, predict, studies, report, dashboard):
    app.include_router(module.router)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
