"""Arranque completo con los modelos reales; se omite si no se han descargado."""

import pytest
from fastapi.testclient import TestClient

import main
from conftest import SAMPLES, install_random_models
from download_models import MODELS, MODELS_DIR

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(not all((MODELS_DIR / n).exists() for n in MODELS),
                       reason="faltan los modelos: python download_models.py"),
]


def test_arranque_y_prediccion_con_los_modelos_reales(client, make_user):
    headers = make_user()
    try:
        with TestClient(main.app) as real:
            assert real.get("/health").json()["models_loaded"]["unet"] is True
            r = real.post("/predict/auto", headers=headers,
                          files={"file": ("ejemplo-2.jpg", (SAMPLES / "ejemplo-2.jpg").read_bytes(), "image/jpeg")})
            assert r.status_code == 200, r.text
            # ejemplo-2.jpg es una neumonía bacteriana del dataset de Kermany et al.
            assert r.json()["ranking"][0]["disease"] == "pneumonia"
    finally:
        # el lifespan vacía los modelos al terminar; el resto de pruebas usa los aleatorios
        install_random_models()
