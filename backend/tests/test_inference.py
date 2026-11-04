import re
import time
from datetime import datetime

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.core import rate_limit
from app.services import inference, records
from conftest import RANDOM_ENSEMBLE, png_bytes


class FixedLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1)


def test_ensemble_promedia_las_probabilidades(monkeypatch):
    models = {"densenet": FixedLogits([2.0, 0.0]), "resnet": FixedLogits([0.0, 2.0]), "efficient": FixedLogits([0.0, 4.0])}
    monkeypatch.setitem(inference.ensemble_models, "covid", models)
    r = inference.ensemble_predict("covid", torch.zeros(1, 3, 224, 224))
    expected = np.mean([torch.softmax(m.logits, 1)[0].numpy() for m in models.values()], axis=0)
    assert np.allclose(r["probs_avg"], expected)
    assert r["predicted_class"] == 1 and r["confidence"] == pytest.approx(expected[1])
    assert set(r["detail"]) == {"densenet", "resnet", "efficient"}


def test_preprocesado():
    tensor, image = inference.preprocess_image(png_bytes((300, 200)))
    assert tuple(tensor.shape) == (1, 3, 224, 224)
    assert image.mode == "RGB" and image.size == (200, 300)


def test_interpretacion_clinica_tb_es_mas_exigente():
    assert inference.get_clinical_interpretation(0.88, "pneumonia")["level"] == "Alto"
    assert inference.get_clinical_interpretation(0.88, "tuberculosis")["level"] == "Medio"
    low = inference.get_clinical_interpretation(0.5, "covid")
    assert low["level"] == "Bajo" and low["action_required"]


def test_nombre_unico_con_fecha_ordenable():
    name = records._unique_name("gradcam")
    m = re.fullmatch(r"gradcam_(\d{8})_(\d{6})_[0-9a-f]{8}\.png", name)
    assert m and m.group(1) == datetime.utcnow().strftime("%Y%m%d")
    assert records._unique_name("x") != records._unique_name("x")


def test_rate_limiter_ventana_deslizante(monkeypatch):
    rl = rate_limit.RateLimiter()
    now = [1000.0]
    monkeypatch.setattr(rate_limit.time, "time", lambda: now[0])
    assert rl.is_allowed("k", 2, 60) == (True, 0)
    assert rl.is_allowed("k", 2, 60) == (True, 0)
    allowed, retry = rl.is_allowed("k", 2, 60)
    assert not allowed and retry == 61
    now[0] += 61
    assert rl.is_allowed("k", 2, 60) == (True, 0)
    now[0] += 3601
    rl.cleanup()
    assert "k" not in rl._store


def test_gradcam_genera_imagen_y_no_acumula_hooks(client):
    densenet = RANDOM_ENSEMBLE["pneumonia"]["densenet"]
    block = densenet.features.denseblock4
    before = (len(block._forward_hooks), len(block._backward_hooks))
    tensor, image = inference.preprocess_image(png_bytes((256, 256)))
    for _ in range(3):
        png = inference.generate_gradcam("pneumonia", tensor, image, 1)
        assert png and png[:8] == b"\x89PNG\r\n\x1a\n"
    assert (len(block._forward_hooks), len(block._backward_hooks)) == before


def test_gradcam_mapa_normalizado():
    from gradcam import create_gradcam_for_densenet
    densenet = RANDOM_ENSEMBLE["pneumonia"]["densenet"]
    cam = create_gradcam_for_densenet(densenet)
    try:
        heat = cam.generate_cam(torch.rand(1, 3, 224, 224), target_class=0)
    finally:
        cam.remove_hooks()
    assert heat.shape == (7, 7) and heat.min() >= 0 and heat.max() <= 1


def test_segmentacion_devuelve_png_512():
    from PIL import Image
    import io
    out = inference.run_segmentation(png_bytes((300, 300)))
    assert Image.open(io.BytesIO(out)).size == (512, 512)
