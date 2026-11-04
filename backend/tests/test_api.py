import json

import pytest

from app.core import rate_limit
from app.services import inference
from app.database.base import SessionLocal
from app.models.study import AuditLog
from conftest import FakeYOLO, dicom_bytes, png_bytes


def upload(client, path, headers, content=None, name="rx.jpg", ctype="image/jpeg", **params):
    return client.post(path, headers=headers, params=params, files={"file": (name, content, ctype)})


def last_audit(action):
    # created_at tiene resolución de segundos: se filtra por acción en vez de fiarse del orden
    with SessionLocal() as db:
        return db.query(AuditLog).filter(AuditLog.action == action).order_by(AuditLog.created_at.desc()).first()


# ── salud y autenticación ───────────────────────────────────────────────

def test_health_sano_sin_yolo(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy" and body["database"] == "ok"
    assert body["models_loaded"]["yolo"] is False and body["models_loaded"]["unet"] is True


def test_health_degradado_si_falta_un_modelo_obligatorio(client, monkeypatch):
    monkeypatch.setattr(inference, "unet_model", None)
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "degraded"


def test_registro_login_y_perfil(client):
    data = {"email": "nuevo@pruebas.com", "password": "contrasena-larga", "full_name": "Usuario Nuevo"}
    assert client.post("/auth/register", json=data).status_code == 201
    assert client.post("/auth/register", json=data).status_code == 400
    assert client.post("/auth/login", json={"email": data["email"], "password": "mala"}).status_code == 401
    token = client.post("/auth/login", json={"email": data["email"], "password": data["password"]}).json()["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json()["role"] == "viewer"


def test_endpoints_protegidos(client):
    assert client.get("/studies").status_code in (401, 403)
    assert client.get("/studies", headers={"Authorization": "Bearer token-falso"}).status_code == 401


# ── predicción ──────────────────────────────────────────────────────────

def test_predict_auto_sin_yolo(client, make_user, xray_bytes):
    r = upload(client, "/predict/auto", make_user(), xray_bytes)
    assert r.status_code == 200, r.text
    body = r.json()
    assert [x["rank"] for x in body["ranking"]] == [1, 2, 3]
    assert {x["disease"] for x in body["ranking"]} == {"pneumonia", "covid", "tuberculosis"}
    probs = [x["probability_disease"] for x in body["ranking"]]
    assert probs == sorted(probs, reverse=True)
    assert body["gradcam"]["heatmap_image"] and body["segmentation"]["image_url"]
    assert body["yolo"]["available"] is False and "YOLO" not in body["conclusion"]

    audit = last_audit("predict")
    assert json.loads(audit.meta_data)["disease"] == body["gradcam"]["analyzed_disease"]


def test_predict_auto_con_yolo(client, make_user, xray_bytes, monkeypatch):
    monkeypatch.setattr(inference, "yolo_model", FakeYOLO())
    body = upload(client, "/predict/auto", make_user(), xray_bytes).json()
    assert body["yolo"]["available"] is True and body["yolo"]["total"] == 2
    assert body["yolo"]["detections"][0]["class_name"] == "Cardiomegaly"
    assert "YOLO detectó 2 hallazgo(s)" in body["conclusion"]


@pytest.mark.parametrize("disease, labels", [
    ("pneumonia", {"NORMAL", "PNEUMONIA"}),
    ("covid", {"Normal", "COVID"}),
    ("tuberculosis", {"Normal", "Tuberculosis"}),
])
def test_predict_por_enfermedad(client, make_user, disease, labels):
    r = upload(client, f"/predict/{disease}", make_user(), png_bytes((256, 256)), name="rx.png", ctype="image/png")
    assert r.status_code == 200, r.text
    result = r.json()["result"]
    assert set(result["probabilities"]) == labels and result["prediction"] in labels
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert r.json()["yolo"]["available"] is False


def test_predict_rechaza_archivo_invalido(client, make_user):
    r = upload(client, "/predict/auto", make_user(), b"no es una imagen", name="rx.png", ctype="image/png")
    assert r.status_code == 400


def test_predict_dicom(client, make_user):
    r = upload(client, "/predict/dicom", make_user(), dicom_bytes(), name="estudio.dcm", ctype="application/dicom")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "dicom" and body["dicom_metadata"]["study_date_formatted"] == "15/03/2024"
    assert body["dicom_metadata"]["patient_id"] == "P-001"
    assert body["yolo"]["available"] is False and "YOLO" not in body["conclusion"]


def test_predict_dicom_exige_extension_dcm(client, make_user):
    assert upload(client, "/predict/dicom", make_user(), dicom_bytes(), name="estudio.png").status_code == 400


def test_limite_de_peticiones(client, make_user, monkeypatch):
    monkeypatch.setitem(rate_limit.RATE_LIMITS, "studies", {"max_requests": 2, "window_seconds": 3600})
    headers = make_user()
    assert [client.get("/studies", headers=headers).status_code for _ in range(2)] == [200, 200]
    r = client.get("/studies", headers=headers)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0


# ── estudios, validación, reporte y dashboard ─────────────────────────────

@pytest.fixture
def study(client, make_user, xray_bytes):
    owner = make_user()
    body = upload(client, "/predict/auto", owner, xray_bytes).json()
    return owner, body["study_id"]


def test_cada_usuario_ve_solo_sus_estudios(client, make_user, study):
    owner, sid = study
    other, admin = make_user(), make_user("admin")
    assert [s["id"] for s in client.get("/studies", headers=owner).json()["studies"]] == [sid]
    assert client.get("/studies", headers=other).json()["total"] == 0
    assert sid in [s["id"] for s in client.get("/studies", headers=admin).json()["studies"]]
    assert client.get(f"/studies/{sid}", headers=other).status_code == 403
    assert client.get("/studies/no-existe", headers=admin).status_code == 404


def test_ver_un_estudio_queda_auditado(client, study):
    owner, sid = study
    r = client.get(f"/studies/{sid}", headers=owner)
    assert r.status_code == 200 and r.json()["predictions"][0]["model_version"] == "ensemble_v2.0"
    assert json.loads(last_audit("view").meta_data) == {"study_id": sid}


def test_solo_radiologos_validan(client, make_user, study):
    owner, sid = study
    pred_id = client.get(f"/studies/{sid}", headers=owner).json()["predictions"][0]["id"]
    params = {"validation_result": "confirmado", "radiologist_notes": "sin comentarios"}
    assert client.put(f"/predictions/{pred_id}/validate", params=params, headers=owner).status_code == 403
    r = client.put(f"/predictions/{pred_id}/validate", params=params, headers=make_user("radiologist"))
    assert r.status_code == 200
    pred = client.get(f"/studies/{sid}", headers=owner).json()["predictions"][0]
    assert pred["is_validated"] and pred["validation_result"] == "confirmado"


def test_reporte_pdf(client, make_user, study):
    owner, sid = study
    r = client.get(f"/studies/{sid}/report", headers=owner)
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF" and len(r.content) > 10_000
    assert client.get(f"/studies/{sid}/report", headers=make_user()).status_code == 403


def test_dashboard(client, study, monkeypatch):
    owner, _ = study
    body = client.get("/dashboard/stats", headers=owner).json()
    assert body["kpis"]["total_studies"] == 1 and len(body["timeline"]) == 30
    assert {d["disease"] for d in body["disease_distribution"]} == {"pneumonia", "covid", "tuberculosis"}
    assert body["system_metrics"]["yolo_map50"] is None
    monkeypatch.setattr(inference, "yolo_model", FakeYOLO())
    assert client.get("/dashboard/stats", headers=owner).json()["system_metrics"]["yolo_map50"] == "39.3%"
